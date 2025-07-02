import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision.ops import box_iou, box_area
import os
from PIL import Image
from pandas import read_csv
from einops import rearrange, repeat
import logging
import pickle
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEBUG = False

def group_element(underlay_id, clses, boxes):
    if underlay_id not in clses:
        element_groups = torch.arange(len(clses)).unsqueeze(1).tolist()
    else:
        mask = (clses == underlay_id)
        underlays = torch.arange(clses.shape[0])[mask]
        underlays = underlays[torch.argsort(box_area(boxes[underlays]))]
        non_underlays = torch.arange(clses.shape[0])[~mask]
        
        iou_between_underlays = box_iou(boxes[underlays], boxes[underlays]).fill_diagonal_(0).tril()
        intersection_underlays = underlays[torch.nonzero(iou_between_underlays > 0.1)]
        
        iou_between_underlays_and_non_underlays = box_iou(boxes[underlays], boxes[non_underlays])
        intersection_elements = torch.nonzero(iou_between_underlays_and_non_underlays > 0.05)
        intersection_elements[:, 0] = underlays[intersection_elements[:, 0]]
        intersection_elements[:, 1] = non_underlays[intersection_elements[:, 1]]
        
        underlays = underlays.tolist()
        non_underlays = non_underlays.tolist()
        intersection_underlays = intersection_underlays.tolist()
        intersection_elements = intersection_elements.tolist()
        
        element_groups_dict = {i: [i] for i in range(len(clses))}
        element_groups = [element_groups_dict[i] for i in range(len(clses))]
        
        # j-th element is underlayed by i-th element
        for i, j in intersection_underlays:
            try:
                if element_groups_dict[j] in element_groups:
                    element_groups_dict[i].append(element_groups_dict[j])
                    element_groups.remove(element_groups_dict[j])
            except Exception as e:
                print(e)
                print(j)
                print(element_groups)
                continue
            
        # j-th element is underlayed by i-th element
        intersection_elements_dict = {i: [] for i in underlays}
        for i, j in intersection_elements:
            intersection_elements_dict[i].append(j)
        
        walked = set()
        for i in underlays:
            for j in intersection_elements_dict[i]:
                if j in walked:
                    continue
                element_groups_dict[i].append(j)
                element_groups_dict[j].clear()
                walked.add(j)
                
        element_groups = [g for g in element_groups if g]
    
    element_groups = [g[0] if len(g) == 1 else g for g in element_groups]
    
    return element_groups

def arrange_sepoint(element_groups, clses, boxes, values):
    underlay_groups_dict = {}
    indices = []
    for g in element_groups:
        if isinstance(g, int):
            index = g * 2
            indices.extend([index, index + 1])
        elif isinstance(g, list):
            # underlayed by g[0]
            index = g[0] * 2
            indices.append(index)
            underlay_groups_dict[index] = arrange_sepoint(g[1:], clses, boxes, values)
            underlay_groups_dict[index].append(index + 1)
        else:
            raise ValueError(f"Invalid element group: {g}")
    
    indices = sorted(indices, key=lambda index: values[index])
    for k, v in underlay_groups_dict.items():
        index = indices.index(k) + 1
        indices[index:index] = v
    
    return indices

class ImageLayoutDataset(Dataset):
    def __init__(
        self,
        args,
        transform,
        split # "train", "valid" (annotated test), "test" (unannotated test)
        ):
        
        self.train = False
        self.transform = transform
        
        self.cache_dir = os.path.join(args.root, "cache", args.dataset)
        self.layout_cache_file = os.path.join(self.cache_dir, args.tokenizer_name, f"{split}_cache.pkl")
        self.image_cache_file = os.path.join(self.cache_dir, "input", f"{split}_cache.pkl")
        
        if split == "train" or split == "valid":
            self.train = True
            self.preview_dir = os.path.join(self.cache_dir, "input", "preview")
            self.canvas_dir = os.path.join(args.dataset_root, args.dataset, "image", "train", "input")
            
            if os.path.exists(self.layout_cache_file):
                logger.info(f"Loading {split} layout cache from {self.layout_cache_file}")
                with open(self.layout_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                self.dict_dataset = cache_data['dict_dataset']
                self.pp = cache_data['pp']
                abandons = cache_data['abandons']
            else:
                os.makedirs(os.path.join(self.cache_dir, args.tokenizer_name), exist_ok=True)
                
                df = read_csv(os.path.join(args.dataset_root, args.dataset, "annotation", "train.csv"))
                df = df[df["split"] == split].reset_index(drop=True)
                
                # preprocess: box<string> -> box<list>, check if box is valid, and box<xyxy> -> 2 * box<xy>
                df["box_elem"] = [eval(box) for box in df["box_elem"]]
                for i, box in enumerate(df["box_elem"]):
                    if box[2] < box[0]:
                        df.box_elem[i][2], df.box_elem[i][0] = box[0], box[2]
                    if box[3] < box[1]:
                        df.box_elem[i][3], df.box_elem[i][1] = box[1], box[3]
                
                groups = df.groupby("poster_path")
                self.dict_dataset = {}
                self.pp = df["poster_path"].unique().tolist()
                    
                underlay_id = args.class_feature.str2int('underlay')
                double_underlay_id = underlay_id * 2
                abandons = []
                for pp, subdf in tqdm(groups, desc=f"Processing {split} layout"):
                    boxes = torch.tensor(subdf["box_elem"].tolist(), dtype=torch.float32)
                    boxes[:, ::2] /= args.original_size[0]
                    boxes[:, 1::2] /= args.original_size[1]
                    clses = torch.tensor(subdf["cls_elem"].tolist(), dtype=torch.long) - 1
                    element_groups = group_element(underlay_id, clses, boxes)
                    
                    boxes = rearrange(boxes, 'n (c d) -> (n c) d', c=2)
                    clses = repeat(clses, 'n -> (n c)', c=2)
                    clses = clses * 2
                    clses[1::2] += 1
                    values = boxes[:, 1] + boxes[:, 0] * 0.01
                    
                    indices = arrange_sepoint(element_groups, clses, boxes, values)
                    
                    if DEBUG:
                        print(pp)           
                        print('element_groups:', element_groups)
                        print('indices:', indices)
                    
                    boxes = boxes[indices]
                    clses = clses[indices]
                    
                    # check if empty underlay exists
                    if 'inference' not in args.__dict__:
                        pattern_check = (clses == double_underlay_id).nonzero().squeeze(-1)
                        if len(pattern_check) > 0:
                            next_idx = pattern_check + 1
                            next_idx = next_idx[next_idx < len(clses)]
                            if any((clses[next_idx] == double_underlay_id + 1).tolist()):
                                if DEBUG:
                                    print(f"Found pattern 4,5 in {pp}: {clses}")
                                abandons.append(pp)
                    
                    self.dict_dataset[pp] = {
                        "boxes": boxes,
                        "clses": clses
                    }
                    
                # Save layout cache
                cache_data = {
                    'dict_dataset': self.dict_dataset,
                    'pp': self.pp,
                    'abandons': abandons
                }
                with open(self.layout_cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                    
            if os.path.exists(self.image_cache_file):
                logger.info(f"Loading {split} image cache from {self.image_cache_file}")
                with open(self.image_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                self.processed_images = cache_data['processed_images']
                assert len(self.processed_images) == len(self.dict_dataset), f"Unmatched length of processed layouts and images: {len(self.processed_images)} vs {len(self.dict_dataset)}"
            else:
                os.makedirs(os.path.join(self.cache_dir, "input"), exist_ok=True)
                self.processed_images = {}
                for pp in tqdm(self.pp, desc=f"Processing {split} image"):
                    image_path = os.path.join(self.canvas_dir, pp)
                    image = Image.open(image_path).convert("RGB")
                    image = self.transform(image)
                    self.processed_images[pp] = image
                
                # Save image cache
                cache_data = {
                    'pp': self.pp,
                    'processed_images': self.processed_images
                }
                with open(self.image_cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                    
            if 'inference' not in args.__dict__:
                for abd in abandons:
                    self.pp.remove(abd)
                logger.info(f"{len(abandons)} {split} samples are abandoned for empty underlay: {abandons}")
        elif split == "test":
            self.canvas_dir = os.path.join(args.dataset_root, args.dataset, "image", "test", "input")
            if os.path.exists(self.image_cache_file):
                logger.info(f"Loading {split} image cache from {self.image_cache_file}")
                with open(self.image_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                self.pp = cache_data['pp']
                self.processed_images = cache_data['processed_images']
            else:
                os.makedirs(os.path.join(self.cache_dir, "input"), exist_ok=True)
                df = read_csv(os.path.join(args.dataset_root, args.dataset, "annotation", "test.csv"))
                self.pp = df["poster_path"].unique().tolist()
                self.processed_images = {}
                for pp in tqdm(self.pp, desc=f"Processing {split} image"):
                    image_path = os.path.join(self.canvas_dir, pp)
                    image = Image.open(image_path).convert("RGB")
                    image = self.transform(image)
                    self.processed_images[pp] = image
                    
                # save image cache
                cache_data = {
                    'pp': self.pp,
                    'processed_images': self.processed_images
                }
                with open(self.image_cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
        else:
            raise ValueError(f"Invalid split: {split}")
        
    def __len__(self):
        return len(self.pp)
    
    def __getitem__(self, idx):
        pp = self.pp[idx]
        image = self.processed_images[pp]
        
        if self.train:
            boxes = self.dict_dataset[pp]["boxes"]
            clses = self.dict_dataset[pp]["clses"]
            return {
                "image": image,
                "boxes": boxes,
                "clses": clses
            }
        else:
            return {
                "image": image
            }
    
    
class ImageSupplLayoutDataset(ImageLayoutDataset):
    def __init__(
        self,
        args,
        transform,
        split # "train", "valid" (annotated test), "test" (unannotated test)
        ):
        
        super().__init__(args, transform, split)
        
        self.suppl_cache_file = os.path.join(self.cache_dir, args.suppl_type, f"{split}_cache.pkl")
        
        if split == "train" or split == "valid":
            self.suppl_dir = os.path.join(args.dataset_root, args.dataset, "image", "train", args.suppl_type)  
        elif split == "test":
            self.suppl_dir = os.path.join(args.dataset_root, args.dataset, "image", "test", args.suppl_type)
        else:
            raise ValueError(f"Invalid split: {split}")
            
        if os.path.exists(self.suppl_cache_file):
            logger.info(f"Loading {split} suppl cache from {self.suppl_cache_file}")
            with open(self.suppl_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            self.processed_suppl = cache_data['processed_suppl']
        else:
            os.makedirs(os.path.join(self.cache_dir, args.suppl_type), exist_ok=True)
            
            if split == "train" or split == "valid":
                df = read_csv(os.path.join(args.dataset_root, args.dataset, "annotation", "train.csv"))
                df = df[df["split"] == split].reset_index(drop=True)
            elif split == "test":
                df = read_csv(os.path.join(args.dataset_root, args.dataset, "annotation", "test.csv"))
            else:
                raise ValueError(f"Invalid split: {split}")
            
            self.processed_suppl = {}
            for pp in tqdm(df["poster_path"].unique(), desc=f"Processing {split} suppl"):
                suppl_path = os.path.join(self.suppl_dir, pp)
                suppl = Image.open(suppl_path).convert("RGB")
                suppl = self.transform(suppl)
                self.processed_suppl[pp] = suppl
            
            # save image cache
            cache_data = {
                'pp': df["poster_path"].unique().tolist(),
                'processed_suppl': self.processed_suppl
            }
            with open(self.suppl_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
    
    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        pp = self.pp[idx]
        suppl = self.processed_suppl[pp]
        batch['image'] = torch.cat([batch['image'], suppl], dim=0)
        
        return batch

def train_collate_fn(batch):
    image = [b['image'] for b in batch]
    image = torch.stack(image)
    
    boxes = [b['boxes'] for b in batch]
    boxes = pad_sequence(boxes, batch_first=True, padding_value=0)
    
    clses = [b['clses'] for b in batch]
    clses = pad_sequence(clses, batch_first=True, padding_value=-1)
    
    mask = torch.full(clses.shape, True)
    mask = mask & (clses >= 0)
    
    return {
        'image': image,
        'boxes': boxes,
        'clses': clses,
        'mask': mask
    }
    
def test():
    import torch
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    from torchvision.transforms.functional import InterpolationMode
    from torch.utils.data import DataLoader
    from argparse import Namespace
    import datasets as ds
    
    for suppl in ['saliency', 'density']:
        # args = Namespace(
        #     root = "/home/xuxiaoyuan/Scan-and-Print",
        #     dataset_root = "/home/xuxiaoyuan/calg_dataset",
        #     dataset = "pku",
        #     suppl_type = suppl,
        #     tokenizer_name = "sepoint",
        #     original_size = (513, 750),
        #     num_element_classes = 3,
        #     class_feature = ds.ClassLabel(names=['text', 'logo', 'underlay'])
        # )
        args = Namespace(
            root = "/home/xuxiaoyuan/Scan-and-Print",
            dataset_root = "/home/xuxiaoyuan/calg_dataset",
            dataset = "cgl",
            suppl_type = suppl,
            tokenizer_name = "sepoint",
            original_size = (514, 751),
            num_element_classes = 4,
            class_feature = ds.ClassLabel(names=["logo", "text", "underlay", "embellishment"])
        )
        
        import sys
        sys.path.append(args.root)
        from model.helper import rich_log
        
        transform = Compose([
            Resize(size=(224, 224), interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        dataset = ImageSupplLayoutDataset(args, transform, "train")
        print('Train dataset:', len(dataset))
        loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=train_collate_fn)
        batch = next(iter(loader))
        images, boxes, clses, mask = batch['image'], batch['boxes'], batch['clses'], batch['mask']
        print('images.shape:', images.shape)
        print('boxes.shape:', boxes.shape)
        print('clses.shape:', clses.shape)
        print('mask.shape:', mask.shape)
        # print('boxes:', boxes)
        # print('clses:', clses)
        # print('mask:', mask)
        dataset = ImageSupplLayoutDataset(args, transform, "valid")
        print('Valid dataset:', len(dataset))
        loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=train_collate_fn)
        batch = next(iter(loader))
        images, boxes, clses, mask = batch['image'], batch['boxes'], batch['clses'], batch['mask']
        print('images.shape:', images.shape)
        print('boxes.shape:', boxes.shape)
        print('clses.shape:', clses.shape)
        print('mask.shape:', mask.shape)
        # print('boxes:', boxes)
        # print('clses:', clses)
        # print('mask:', mask)
        dataset = ImageSupplLayoutDataset(args, transform, "test")
        print('Test dataset:', len(dataset))
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        batch = next(iter(loader))
        images = batch['image']
        print('images.shape:', images.shape)
        
if __name__ == "__main__":
    # DEBUG = True
    if DEBUG:
        print("**** Debug Mode ****")
    test()
