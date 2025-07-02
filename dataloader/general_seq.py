import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
from PIL import Image
from pandas import read_csv
from torch.nn.utils.rnn import pad_sequence
import logging
import pickle
from tqdm import tqdm

logger = logging.getLogger(__name__)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

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
            else:
                os.makedirs(os.path.join(self.cache_dir, args.tokenizer_name), exist_ok=True)
                
                df = read_csv(os.path.join(args.dataset_root, args.dataset, "annotation", "train.csv"))
                df = df[df["split"] == split].reset_index(drop=True)
            
                # preprocess: box<string> -> box<list>, check if box is valid, and box<xyxy> -> box<xywh>
                df["box_elem"] = [eval(box) for box in df["box_elem"]]
                for i, box in enumerate(df["box_elem"]):
                    if box[2] < box[0]:
                        df.box_elem[i][2], df.box_elem[i][0] = box[0], box[2]
                    if box[3] < box[1]:
                        df.box_elem[i][3], df.box_elem[i][1] = box[1], box[3]
                
                groups = df.groupby("poster_path")
                self.dict_dataset = {}
                self.pp = df["poster_path"].unique()
                
                for pp, subdf in tqdm(groups, desc=f"Processing {split} layout"):
                    boxes = torch.tensor(subdf["box_elem"].tolist(), dtype=torch.float32)
                    boxes[:, ::2] /= args.original_size[0]
                    boxes[:, 1::2] /= args.original_size[1]
                    boxes = box_xyxy_to_cxcywh(boxes)
                    clses = torch.tensor(subdf["cls_elem"].tolist(), dtype=torch.long) - 1
                    # clses = F.one_hot(clses, num_classes=args.num_element_classes + 1)
                    self.dict_dataset[pp] = {
                        "boxes": boxes,
                        "clses": clses
                    }
                
                # Save layout cache
                cache_data = {
                    'dict_dataset': self.dict_dataset,
                    'pp': self.pp
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
    boxes = pad_sequence(boxes, batch_first=True, padding_value=0.0)
    
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
        #     tokenizer_name = "general",
        #     original_size = (513, 750),
        #     num_element_classes = 3,
        #     class_feature = ds.ClassLabel(names=['text', 'logo', 'underlay'])
        # )
        
        args = Namespace(
            root = "/home/xuxiaoyuan/Scan-and-Print",
            dataset_root = "/home/xuxiaoyuan/calg_dataset",
            dataset = "cgl",
            suppl_type = suppl,
            tokenizer_name = "general",
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
    test()

