import pickle as pkl
import torch
import logging
import os

logger = logging.getLogger(__name__)

def read_pkl(path):
    with open(path, 'rb') as f:
        _input = pkl.load(f)
    
    return _input

def indices_flatten(indices, P):
    indices = torch.tensor(indices)
    # return indices[:, 0] * P + indices[:, 1]
    return indices[:, 1] * P + indices[:, 0]

def load_filter_setting(LayoutDataset, train_collate_fn):
    
    class FilterLayoutDataset(LayoutDataset):
        def __init__(self, 
                     args,
                     transform,
                     split):
            
            super().__init__(args, transform, split)
            
            P = args.P
            pick_topk = args.pick_topk
            topk = args.topk
            
            assert pick_topk <= topk, "pick_topk must be less or equal than topk"
            assert 'patch_indices_dir' in args.__dict__, "patch_indices_dir must be provided for initializing filter setting"
            
            patch_indices_path = f"{args.dataset}_{split}_patch_{P}_topk_{topk}.pkl"
            logger.info(f"Loading filter setting for {LayoutDataset.__name__}, using {patch_indices_path}, pick_topk: {pick_topk}")
            patch_indices = read_pkl(os.path.join(args.patch_indices_dir, patch_indices_path))
            
            if pick_topk == topk:
                self.dict_patch_indices = {
                    patch['id']: indices_flatten(patch['sorted_top_k_indice'], P) 
                    for patch in patch_indices
                    }
            else:
                self.dict_patch_indices = {
                    patch['id']: indices_flatten(sorted(patch['top_k_indice'][::-1][:pick_topk]), P)
                    for patch in patch_indices
                    }
            
        def __getitem__(self, idx):
            batch = super().__getitem__(idx)
            pp = self.pp[idx]
            batch['patch_indices'] = self.dict_patch_indices[pp]
            
            return batch
    
    def filter_train_collate_fn(batch):
        patch_indices = [b['patch_indices'] for b in batch]
        patch_indices = torch.stack(patch_indices)
        
        batch_filter = train_collate_fn(batch)
        batch_filter['patch_indices'] = patch_indices
        
        return batch_filter
    
    def filter_test_collate_fn(batch):
        image = [b['image'] for b in batch]
        image = torch.stack(image)
        
        patch_indices = [b['patch_indices'] for b in batch]
        patch_indices = torch.stack(patch_indices)
        
        return {
            'image': image,
            'patch_indices': patch_indices
        }
    
    return FilterLayoutDataset, filter_train_collate_fn, filter_test_collate_fn

def test():
    import sys
    sys.path.append(os.environ['ROOT'])
    from dataloader.init import initialize_dataloader
    from model.helper import rich_log
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    from torchvision.transforms.functional import InterpolationMode
    from torch.utils.data import DataLoader
    from argparse import Namespace
    import datasets as ds
    args = Namespace(
        dataset_root = os.environ['DATASET_ROOT'],
        dataset = "pku",
        suppl_type = None,
        tokenizer_name = "sepoint",
        original_size = (513, 750),
        num_element_classes = 3,
        class_feature = ds.ClassLabel(names=['text', 'logo', 'underlay']),
        patch_indices_dir = f"{os.environ['ROOT']}/cache/topk_patch_indices",
        P = 14,
        topk = 96,
        pick_topk = 96,
    )
    
    LayoutDataset, train_collate_fn = initialize_dataloader(args)
    LayoutDataset, train_collate_fn, test_collate_fn = load_filter_setting(LayoutDataset, train_collate_fn)
    
    transform = Compose([
        Resize(size=(224, 224), interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    dataset = LayoutDataset(args, transform, "train")
    print('Train dataset:', len(dataset))
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=train_collate_fn)
    batch = next(iter(loader))
    for k, v in batch.items():
        print(k, v.shape)
    
    dataset = LayoutDataset(args, transform, "valid")
    print('Valid dataset:', len(dataset))
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=train_collate_fn)
    batch = next(iter(loader))
    for k, v in batch.items():
        print(k, v.shape)
    
    args.inference = True
    
    dataset = LayoutDataset(args, transform, "valid")
    print('Valid dataset:', len(dataset))
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=test_collate_fn)
    batch = next(iter(loader))
    for k, v in batch.items():
        print(k, v.shape)
    
    dataset = LayoutDataset(args, transform, "test")
    print('Test dataset:', len(dataset))
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=test_collate_fn)
    batch = next(iter(loader))
    for k, v in batch.items():
        print(k, v.shape)
    
if __name__ == "__main__":
    test()