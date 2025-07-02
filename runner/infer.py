import torch
from torch.utils.data import DataLoader
import sys
import logging
from tqdm import tqdm
from config import EvalArgumentCfg
import os
import pickle as pkl
import time

logger = logging.getLogger(__name__)

def execute(args, test_speed=False):
    if args.condition != "uncond":
        assert args.condition in ["c", "partial"], "not supported condition: {}".format(args.condition)
        assert args.tokenizer_name == "sepoint", "not supported tokenizer for conditional inference: {}".format(args.tokenizer_name)
        
    model = args.load_model()
    
    LayoutDataset, _, valid_collate_fn, test_collate_fn = args.load_dataloader()
    for i in range(args.num_samples):
        args.refresh_seed(i)
        
        valid_dataset = LayoutDataset(args, transform=model.vit_transforms, split="valid")
        valid_loader = DataLoader(valid_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                collate_fn=valid_collate_fn)
        test_dataset = LayoutDataset(args, transform=model.vit_transforms, split="test")
        if test_collate_fn:
            test_loader = DataLoader(test_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                collate_fn=test_collate_fn)
        else:
            test_loader = DataLoader(test_dataset,
                                batch_size=args.batch_size,
                                shuffle=False)
                
        
        logger.info(f"Dataset size, Valid/Test: {len(valid_dataset)}/{len(test_dataset)}")
        
        if not test_speed:
            outputs = infer(model, valid_loader, args.sampling_cfg, args.device, condition=args.condition)
            save_layout_pkl(outputs[:len(valid_dataset)], valid_dataset, args.save_pkl_path.format("valid"))
        if args.condition == "uncond":
            outputs = infer(model, test_loader, args.sampling_cfg, args.device)
            if not test_speed:
                save_layout_pkl(outputs[:len(test_dataset)], test_dataset, args.save_pkl_path.format("test"))
    
@torch.no_grad()
def infer(model, dataloader, sampling_cfg, device, condition='uncond'):
    model.eval()
    outputs = []
    i = 0
    total_time = 0
    
    # for batch in tqdm(dataloader):
    for batch in dataloader:
        if condition == 'uncond':
            x = {
                k: v.to(device) for k, v in batch.items()
            }
            start_time = time.time()
            output = model.sample(x, sampling_cfg)
            end_time = time.time()
            total_time += end_time - start_time
        elif condition in ["c", "partial"]:
            x, y_true = model.preprocess({
                "image": batch['image'],
                "layout": batch,
                "patch_indices": batch["patch_indices"]
            }, device)
            output = model.sample_with_condition(x, y_true, sampling_cfg, condition=condition)
        else:
            raise ValueError(f"Not supported condition: {condition}")
        
        output = model.tokenizer.decode(output.detach().clone().cpu())
        output = model.tokenizer.process_layout(output)
        outputs.extend(output)
    
    if condition == 'uncond':
        print(f"Time taken: {total_time} seconds, for {len(dataloader.dataset)} samples")
    return outputs
        
def save_layout_pkl(outputs, dataset, save_path):
    for i in range(len(outputs)):
        outputs[i]['id'] = dataset.pp[i]
    with open(save_path, "wb") as f:
        pkl.dump(outputs, f)
    logger.info(f"Saved to {save_path}")
    
def speed_main(save_path):
    from argparse import Namespace
    import sys
    args = Namespace(
        save_path=save_path
    )
    sys.argv = ['infer.py'] + [f'--{k}={v}' for k, v in args.__dict__.items()]
    args = EvalArgumentCfg()
    args.batch_size = 1
    execute(args, test_speed=True)
    

def main():
    args = EvalArgumentCfg()
    execute(args)
    
if __name__ == "__main__":
    main()