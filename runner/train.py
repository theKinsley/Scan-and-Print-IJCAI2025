import torch
from torch.utils.data import DataLoader
import sys
import logging
from tqdm import tqdm
from config import TrainArgumentCfg
import os
import time
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

def execute(args):
    min_loss = float('inf')
    last_save_path = None
    augmented_sample = 0
    
    sys.path.append(args.root)
    from model.utils import compute_params, visualize_layout
    
    model, optimizer, scheduler = args.load_model_and_optimizer()
    logger.info(f"Model parameters: {compute_params(model):.3f}M")
    
    LayoutDataset, train_collate_fn, valid_collate_fn, _ = args.load_dataloader()
    
    train_dataset = LayoutDataset(args, transform=model.vit_transforms, split="train")
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=train_collate_fn)
    valid_dataset = LayoutDataset(args, transform=model.vit_transforms, split="valid")
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              collate_fn=valid_collate_fn)
    
    logger.info(f"Train/Valid dataset: {len(train_dataset)}/{len(valid_dataset)}")
    
    for epoch in range(args.num_epochs):
        start_time = time.time()
        train_log_losses, batch_sample = train(model, optimizer, train_loader, args.device)
        batch_sample = batch_sample - len(train_loader.dataset)
        scheduler.step()
        logger.info("Epoch %d/%d: elapsed = %.1fs, learning_rate = %s"
                    % (
                        epoch,
                        args.num_epochs,
                        time.time() - start_time,
                        scheduler.get_last_lr(),
                    )
        )
        logger.info(f"Train loss: {train_log_losses}, Augmented samples: {batch_sample}")
        augmented_sample += batch_sample
        valid_log_losses, output = valid(model, valid_loader, args.device, args.vis_preview)
        logger.info(f"Valid loss: {valid_log_losses}")
        
        if args.vis_preview:
            layouts = args.tokenizer.process_layout(output, args.preview_size)
            vis = []
            for i, layout in enumerate(layouts):
                preview = visualize_layout(os.path.join(valid_dataset.preview_dir, valid_dataset.pp[i]), layout, args.preview_size)
                vis.append(np.array(preview))
            vis = np.array(vis)
            vis = vis.reshape(-1, 8, *vis.shape[1:]).transpose(0, 2, 1, 3, 4)
            vis = vis.reshape(vis.shape[0] * vis.shape[1], vis.shape[2] * vis.shape[3], vis.shape[4])
            vis = Image.fromarray(vis)
            vis.save(os.path.join(args.save_path, "vis", f"epoch_{epoch}.png"))
        
        valid_loss = sum(valid_log_losses.values())
        if valid_loss < min_loss:
            min_loss = valid_loss
            logger.info(f"Saving model to {args.save_path}")
            if last_save_path is not None:
                os.remove(last_save_path)
            last_save_path = os.path.join(args.save_path, "checkpoint", f"epoch_{epoch}_loss_{valid_loss:.4f}.pth")
            torch.save(model.state_dict(), last_save_path)
            
    logger.info(f"Total augmented sample: {augmented_sample}")

def train(model, optimizer, dataloader, device):
    model.train()
    
    log_losses = {}
    total_sample = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        layouts = batch
        images = batch['image']
        total_sample += images.shape[0]
        patch_indices = None if 'patch_indices' not in batch else batch['patch_indices']
        
        x, y_true = model.preprocess({
            "image": images,
            "layout": layouts,
            "patch_indices": patch_indices
        }, device)
        
        model.train()
        model.zero_grad()
        output, losses = model.train_loss(x, y_true)
        loss = sum(losses.values())
        loss.backward()
        optimizer.step()
        
        for k, v in losses.items():
            if k not in log_losses:
                log_losses[k] = []
            log_losses[k].append(v.item())
    
    for k, v in log_losses.items():
        log_losses[k] = sum(v) / len(v)
            
    return log_losses, total_sample
        
@torch.no_grad()
def valid(model, dataloader, device, vis_preview):
    model.eval()
    
    log_losses = {}
    if vis_preview:
        init_output = None
    
    for batch in tqdm(dataloader, desc="Validating"):
        layouts = batch
        images = batch['image']
        patch_indices = None if 'patch_indices' not in batch else batch['patch_indices']
        
        x, y_true = model.preprocess({
            "image": images,
            "layout": layouts,
            "patch_indices": patch_indices
        }, device)
        
        output, losses = model.train_loss(x, y_true)
        for k, v in losses.items():
            if k not in log_losses:
                log_losses[k] = []
            log_losses[k].append(v.item())
        
        if vis_preview and init_output is None:
            init_output = output
        
    for k, v in log_losses.items():
        log_losses[k] = sum(v) / len(v)
    
    return log_losses, model.postprocess(init_output.detach().clone().cpu())

def main():
    args = TrainArgumentCfg()
    execute(args)
    
if __name__ == "__main__":
    main()