import torch
import argparse
import pickle as pkl
import sys
import os
from PIL import Image, ImageDraw
import datasets as ds
from tqdm import tqdm

visualize_layout = None # late init

def execute(args):
    global visualize_layout
    eval_dir = os.path.join(args.job_root, args.dataset_name, 'eval' if args.condition == "uncond" else f"eval_{args.condition}")
    assert os.path.exists(eval_dir), f"{eval_dir} does not exist"
    
    sys.path.append(args.root)
    from model.helper import rich_log
    from model.utils import visualize_layout
    import logging
    logger = logging.getLogger(__name__)
    
    
    base_dir = {
        'valid': {'general': os.path.join(args.dataset_root, args.dataset_name, 'image', 'train')},
        'test': {'general': os.path.join(args.dataset_root, args.dataset_name, 'image', 'test')}
    }
    
    visualize(eval_dir, base_dir, args.class_feature, logger)
    
def visualize(eval_dir, base_dir, class_feature, logger):
    file = os.listdir(eval_dir)
    file = list(filter(lambda x: x.endswith('.pkl'), file))
    
    splits = ["test", "valid"]
    for split in splits:
        save_dir = os.path.join(eval_dir, 'visualize', split)
        os.makedirs(save_dir, exist_ok=True)
        
        pkl_files = list(filter(lambda x: split in x, file))
        logger.info(f"Found {len(pkl_files)} samples for {split} split: {pkl_files}")
        
        for i_pkl_file, pkl_file in enumerate(pkl_files):
            save_path = os.path.join(save_dir, "{}_" + str(i_pkl_file) + '.png')
            
            layouts = read_pkl(os.path.join(eval_dir, pkl_file))
            for layout in tqdm(layouts):
                pp = layout['id']
                
                if os.path.exists(save_path.format(pp.split('.')[0])):
                    continue
                
                layout['clses'] = class_feature.int2str(layout['clses'])
                if layout['clses']:
                    # avoid empty tensor operation error
                    boxes = torch.tensor(layout['boxes'])
                    boxes[:, ::2] *= 240
                    boxes[:, 1::2] *= 350
                    layout['boxes'] = boxes.int().tolist()
                rd = visualize_layout(os.path.join(base_dir[split]['general'], 'input', pp), layout)
                rd.save(save_path.format(pp.split('.')[0]))
    
def read_pkl(path):
    with open(path, 'rb') as f:
        _input = pkl.load(f)
    
    return _input
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/xuxiaoyuan/Scan-and-Print')
    parser.add_argument('--dataset_root', type=str, default='/home/xuxiaoyuan/calg_dataset/')
    parser.add_argument('--job_root', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default='pku', choices=['pku', 'cgl'])
    parser.add_argument('--condition', type=str, default='uncond', choices=['uncond', 'c'])
    args = parser.parse_args()
    
    if args.dataset_name == 'pku':
        args.class_feature = ds.ClassLabel(names=["text", "logo", "underlay"])
    elif args.dataset_name == 'cgl':
        args.class_feature = ds.ClassLabel(names=["logo", "text", "underlay", "embellishment"])
    else:
        raise ValueError(f"Invalid dataset: {args.dataset_name}")
                
    execute(args)
    
if __name__ == "__main__":
    main()