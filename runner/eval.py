import torch
import argparse
import pickle as pkl
import sys
import os
from metrics import compute_validity, compute_overlay, compute_alignment, compute_underlay_effectiveness, \
    compute_density_aware_metrics, compute_saliency_aware_metrics
import numpy as np

import warnings
def junk(x, s, d, g, e, r):
    return None
warnings.showwarning = junk

def execute(args):
    splits = ["test", "valid"] if args.condition == "uncond" else ["valid"]
    eval_dir = os.path.join(args.job_root, args.dataset_name, 'eval' if args.condition == "uncond" else f"eval_{args.condition}")
    assert os.path.exists(eval_dir), f"{eval_dir} does not exist"
    write_txt_path = os.path.join(eval_dir, 'scores.txt') if not args.DENSITY else os.path.join(eval_dir, 'scores_density.txt')
    
    sys.path.append(args.root)
    from model.helper import rich_log
    import logging
    logger = logging.getLogger(__name__)
    
    if not args.DEBUG and os.path.exists(write_txt_path):
        logger.info(f"Result already exists at {write_txt_path}")
        return
    
    logger.info(f"Evaluating job {eval_dir}, result will be written to {write_txt_path}")
    
    dataset_label = {
        'pku': {0: 'text', 1: 'logo', 2: 'underlay'},
        'cgl': {0: 'logo', 1: 'text', 2: 'underlay', 3: 'embellishment'}
    }
    label_id = {v: k for k, v in dataset_label[args.dataset_name].items()}
    base_dir = {
        'valid': {
            'general': os.path.join(args.dataset_root, args.dataset_name, 'image', 'train'),
            'density': os.path.join(args.dataset_root, 'all', 'density', 'train')
        },
        'test': {
            'general': os.path.join(args.dataset_root, args.dataset_name, 'image', 'test'),
            'density': os.path.join(args.dataset_root, 'all', 'density', 'test')
        }
    }
    
    result_strs = eval(splits, eval_dir, base_dir, label_id, args.dataset_name, logger, args.DEBUG, args.DENSITY) 
    if not args.DEBUG:
        with open(write_txt_path, 'w') as f:
            if args.condition == "uncond":
                f.write(f"{eval_dir.split('/')[-3]}/Test/Valid\n")
            else:
                f.write(f"{eval_dir.split('/')[-3]}/Valid\n")
            for split, result_str in result_strs.items():
                f.write(result_str)
    
def eval(splits, eval_dir, base_dir, label_id, dataset_name, logger, DEBUG=False, DENSITY=False):
    text_id = label_id['text']
    underlay_id = label_id['underlay']
    
    file = os.listdir(eval_dir)
    file = list(filter(lambda x: x.endswith('.pkl'), file))
    
    result_strs = {split: "" for split in splits}
    for split in splits:
        metrics = []
        pkl_files = list(filter(lambda x: split in x, file))
        logger.info(f"Found {len(pkl_files)} samples for {split} split: {pkl_files}")
        str_len = []
        for pkl_file in pkl_files:
            p = read_pkl(os.path.join(eval_dir, pkl_file))
            if 'center_x' not in p[0] or 'center_y' not in p[0] or 'width' not in p[0] or 'height' not in p[0]:
                p = boxesToCenters(p)
            valid_pts, metric_val = compute_validity(p)
            str_len.append(len(valid_pts))
            batch = ptsToBatch(valid_pts)

            metric = {"validity": metric_val}
            if DENSITY:
                metric.update(compute_density_aware_metrics(batch, base_dir[split], dataset=dataset_name))
            else:
                metric.update(compute_overlay(batch, underlay_id=underlay_id))
                metric.update(compute_alignment(batch))
                metric.update(compute_underlay_effectiveness(batch, underlay_id=underlay_id))
                if not DEBUG:
                    metric.update(compute_saliency_aware_metrics(batch, base_dir[split], text_id=text_id, underlay_id=underlay_id))
                # metric.update(compute_density_aware_metrics(batch, base_dir[split], dataset=dataset_name))
            metrics.append(metric)
        logger.info(f"Len: {str_len}")
        metric = {}
        for k in metrics[0].keys():
            metric[k] = sum([np.mean(m[k]) for m in metrics]) / len(metrics)
            
        result_str = ""
        for k, v in metric.items():
            result_str += f"{k}: {np.mean(v)}\n"
            # print(f"{k}: {np.mean(v)}")

        logger.info(result_str)
        if not DEBUG:
            result_str = parseResultStr(result_str, DENSITY)
        result_strs[split] = result_str
        
    return result_strs

def read_pkl(path):
    with open(path, 'rb') as f:
        _input = pkl.load(f)
    
    return _input

def boxesToCenters(pts):
    for pt in pts:
        if pt['clses']:
            boxes = torch.tensor(pt['boxes'])
            pt['center_x'] = (boxes[:, 0] + boxes[:, 2]) / 2
            pt['center_y'] = (boxes[:, 1] + boxes[:, 3]) / 2
            pt['width'] = boxes[:, 2] - boxes[:, 0]
            pt['height'] = boxes[:, 3] - boxes[:, 1]
        else:
            pt['clses'] = torch.tensor([0])
            pt['center_x'] = torch.tensor([0.0])
            pt['center_y'] = torch.tensor([0.0])
            pt['width'] = torch.tensor([0.0])
            pt['height'] = torch.tensor([0.0])
        assert len(pt['clses']) == len(pt['center_x']) == len(pt['center_y']) == len(pt['width']) == len(pt['height']), \
            f"Length mismatch: {len(pt['clses'])}, {len(pt['center_x'])}, {len(pt['center_y'])}, {len(pt['width'])}, {len(pt['height'])}"
        del pt['boxes']
        
    return pts

def ptsToBatch(pts):
    '''
    output: Dict[str, torch.Tensor]
        - center_x: [max_len, 10]
        - center_y: [max_len, 10]
        - width: [max_len, 10]
        - height: [max_len, 10]
        - label: [max_len, 10]
        - id: List(max_len)
        - mask: [max_len, 10]
    '''
    
    center_xs = []
    center_ys = []
    widths = []
    heights = []
    labels = []
    ids = []
    masks = []
    
    for pt in pts:
        valid_n = len(pt["clses"])
        max_len = 10
        center_x = torch.zeros(max_len)
        center_y = torch.zeros(max_len)
        width = torch.zeros(max_len)
        height = torch.zeros(max_len)
        label = torch.zeros(max_len)
        mask = torch.concat([torch.ones(min(max_len, valid_n)), torch.zeros(max(0, max_len - valid_n))]).bool()
        
        center_x[:valid_n] = torch.tensor(pt["center_x"][:max_len])
        center_y[:valid_n] = torch.tensor(pt["center_y"][:max_len])
        width[:valid_n] = torch.tensor(pt["width"][:max_len])
        height[:valid_n] = torch.tensor(pt["height"][:max_len])
        label[:valid_n] = torch.tensor(pt["clses"][:max_len])
        pid = pt["id"]
        
        center_xs.append(center_x)
        center_ys.append(center_y)
        widths.append(width)
        heights.append(height)
        labels.append(label)
        ids.append(pid)
        masks.append(mask)

        
    center_xs = torch.stack(center_xs, dim=0)
    center_ys = torch.stack(center_ys, dim=0)
    widths = torch.stack(widths, dim=0)
    heights = torch.stack(heights, dim=0)
    labels = torch.stack(labels, dim=0)
    masks = torch.stack(masks, dim=0)

    batch = {
        "center_x": center_xs, 
        "center_y": center_ys, 
        "width": widths, 
        "height": heights, 
        "label": labels, 
        "id": ids, 
        "mask": masks
    }

    return batch

def parseResultStr(result_str, DENSITY):
    result_str = result_str.split('\n')
    result_dict = {}
    wanted = ["validity",
              "overlay",
              "alignment-LayoutGAN++",
              "underlay_effectiveness_loose",
              "underlay_effectiveness_strict",
            #   "intention_coverage",
            #   "intention_conflict",
              "utilization",
              "occlusion",
              "unreadability"] if not DENSITY else ["intention_coverage", "intention_conflict"]
    
    for s in result_str:
        try:
            k, v = s.split(': ')
        except:
            continue
        if k in wanted:
            if k in result_dict:
                result_dict[k].append(v)
            else:
                result_dict[k] = [v]
    
    return_str = ""
    for i in range(len(result_dict[wanted[0]])):
        return_str += " & ".join([result_dict[w][i] for w in wanted]) + r" \\" + "\n"
    
    return return_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/xuxiaoyuan/Scan-and-Print')
    parser.add_argument('--dataset_root', type=str, default='/home/xuxiaoyuan/calg_dataset/')
    parser.add_argument('--job_root', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default='pku', choices=['pku', 'cgl'])
    parser.add_argument('--condition', type=str, default='uncond', choices=['uncond', 'c', 'partial'])
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--DENSITY', action='store_true')
    args = parser.parse_args()
    
    if args.DEBUG:
        print("**Be careful it's in DEBUG mode! No existing check nor writing to file!**")
    if args.DENSITY:
        print("**Be careful it's only for DENSITY!**")
    
    execute(args)
    
if __name__ == "__main__":
    main()