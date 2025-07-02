import torch
import random
import logging
import numpy as np
from torch.nn.functional import cosine_similarity
from einops import rearrange, repeat
from tqdm import tqdm
import time
import os

logger = logging.getLogger(__name__)

def load_augment_setting(args, train_collate_fn):
    P = args.P
    tokenizer = args.tokenizer
    num_augment = args.num_augment
    
    if args.similarity_type == 'pearson':
        similarity_function = calculate_pearson_similarity
    elif args.similarity_type == 'cosine':
        similarity_function = calculate_cosine_similarity
    elif args.similarity_type == None or args.similarity_type == 'None':
        similarity_function = None
    else:
        raise ValueError(f"Invalid similarity type: {args.similarity_type}")
    
    def select_mixing_samples(indices):
        if similarity_function is None:
            # avoid self-mixing
            sample_indices = torch.randint(0, len(indices), (num_augment * 2, 2))
            sample_indices = sample_indices[sample_indices[:, 0] != sample_indices[:, 1]]
            sample_indices = sample_indices[:num_augment]
        else:
            similarities = similarity_function(indices)
            sample_indices = torch.triu_indices(len(indices), len(indices), 1)
            sample_indices = sample_indices[:, torch.argsort(similarities[sample_indices[0], sample_indices[1]])[:num_augment]]
            sample_indices = rearrange(sample_indices, 'a b -> b a')
            
        return sample_indices
    
    def mix_samples(images, layouts, patch_indices):
        image, symmetric = mix_images(patch_indices, P, images)
        areas = calculate_areas(symmetric)
        layout = mix_layouts(layouts[0], layouts[1], areas, tokenizer, P)
        if layout == -1 or layout['boxes'].shape[0] == 0 or layout['clses'].shape[0] == 0:
            return None
        else:
            return {
                'image': image,
                'boxes': layout['boxes'],
                'clses': layout['clses']
            }
    
    def augment_train_collate_fn(batch):
        batch_augment = train_collate_fn(batch)
        
        start_time = time.time()
        unflattened_indices = indices_unflatten(batch_augment['patch_indices'], P)
        samples = select_mixing_samples(unflattened_indices) # (num_augment, 2)
        
        augment_images = []
        augment_boxes = []
        augment_clses = []
        augment_patch_indices = []
        
        for i, j in tqdm(samples, desc="Augmenting"):
            boxes = torch.zeros_like(batch_augment['boxes'][i])
            clses = torch.full_like(batch_augment['clses'][i], -1)
            
            mixed_samples = mix_samples(images=[batch_augment['image'][i], batch_augment['image'][j]],
                                        layouts=[{'boxes': batch_augment['boxes'][i][batch_augment['mask'][i]],
                                                  'clses': batch_augment['clses'][i][batch_augment['mask'][i]]},
                                                 {'boxes': batch_augment['boxes'][j][batch_augment['mask'][j]],
                                                  'clses': batch_augment['clses'][j][batch_augment['mask'][j]]}],
                                        patch_indices=[unflattened_indices[i], unflattened_indices[j]])
            
            if mixed_samples is None:
                continue
            else:
                image = mixed_samples['image']
                boxes[:len(mixed_samples['boxes'])] = mixed_samples['boxes']
                clses[:len(mixed_samples['clses'])] = mixed_samples['clses']
                
                augment_images.append(image)
                augment_boxes.append(boxes)
                augment_clses.append(clses)
                augment_patch_indices.append(batch_augment['patch_indices'][i])
        
        if len(augment_images) == 0:
            return batch_augment
        
        augment_images = torch.stack(augment_images)
        augment_boxes = torch.stack(augment_boxes)
        augment_clses = torch.stack(augment_clses)
        augment_patch_indices = torch.stack(augment_patch_indices)
        mask = torch.full(augment_clses.shape, True)
        mask = mask & (augment_clses >= 0)
        
        batch_augment['image'] = torch.cat([batch_augment['image'], augment_images], dim=0)
        batch_augment['boxes'] = torch.cat([batch_augment['boxes'], augment_boxes], dim=0)
        batch_augment['clses'] = torch.cat([batch_augment['clses'], augment_clses], dim=0)
        batch_augment['patch_indices'] = torch.cat([batch_augment['patch_indices'], augment_patch_indices], dim=0)
        batch_augment['mask'] = torch.cat([batch_augment['mask'], mask], dim=0)
        
        end_time = time.time()
        duration = end_time - start_time
        if os.path.exists(os.path.join(args.save_path, "augment_time.npy")):
            augment_time = np.load(os.path.join(args.save_path, "augment_time.npy"))
            augment_time = np.concatenate([augment_time, [duration]])
            np.save(os.path.join(args.save_path, "augment_time.npy"), augment_time)
        else:
            np.save(os.path.join(args.save_path, "augment_time.npy"), [duration])
        
        return batch_augment
    
    return augment_train_collate_fn
    
            
def calculate_pearson_similarity(samples):
    pearson_similarities = torch.full((len(samples), len(samples)), float('nan'))
    
    x_samples = samples[:, :, 0].numpy()
    y_samples = samples[:, :, 1].numpy()
    x_corr_matrix = torch.tensor(np.corrcoef(x_samples))
    y_corr_matrix = torch.tensor(np.corrcoef(y_samples))
    pearson_similarities = (x_corr_matrix + y_corr_matrix) / 2
    
    return pearson_similarities
    
def calculate_cosine_similarity(samples):
    samples_flat = samples.view(len(samples), -1).float()
    cosine_similarities = cosine_similarity(samples_flat.unsqueeze(1),
                                            samples_flat.unsqueeze(0),
                                            dim=2)
    return cosine_similarities

def indices_unflatten(indices, P):
    dim0 = indices % P
    dim1 = indices // P
    return torch.stack([dim0, dim1], dim=-1)

def mix_choices_and_symmetric(indices_i, indices_j, P):
    choices = torch.zeros(P, P)
    choices[indices_i[:, 1], indices_i[:, 0]] = 1
    # choices[indices_j[:, 1], indices_j[:, 0]] = 0
    
    mask_i = torch.zeros(P, P).bool()
    mask_i[indices_i[:, 0], indices_i[:, 1]] = True
    mask_j = torch.zeros(P, P).bool()
    mask_j[indices_j[:, 0], indices_j[:, 1]] = True
    symmetric = mask_i & mask_j
    
    return choices, symmetric

def mix_images(indices, P, images):
    choices, symmetric = mix_choices_and_symmetric(indices[0], indices[1], P)
    symmetric = symmetric.nonzero().tolist()
    
    W = images[0].shape[1] // P
    H = images[0].shape[2] // P
    
    images[0] = rearrange(images[0].clone(), 'c (h1 h2) (w1 w2) -> h1 w1 h2 w2 c', h1=P, h2=H, w1=P, w2=W)
    images[1] = rearrange(images[1].clone(), 'c (h1 h2) (w1 w2) -> h1 w1 h2 w2 c', h1=P, h2=H, w1=P, w2=W)
    images[0][choices == 1] = images[1][choices == 1]
    images[0] = rearrange(images[0], 'h1 w1 h2 w2 c -> c (h1 h2) (w1 w2)', h1=P, h2=H, w1=P, w2=W)
    
    return images[0], symmetric
    
def dfs(i, j, visited, symmetric):
    stack = [(i, j)]
    area = 0
    indices = []
    while stack:
        x, y = stack.pop()
        if (x, y) not in visited:
            visited.add((x, y))
            area += 1
            indices.append([x, y])
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if [nx, ny] in symmetric and (nx, ny) not in visited:
                    stack.append((nx, ny))
    return area, indices

def is_valid_area(indices):
    top_left = min(indices)
    bottom_right = max(indices)
    if bottom_right[0] > top_left[0] and bottom_right[1] > top_left[1]:
        return True, top_left, bottom_right
    else:
        return False, None, None

def calculate_areas(symmetric):
    visited = set()
    areas = []
    for i, j in symmetric:
        if (i, j) not in visited:
            area, indices = dfs(i, j, visited, symmetric)
            if area >= 3:
                valid, top_left, bottom_right = is_valid_area(indices)
                if valid:
                    areas.append({
                        'area': area,
                        'indices': indices,
                        'top_left': top_left,
                        'bottom_right': bottom_right
                    })
    return areas

def find_lcs(cls_i, cls_j):
    n, m = len(cls_i), len(cls_j)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if cls_i[i - 1] == cls_j[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    lcs = []
    i, j = n, m
    lcs_i, lcs_j = torch.zeros(n).bool(), torch.zeros(m).bool()
    while i > 0 and j > 0:
        if cls_i[i - 1] == cls_j[j - 1]:
            lcs.append(cls_i[i - 1])
            lcs_i[i - 1] = True
            lcs_j[j - 1] = True
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    return lcs[::-1], lcs_i.bool(), lcs_j.bool()

def empty_underlay(clses, tokenizer):
    pattern_check = torch.tensor(clses == tokenizer.underlay_start_id).nonzero().squeeze(-1)
    if len(pattern_check) > 0:
        next_idx = pattern_check + 1
        next_idx = next_idx[next_idx < len(clses)]
        if any((clses[next_idx] == tokenizer.underlay_start_id + 1).tolist()):
            return True
    return False

def mix_layouts(layout_i, layout_j, areas, tokenizer, P):
    clses, lcs_i, lcs_j = find_lcs(layout_i['clses'].tolist(), layout_j['clses'].tolist())
    if len(clses) == 0 or len(areas) == 0 or empty_underlay(clses, tokenizer):
        return -1
    points_i, points_j = layout_i['boxes'][lcs_i], layout_j['boxes'][lcs_j]
    
    valid_mask = torch.zeros(len(clses)).bool()
    start_mask = torch.zeros(len(clses)).bool()
    
    FIFO = []
    for i in range(len(clses)):
        if clses[i] in tokenizer.start_of_id.values():
            FIFO.append((i, clses[i]))
        elif clses[i] in tokenizer.end_of_id.values():
            for j in range(len(FIFO)):
                if FIFO[j][1] == tokenizer.start_of_id[clses[i]]:
                    break
            else:
                continue
            start_mask[FIFO[j][0]] = True
            valid_mask[FIFO[j][0]] = True
            valid_mask[i] = True
            FIFO.pop(j)
        else:
            raise ValueError(f"Invalid clses: {clses}")
    
    clses = torch.tensor(clses)
    clses = clses[valid_mask].tolist()
    points = torch.where(repeat(start_mask, 'a -> a c', c=2), points_i, points_j)
    points = points[valid_mask] % (1 / P)
    points = points.tolist()
    
    groups = group_clses_points(list(zip(clses, points)), tokenizer, depth=0)
    random.shuffle(areas)
    random.shuffle(groups)
    try:
        layout = arrange_sepoints(groups, areas, 1 / P)
    except:
        return -1
    layout['clses'] = torch.tensor(layout['clses'])
    layout['boxes'] = torch.tensor(layout['boxes'])
    
    return layout

def group_clses_points(elements, tokenizer, depth):
    groups = []
    FIFO = []
    i = 0
    
    while i < len(elements):
        if elements[i][0] == tokenizer.underlay_start_id:
            nested_count = 0
            end_idx = i + 1
            while end_idx < len(elements):
                if elements[end_idx][0] == tokenizer.underlay_start_id:
                    nested_count += 1
                elif elements[end_idx][0] == tokenizer.end_of_id[tokenizer.underlay_start_id]:
                    if nested_count == 0:
                        break
                    nested_count -= 1
                end_idx += 1
            
            if depth == 0:
                nested_group = [
                    [elements[i]], 
                    *group_clses_points(elements[i+1:end_idx], tokenizer, depth+1),
                    [elements[end_idx]]
                ]
                groups.append(nested_group)
                
            i = end_idx + 1
        else:
            if elements[i][0] in tokenizer.start_of_id.values():
                FIFO.append(elements[i])
            else:
                for j in range(len(FIFO)):
                    if FIFO[j][0] == tokenizer.start_of_id[elements[i][0]]:
                        groups.append([FIFO.pop(j), elements[i]])
                        break
            i += 1
            
    return groups

def arrange_sepoints(groups, areas, d):
    layout = {
        'clses': [],
        'boxes': []
    }
    
    for area, group in zip(areas, groups):
        top_left, bottom_right = area['top_left'], area['bottom_right']
        top_left = top_left[0] * d, top_left[1] * d
        bottom_right = bottom_right[0] * d, bottom_right[1] * d
        if len(group) == 2:
            layout['clses'].append(group[0][0])
            layout['boxes'].append([group[0][1][0] + top_left[0], group[0][1][1] + top_left[1]])
            layout['clses'].append(group[1][0])
            layout['boxes'].append([group[1][1][0] + bottom_right[0], group[1][1][1] + bottom_right[1]])
        elif len(group) > 2:
            start = group[0][0]
            end = group[-1][0]
            layout['clses'].append(start[0])
            layout['boxes'].append([start[1][0] + top_left[0], start[1][1] + top_left[1]])

            elements = group[1]
            layout['clses'].append(elements[0][0])
            layout['boxes'].append([elements[0][1][0] + top_left[0], elements[0][1][1] + top_left[1]])
            layout['clses'].append(elements[1][0])
            layout['boxes'].append([elements[1][1][0] + bottom_right[0], elements[1][1][1] + bottom_right[1]])

            # N = len(group) - 2
            # w = bottom_right[0] - top_left[0]
            # h = bottom_right[1] - top_left[1]
            # w = w / N
            # h = h / N
            
            # for i in range(1, N + 1):
            #     elements = group[i]
                
            #     top_left = top_left[0] + w, top_left[1] + h
            #     bottom_right = bottom_right[0] - w, bottom_right[1] - h
                
            #     layout['clses'].append(elements[0][0])
            #     layout['boxes'].append([elements[0][1][0] + top_left[0], elements[0][1][1] + top_left[1]])
            #     layout['clses'].append(elements[1][0])
            #     layout['boxes'].append([elements[1][1][0] + bottom_right[0], elements[1][1][1] + bottom_right[1]])
            
            layout['clses'].append(end[0])
            layout['boxes'].append([end[1][0] + bottom_right[0], end[1][1] + bottom_right[1]])
        else:
            raise ValueError(f"Invalid group: {group}")
        
    return layout

def test():
    from argparse import Namespace
    args = Namespace(
        root=os.environ['ROOT'],
        gpu='0',
        dataset='pku',
        model_name='filtering',
        tokenizer_name='sepoint',
        var_order=('clses', 'x', 'y'),
        suppl_type=None,
        use_layout_encoder=True,
        patch_indices_dir=f"{os.environ['ROOT']}/cache/topk_patch_indices",
        P=14,
        topk=96,
        pick_topk=96,
        similarity_type='cosine',
        num_augment=32,
        augment=True,
    )
    
    import torch
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode, ToPILImage
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import sys
    sys.path.append(args.root)
    from runner.config import TrainArgumentCfg
    from dataloader.init import initialize_dataloader
    from dataloader.filter_setting import load_filter_setting
    sys.argv = [f'--{k}={v}' for k, v in args.__dict__.items()]
    args = TrainArgumentCfg()
    
    transform = Compose([
        Resize(size=(224, 224), interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    detransform = Compose([
        Normalize(mean=(-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225), std=(1 / 0.229, 1 / 0.224, 1 / 0.225)),
        ToPILImage()
    ])
    
    LayoutDataset, train_collate_fn = initialize_dataloader(args)
    LayoutDataset, train_collate_fn, test_collate_fn = load_filter_setting(LayoutDataset, train_collate_fn)
    augment_train_collate_fn = load_augment_setting(args, train_collate_fn)
    
    dataset = LayoutDataset(args, transform, "train")
    print('Train dataset - no shuffle:', len(dataset))
    loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=augment_train_collate_fn)
    batch = next(iter(loader))
    for k, v in batch.items():
        print(k, v.shape)
    # print(batch['clses'][128:])
    # print(batch['boxes'][128:])
    
    print('Train dataset - shuffle:', len(dataset))
    loader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=augment_train_collate_fn)
    batch = next(iter(loader))
    for k, v in batch.items():
        print(k, v.shape)
    # print(batch['clses'][128:])
    # print(batch['boxes'][128:])
        
    dataset = LayoutDataset(args, transform, "valid")
    print('Valid dataset:', len(dataset))
    loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=train_collate_fn)
    batch = next(iter(loader))
    for k, v in batch.items():
        print(k, v.shape)

if __name__ == "__main__":
    test()