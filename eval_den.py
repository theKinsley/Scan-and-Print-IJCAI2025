from pandas import read_csv
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import heapq
import pickle as pkl
from tqdm import tqdm

image_path = {
    'con_pku': '/home/xuxiaoyuan/calg_dataset/pku/image/train/density',
    'con_cgl': '/home/xuxiaoyuan/calg_dataset/cgl/image/train/density',
}

csv_path = {
    'pku': '/home/xuxiaoyuan/calg_dataset/pku/annotation/train.csv',
    'cgl': '/home/xuxiaoyuan/calg_dataset/cgl/annotation/train.csv'
}

def draw_patch_edge(image, patch_stride, mask=False):
    w, h = image.size
    dw, dh = patch_stride
    draw = ImageDraw.Draw(image)
    if mask:
        masked_image = np.zeros((h, w, 3), dtype=float)
        thres = np.mean(np.array(image)) * 1.8
        count = 0
    for x in range(0, w, dw):
        for y in range(0, h, dh):
            if mask:
                mean = np.mean(np.array(image)[y:y+dh, x:x+dw])
                masked_image[y:y+dh, x:x+dw] = mean
                if mean < thres:
                    draw.rectangle((x, y, x+dw, y+dh), fill='black', outline='black', width=3)
                else:
                    count += 1
                    draw.rectangle((x, y, x+dw, y+dh), outline='black', width=3)
            else:
                draw.rectangle((x, y, x+dw, y+dh), outline='red', width=3)
    
    if mask:
        print('Light:', count)
        return image, masked_image
    else:
        return image

def draw_top_k_patch(image, patch_stride, top_k_indice):
    w, h = image.size
    dw, dh = patch_stride
    draw = ImageDraw.Draw(image)
    for i, x in enumerate(range(0, w, dw)):
        for j, y in enumerate(range(0, h, dh)):
            if (i, j) in top_k_indice:
                draw.rectangle((x, y, x+dw, y+dh), outline='black', width=3)
            else:
                draw.rectangle((x, y, x+dw, y+dh), fill='black', outline='black', width=3)
    return image

def get_top_k_with_thres(image, patch_stride, k=20, thres_p=1.8):
    w, h = image.size
    dw, dh = patch_stride
    thres = np.mean(np.array(image)) * thres_p
    results = []
    for i, x in enumerate(range(0, w, dw)):
        for j, y in enumerate(range(0, h, dh)):
            mean = np.mean(np.array(image)[y:y+dh, x:x+dw])
            if mean > thres:
                results.append((i, j, mean))
    
    if len(results) > k:
        results = sorted(results, key=lambda x: x[2], reverse=True)
    
    return sorted([r[:2] for r in results[:k]])

# def get_top_k_with_heapq(image, patch_stride, k=20):
#     w, h = image.size
#     dw, dh = patch_stride
#     min_heap = []
#     for i, x in enumerate(range(0, w, dw)):
#         for j, y in enumerate(range(0, h, dh)):
#             mean = np.mean(np.array(image)[y:y+dh, x:x+dw])
#             if len(min_heap) < k:
#                 heapq.heappush(min_heap, (mean, (i, j)))
#             else:
#                 if mean > min_heap[0][0]:
#                     heapq.heapreplace(min_heap, (mean, (i, j)))
    
#     return [item[1] for item in min_heap]

def get_top_k_with_heapq(image, patch_size, patch_stride, k=20):
    # print('patch_size:', patch_size, 'patch_stride:', patch_stride)
    p = patch_size
    dw, dh = patch_stride
    min_heap = []
    
    for i in range(p):
        x = int(i * dw)
        x_end = int((i + 1) * dw)
        for j in range(p):
            y = int(j * dh)
            y_end = int((j + 1) * dh)
            mean = np.mean(np.array(image)[y:y_end, x:x_end])
            if len(min_heap) < k:
                heapq.heappush(min_heap, (mean, (i, j)))
            else:
                if mean > min_heap[0][0]:
                    heapq.heapreplace(min_heap, (mean, (i, j)))
    
    return [item[1] for item in min_heap]

def eval_acc(dataset, conloss, fs, P, k, split='train'):
    ps = (fs[0] / P, fs[1] / P)
    df = read_csv(csv_path[dataset])
    splits = df.drop_duplicates(subset=['poster_path']).groupby(df.split)
    splits = {
        'train': splits.get_group('train')['poster_path'].tolist(),
        'valid': splits.get_group('valid')['poster_path'].tolist(),
    }
    groups = df.groupby(df.poster_path)
    dendir = image_path[('non_' if not conloss else '') + f'con_{dataset}']

    acc = []
    results = []
    for pp in tqdm(splits[split]):
        subdf = groups.get_group(pp).reset_index()
        # origin = Image.open(os.path.join(f'/home/xuxiaoyuan/calg_dataset/{dataset}/image/train/original', pp))
        # origin = origin.resize(fs)
        denmap = Image.open(os.path.join(dendir, pp))
        denmap = denmap.resize(fs)
        # p_origin = draw_patch_edge(origin.copy(), ps)
        # p_denmap, m_p_denmap = draw_patch_edge(denmap.copy(), ps, True)
        top_k_indice = get_top_k_with_heapq(denmap, P, ps, k)
        result = {
            'id': pp,
            'top_k_indice': top_k_indice,
            'sorted_top_k_indice': sorted(top_k_indice),
        }
        results.append(result)
        
        # m_k_denmap = draw_top_k_patch(denmap.copy(), ps, top_k_indice)
        # m_k_origin = draw_top_k_patch(origin.copy(), ps, top_k_indice)
        
        # plt.subplot(1, 4, 1)
        # plt.imshow(origin)
        # plt.axis('off')
        # plt.subplot(1, 4, 2)
        # plt.imshow(denmap)
        # plt.axis('off')
        # plt.subplot(1, 4, 3)
        # # plt.imshow(origin)
        # # plt.imshow(Image.fromarray(np.array(m_p_denmap, dtype=np.uint8)))
        # plt.imshow(m_k_denmap)
        # plt.axis('off')
        # plt.subplot(1, 4, 4)
        # # plt.imshow(p_denmap)
        # plt.imshow(m_k_origin)
        # plt.axis('off')
        # plt.show()
        
        # print(subdf)
        
        boxes = [eval(box) for box in subdf.box_elem.tolist()]
        boxes = [[box[0] // ps[0], box[1] // ps[1], box[2] // ps[0], box[3] // ps[1]] for box in boxes]
        start_points = [box[:2] for box in boxes]
        end_points = [box[2:] for box in boxes]
        
        tp = 0
        fn = 0
        for p in start_points + end_points:
            if tuple(p) in top_k_indice:
                tp += 1
            else:
                fn += 1

        # print(sorted(start_points + end_points))
        # print(sorted(top_k_indice))
        acc.append(tp / (tp + fn))
        
    return acc, results

evals_valid = {}

fs = (513, 750)
P = 14
k = 96
dataset = ['pku', 'cgl']
conloss = [True]
splits = ['train', 'valid']

    

for ds in dataset:
    for cl in conloss:
        for split in splits:
            save_path = f'cache/{ds}_{split}_patch_{P}_topk_{k}.pkl'
            if os.path.exists(save_path):
                print(f'{save_path} exists')
                continue
            else:
                print('Result will be saved to', save_path)
                evals_valid[split + ('_non_' if not cl else '_') + f'con_{ds}'], \
                    results = eval_acc(ds, cl, fs, P, k, split)
                with open(save_path, 'wb') as f:
                    pkl.dump(results, f)
        
for k, v in evals_valid.items():
    print(f"{k}: {np.mean(v)}")
    