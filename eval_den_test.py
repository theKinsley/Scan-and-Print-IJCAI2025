from pandas import read_csv
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import heapq
import pickle as pkl
from tqdm import tqdm

image_path = {
    'con_pku': '/home/xuxiaoyuan/calg_dataset/pku/image/test/density',
    'con_cgl': '/home/xuxiaoyuan/calg_dataset/cgl/image/test/density',
}

csv_path = {
    'pku': '/home/xuxiaoyuan/calg_dataset/pku/annotation/test.csv',
    'cgl': '/home/xuxiaoyuan/calg_dataset/cgl/annotation/test.csv'
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

def eval(dataset, conloss, fs, P, k):
    ps = (fs[0] / P, fs[1] / P)
    df = read_csv(csv_path[dataset])
    dendir = image_path[('non_' if not conloss else '') + f'con_{dataset}']

    results = []
    for pp in tqdm(df.poster_path.unique()):
        denmap = Image.open(os.path.join(dendir, pp))
        denmap = denmap.resize(fs)
        top_k_indice = get_top_k_with_heapq(denmap, P, ps, k)
        result = {
            'id': pp,
            'top_k_indice': top_k_indice,
            'sorted_top_k_indice': sorted(top_k_indice),
        }
        results.append(result)
        
    return results

evals_valid = {}

fs = (513, 750)
P = 14
k = 96
dataset = ['pku', 'cgl']
conloss = [True]
split = 'test'

for ds in dataset:
    for cl in conloss:
        save_path = f'cache/{ds}_test_patch_{P}_topk_{k}.pkl'
        if os.path.exists(save_path):
            print(f'{save_path} exists')
            continue
        else:
            print('Result will be saved to', save_path)
            results = eval(ds, cl, fs, P, k)
            with open(save_path, 'wb') as f:
                pkl.dump(results, f)