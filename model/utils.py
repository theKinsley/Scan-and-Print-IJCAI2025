import torch
import numpy as np
import random
import os
from PIL import Image, ImageDraw
import logging
from thop import profile

logger = logging.getLogger(__name__)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(seed % torch.cuda.device_count())
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PL_GLOBAL_SEED'] = str(seed)
    logger.info(f"Set seed to {seed}")

def compute_params(model):
    total_params = 0
    # print("Model parameters:")
    for name in model._names:
        module_params = sum(p.numel() for p in getattr(model, name).parameters()) / 1e6
        # print(f"\t{name}: {module_params:.3f}M")
        total_params += module_params
    # print(f"Total params: {total_params:.3f}M")
    return total_params

def visualize_layout(image_path, layout, size=(240, 350)):
    '''
    parameters:
        image_path: str
        layout: Dict[str, torch.Tensor]
            - boxes: (N, 4)
            - clses: (N)
    '''
    color_dict = {
        'text': 'green',
        'logo': 'red',
        'underlay': 'orange',
        'embellishment': 'blue',
    }
    
    N = len(layout['clses'])
    image = Image.open(image_path).resize(size).convert('RGBA')
    outline = image.copy()
    fill = image.copy()
    
    outline_draw = ImageDraw.Draw(outline)
    fill_draw = ImageDraw.Draw(fill)
    
    for i in range(N):
        color = color_dict[layout['clses'][i]]
        outline_draw.rectangle([layout['boxes'][i][0], layout['boxes'][i][1], layout['boxes'][i][2], layout['boxes'][i][3]], fill=None, outline=color, width=2)
        fill_draw.rectangle([layout['boxes'][i][0], layout['boxes'][i][1], layout['boxes'][i][2], layout['boxes'][i][3]], fill=color)
        
    fill.putalpha(int(256 * 0.4))
    return Image.alpha_composite(outline, fill)