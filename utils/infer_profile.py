import os
import sys
import traceback
import numpy as np
import torch
from copy import deepcopy
from thop import profile


def model_info(model, img_size=416):
    try:  # FLOPs
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        # print(profile(deepcopy(model), inputs=(img,), verbose=False)[
        #       0]/1E9 * 2 * img_size/stride * img_size/stride, profile(deepcopy(model), inputs=(img,), verbose=False))
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = flops * img_size[0] / stride * img_size[1] / stride # 640x640 GFLOPs
    except (ImportError, Exception):
        traceback.print_exc()
        fs = None

    return fs
