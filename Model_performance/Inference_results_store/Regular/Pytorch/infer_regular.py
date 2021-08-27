# from dts.utils.load_the_models import load_the_model
from yolov5.models.yolo import Model
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.torch_utils import intersect_dicts
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.datasets import create_dataloader
from yolov5.utils.general import colorstr
from yolov5.utils.torch_utils import select_device
from yolov5 import val
from pathlib import Path

import argparse
import numpy as np
import os
import torch
import yaml

@torch.no_grad()
def main(opt):
    """
        Return inference results.
    """
    if type(opt.save_dir) == type('stringtype'):
        opt.save_dir = Path(opt.save_dir)

    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)   # data dict
    
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)         # load hyps dict

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes

    device = select_device(opt.device, batch_size=opt.batch_size)
    model = Model(cfg = opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
    
    ckpt = torch.load(opt.weights, map_location=torch.device(device))

    state_dict = ckpt['model'].state_dict()

    state_dict = intersect_dicts(state_dict, model.state_dict())    # intersect   
    model.load_state_dict(state_dict, strict=False)     

    nc = 1 if opt.single_cls else int(data_dict['nc'])      # number of classes
    WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
    imgsz = opt.img_size
    gs = max(int(model.stride.max()), 32)
    nl = model.model[-1].nl                             # number of detection layers (used for scaling hyp['obj'])
    workers = 8

    # Model parameters
    hyp['box'] *= 3. / nl                               # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl                    # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl          # scale to image size and layers
    hyp['label_smoothing'] = 0.0
    model.nc = nc                                       # attach number of classes to model
    model.hyp = hyp                                     # attach hyperparameters to model
    model.gr = 1.0                                      # iou loss ratio (obj_loss = 1.0 or iou)

    compute_loss = ComputeLoss(model)

    val_path = data_dict['val']
    # load images
    val_loader = create_dataloader(val_path, imgsz, opt.batch_size // WORLD_SIZE * 2, gs,
                                        hyp=hyp, rect=True, rank=-1,
                                        workers=workers, pad=0.5,
                                        cache = True,
                                        prefix=colorstr('val: '))[0]

    # get output results 
    results, class_wise_maps, t = val.run(data_dict,
                                batch_size=opt.batch_size // WORLD_SIZE * 2,
                                imgsz=imgsz,
                                model=model,
                                dataloader=val_loader,
                                save_dir=opt.save_dir,
                                save_txt = opt.save_txt,
                                compute_loss=compute_loss
                                )

    mp, mr, map50, map, loss, = [results[i] for i in range(0,5)] 

    
    from yolov5.utils.metrics import fitness
    
    fi = fitness(np.array([mp, mr, map50, map]).reshape(1, -1))
    size = os.stat(opt.weights).st_size/(1024.0*1024.0)

    mAP50, mAP, fitness, size, latency, gflops = map50, map, fi[0], size, t, None
    print("class_wise_maps", class_wise_maps)
    print("fitness_score", fitness)
    return {'mAP50' : mAP50, 'mAP' : mAP, 'fitness' : fitness, 'size' : size, 'latency' : latency, 'GFLOPS' : gflops}

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--batch-size', type=int, default=128, help='training batch size')
    parser.add_argument('--img-size', type=int, default = 416, help = 'Image size suitable for feeding to the model and train, val image size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    
    parser.add_argument('--cfg', type=str, default='../yolov5/models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='../yolov5/data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--save-dir', type=str, default='../infer_res', help='location where inference results will be stored')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    return opt

def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
