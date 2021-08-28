from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.metrics import fitness
import torch
import sys
import yaml
import os
import argparse

from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.models.yolo import Model
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.torch_utils import ModelEMA

from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.general import colorstr
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.loss import ComputeLoss
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.datasets import create_dataloader
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.torch_utils import intersect_dicts
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo import val  # for end-of-epoch mAP

from pathlib import Path

def quantized_load(
        weights = None,
        cfg = None,
        device = 'cpu',
        img_size = 416,
        data = 'data.yaml',
        hyp = 'data/hyps/hyp.scratch.yaml',
        single_cls = False,
        fuseQ = True
    ):


    with open(data) as f:
        data_dict = yaml.safe_load(f)   # data dict
    
    with open(hyp) as f:
        hyp = yaml.safe_load(f)         # load hyps dict
  
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    model = Model(cfg = cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
    model.train()
    quantization_config = torch.quantization.get_default_qat_qconfig("fbgemm")
    model.qconfig = quantization_config
    if fuseQ == True:
        model.fuse()
    torch.quantization.prepare_qat(model, inplace=True)

    imgs = torch.randint(255, (2,3, img_size, img_size))
    imgs = imgs.to(device = device, non_blocking=True).float() / 255.0
    _ = model(imgs)


    model.eval()
    model = torch.quantization.convert(model)

    ckpt = torch.load(weights, map_location=torch.device(device))

    state_dict = ckpt['model']

    state_dict = intersect_dicts(state_dict, model.state_dict())    # intersect   
    model.load_state_dict(state_dict, strict=False)                 # load
    return model

def get_mAP_and_fitness_score(opt):

    model = quantized_load(opt.weights, opt.cfg, opt.device, opt.img_size, opt.data, opt.hyp, opt.single_cls, opt.fuseQ)
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)               # data dict
    
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)                     # load hyps dict

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
    imgsz = opt.img_size
    gs = max(int(model.stride.max()), 32)
    nl = model.model[-1].nl                     # number of detection layers (used for scaling hyp['obj'])
    workers = 8

    # Model parameters
    hyp['box'] *= 3. / nl                       # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl            # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = 0.0
    model.nc = nc                               # attach number of classes to model
    model.hyp = hyp                             # attach hyperparameters to model
    model.gr = 1.0                              # iou loss ratio (obj_loss = 1.0 or iou)
    val_path = data_dict['val']
    val_loader = create_dataloader(val_path, imgsz, opt.batch_size_inferquant // WORLD_SIZE * 2, gs,
                                        hyp=hyp, rect=True, rank=-1,
                                        workers=workers, pad=0.5,
                                        cache = True,
                                        prefix=colorstr('val: '))[0]

    results, class_wise_maps, t = val.run(data_dict,
                                batch_size_QAT = opt.batch_size_inferquant  // WORLD_SIZE * 2,
                                imgsz=imgsz,
                                model=model,
                                dataloader=val_loader,
                                save_dir=opt.save_dir,
                                save_txt = opt.save_txt,
                                )

    return results, class_wise_maps, t


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/QAT/yolov5s_results14/weights/best.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--data', type=str, default='../../../../../val_data.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')

    parser.add_argument('--batch-size-inferquant', type=int, default=32, help='batch size for quantization inference')
    parser.add_argument('--fuseQ', action='store_true', help = 'To infer on the quantized model is fusion is applied')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    results, class_wise_maps, t = get_mAP_and_fitness_score(opt)

    mp, mr, map50, map, loss, = [results[i] for i in range(0,5)] 
    from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.metrics import fitness
    import numpy as np
    fi = fitness(np.array([mp, mr, map50, map]).reshape(1, -1))
    size = os.stat(opt.weights).st_size/(1024.0*1024.0)

    mAP50, mAP, fitness, size, latency, gflops = map50, map, fi[0], size, t, None
    print("class_wise_maps", class_wise_maps)
    print("fitness_score", fitness)
    return {'mAP50' : mAP50, 'mAP' : mAP, 'fitness' : fitness, 'size' : size, 'latency' : latency, 'GFLOPS' : gflops}

def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    return main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
