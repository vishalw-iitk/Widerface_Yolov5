# from dts.utils.load_the_models import load_the_model
from yolov5.models.yolo import Model
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.torch_utils import intersect_dicts
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.datasets import create_dataloader
from yolov5.utils.general import colorstr
from yolov5.utils.torch_utils import select_device
from yolov5 import val
from pathlib import Path
import numpy as np
import os
import torch
import yaml

@torch.no_grad()
def run(
        weights = None,
        cfg = None,
        device = 'cpu',
        img_size = 416,
        batch_size = 32,
        data = 'data.yaml',
        hyp = 'data/hyps/hyp.scratch.yaml',
        single_cls = False,
        save_dir = Path(''),
        save_txt = True
    ):
    """
        Return inference results.
    """
    # Load model
    # MLmodel = load_the_model('cpu')
    # model_type = 'Regular'
    # model_name_user_defined = "Regular trained pytorch model"
    # MLmodel.load_pytorch(
    #     model_path = weights,
    #     model_name_user_defined = model_name_user_defined,
    #     model_class = model_type
    # )
    # print(MLmodel.statement)
    # model = MLmodel.model
    with open(data) as f:
        data_dict = yaml.safe_load(f)   # data dict
    
    with open(hyp) as f:
        hyp = yaml.safe_load(f)         # load hyps dict

    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes

    device = select_device(device, batch_size=batch_size)
    model = Model(cfg = cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
    
    ckpt = torch.load(weights, map_location=torch.device(device))

    state_dict = ckpt['model'].state_dict()

    state_dict = intersect_dicts(state_dict, model.state_dict())    # intersect   
    model.load_state_dict(state_dict, strict=False)     

    nc = 1 if single_cls else int(data_dict['nc'])      # number of classes
    WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
    imgsz = img_size
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
    val_loader = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, gs,
                                        hyp=hyp, rect=True, rank=-1,
                                        workers=workers, pad=0.5,
                                        cache = True,
                                        prefix=colorstr('val: '))[0]

    # get output results 
    results, class_wise_maps, t = val.run(data_dict,
                                batch_size=batch_size // WORLD_SIZE * 2,
                                imgsz=imgsz,
                                model=model,
                                dataloader=val_loader,
                                save_dir=save_dir,
                                save_txt = save_txt,
                                compute_loss=compute_loss
                                )

    mp, mr, map50, map, loss, = [results[i] for i in range(0,5)] 

    
    from yolov5.utils.metrics import fitness
    
    fi = fitness(np.array([mp, mr, map50, map]).reshape(1, -1))
    size = os.stat(weights).st_size/(1024.0*1024.0)

    mAP50, mAP, fitness, size, latency, gflops = map50, map, fi[0], size, t, None
    print("class_wise_maps", class_wise_maps)
    print("fitness_score", fitness)
    return {'mAP50' : mAP50, 'mAP' : mAP, 'fitness' : fitness, 'size' : size, 'latency' : latency, 'GFLOPS' : gflops}
