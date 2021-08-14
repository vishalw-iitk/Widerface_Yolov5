from dts.utils.load_the_models import load_the_model
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.datasets import create_dataloader
from yolov5.utils.general import colorstr
from yolov5 import val

import os
import torch
import yaml

@torch.no_grad()
def run(
        weights = None,
        cfg = None,
        device = 'cpu',
        img_size = 416,
        data = 'data.yaml',
        hyp = 'data/hyps/hyp.scratch.yaml',
        single_cls = False,
        project = None,
        name = None
    ):

    
    MLmodel = load_the_model('cpu')
    model_type = 'Regular'
    model_name_user_defined = "Regular trained pytorch model"
    MLmodel.load_pytorch(
        model_path = weights,
        model_name_user_defined = model_name_user_defined,
        cfg = cfg,
        imgsz = img_size,
        data = data,
        hyp = hyp,
        single_cls = single_cls,
        model_class = model_type
    )
    print(MLmodel.statement)
    model = MLmodel.model

    # model = quantized_load(weights, cfg, device, img_size, data, hyp, single_cls)

    ckpt = torch.load(weights)
    fitness_score = ckpt['best_fitness'] if ckpt.get('best_fitness') else None

    with open(data) as f:
        data_dict = yaml.safe_load(f)  # data dict
    
    # Hyperparameters
    # if isinstance(hyp, str):
    with open(hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    batch_size = 4
    # WORLD_SIZE = 2
    WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
    imgsz = img_size
    gs = max(int(model.stride.max()), 32)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    workers = 8

    # Model parameters
    print(hyp)
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = 0.0
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)

    compute_loss = ComputeLoss(model)
    # print("loss value", compute_loss)

    # train_path = data_dict['train']
    val_path = data_dict['val']
    # print("****************")
    # print(val_path)

    val_loader = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, gs,
                                        hyp=hyp, rect=True, rank=-1,
                                        workers=workers, pad=0.5,
                                        cache = True,
                                        prefix=colorstr('val: '))[0]

    results, class_wise_maps, t = val.run(data_dict,
                                batch_size=batch_size // WORLD_SIZE * 2,
                                imgsz=imgsz,
                                model=model,
                                # single_cls=single_cls,
                                dataloader=val_loader,
                                project=project,
                                name = name,
                                # save_dir=Path(save_dir),
                                # conf_thres = 0.0001,
                                # iou_thres = 0.00001,
                                # save_json=is_coco and final_epoch,
                                # verbose=nc < 50 and final_epoch,
                                # plots=plots and final_epoch,
                                # wandb_logger=wandb_logger,
                                compute_loss=compute_loss
                                )

    mp, mr, map50, map, loss, = [results[i] for i in range(0,5)] 
    # from yolov5.utils.metrics import fitness
    # import numpy as np
    # fi = fitness(np.array([mp, mr, map50, map]).reshape(1, -1))
    size = os.stat(weights).st_size/(1024.0*1024.0)

    mAP50, mAP, fitness, size, latency, gflops = map50, map, fitness_score, size, t, None
    print("class_wise_maps", class_wise_maps)
    print("fitness_score", fitness)
    return {'mAP50' : mAP50, 'mAP' : mAP, 'fitness' : fitness, 'size' : size, 'latency' : latency, 'GFLOPS' : gflops}

    # return results, class_wise_maps, fitness_score, t
