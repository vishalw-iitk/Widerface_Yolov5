from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.metrics import fitness
import torch
import sys
import yaml
import os
import argparse

# sys.path.append("../../../../..")
# from dts.Model_performance.Inference_results_store.Quantization.Pytorch.load_and_infer
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.models.yolo import Model
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.torch_utils import ModelEMA

from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.general import colorstr
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.loss import ComputeLoss
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.datasets import create_dataloader
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.torch_utils import intersect_dicts
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo import val  # for end-of-epoch mAP

from pathlib import Path

# class pytorch_load_quantized_model:
#     def __init__():
def quantized_load(
        weights = None,
        cfg = None,
        device = 'cpu',
        img_size = 416,
        data = 'data.yaml',
        hyp = 'data/hyps/hyp.scratch.yaml',
        single_cls = False,
        fuse = True
    ):

    # rel_path = '../../../../..'

    with open(data) as f:
        data_dict = yaml.safe_load(f)  # data dict
    
    # Hyperparameters
    # if isinstance(hyp, str):
    with open(hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    # x = torch.load(f, map_location=torch.device('cpu'))
    
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    model = Model(cfg = cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
    model.train()
    quantization_config = torch.quantization.get_default_qat_qconfig("fbgemm")
    model.qconfig = quantization_config
    if fuse == True:
        model.fuse()
    torch.quantization.prepare_qat(model, inplace=True)

    imgs = torch.randint(255, (2,3, img_size, img_size))
    imgs = imgs.to(device = 'cpu', non_blocking=True).float() / 255.0
    # nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    # model = Model(cfg = cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
    # necessary to fix the min_max tensors shape
    _ = model(imgs)


    model.eval()
    model = torch.quantization.convert(model)

    # model.load_state_dict(x['model'])



    # model.train()

    # quantization_config = torch.quantization.get_default_qat_qconfig("fbgemm")
    # model.qconfig = quantization_config
    # torch.quantization.prepare_qat(model, inplace=True)

    # model.eval()

    # model = torch.quantization.convert(model)
    
    # print("is it errr")
    # pred = model(imgs)
    # print("pred1", pred.shape)

    ckpt = torch.load(weights, map_location=torch.device(device))

    state_dict = ckpt['model']

    state_dict = intersect_dicts(state_dict, model.state_dict())  # intersect   
    model.load_state_dict(state_dict, strict=False)  # load

    # print(model)
    # pred = model(imgs)
    # print("pred2", pred.shape)

    # c = 2
    # for i ,j in zip(state_dict.keys(), model.state_dict().keys()):
    #     c = c - 1
    #     print(state_dict[i])
    #     print(model.state_dict()[i])
    #     print('\n')
    #     if c == 0:
    #         break
        

    return model

def get_mAP_and_fitness_score(
        weights = None,
        cfg = None,
        device = 'cpu',
        img_size = 416,
        data = 'data.yaml',
        hyp = 'data/hyps/hyp.scratch.yaml',
        single_cls = False,
        project = None,
        name = None,
        fuse = True
    ):

    # save_dir = name

    model = quantized_load(weights, cfg, device, img_size, data, hyp, single_cls, fuse)


    # imgs = torch.randint(255, (1,3, img_size, img_size))
    # import numpy as np
    # imgs = np.random.randint(256, size=(1,3, img_size, img_size)) # 0 to 255
    # imgs = torch.from_numpy(imgs).to(device)
    # imgs = imgs.float()  # uint8 to fp16/32
    # _ = model(imgs)
    # print("pred shape", pred.shape)


    # ckpt = torch.load(weights, map_location=torch.device(device))
    # fitness_score = ckpt['best_fitness'] if ckpt.get('best_fitness') else None

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
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = 0.0
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)

    # compute_loss = ComputeLoss(model)
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
                                # compute_loss=compute_loss
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
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    # model = quantized_load(
    #     weights = opt.weights,
    #     cfg = opt.cfg,
    #     device = 'cpu',
    #     img_size = opt.img_size,
    #     data = opt.data,
    #     hyp = opt.hyp,
    #     single_cls = opt.single_cls
    # )
    # imgs = torch.randint(255, (1,3, opt.img_size, opt.img_size))
    # # pred = model(imgs)
    # # print("pred shape", pred.shape)
    
    # from flopth import flopth
    # try:
    #     print("opoppp")
    #     pred = model(imgs.float()/255.0)
    #     print(list(pred.shape))
    #     sum_flops = flopth(model, in_size=[[1, 3, 416, 416], list(pred.shape)])
    #     print(sum_flops)
    # except Exception as e:
    #     print(e)

    # print(model)
    results, class_wise_maps, fitness, t = get_mAP_and_fitness_score(
            weights = opt.weights,
            cfg = opt.cfg,
            device = 'cpu',
            img_size = opt.img_size,
            data = opt.data,
            hyp = opt.hyp,
            single_cls = opt.single_cls,
            project = opt.project,
            name = opt.name,
            fuse = opt.fuse
        )
    mp, mr, map50, map, loss, = [results[i] for i in range(0,5)] 
    from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.metrics import fitness
    import numpy as np
    fi = fitness(np.array([mp, mr, map50, map]).reshape(1, -1))
    size = os.stat(opt.weights).st_size/(1024.0*1024.0)

    mAP50, mAP, fitness, size, latency, gflops = map50, map, fi, size, t, None
    print("class_wise_maps", class_wise_maps)
    print("fitness_score", fitness)
    return {'mAP50' : mAP50, 'mAP' : mAP, 'fitness' : fitness, 'size' : size, 'latency' : latency, 'GFLOPS' : gflops}

def run(**kwargs):
    # Usage: import train; train.run(imgsz=416, weights='best.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    return main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
