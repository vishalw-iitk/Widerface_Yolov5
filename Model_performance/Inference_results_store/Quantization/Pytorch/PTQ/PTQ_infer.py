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
        single_cls = False
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
    # model.fuse()
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
        name = None
    ):

    # save_dir = name

    model = quantized_load(weights, cfg, device, img_size, data, hyp, single_cls)


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
    results, class_wise_maps, t = get_mAP_and_fitness_score(
            weights = opt.weights,
            cfg = opt.cfg,
            device = 'cpu',
            img_size = opt.img_size,
            data = opt.data,
            hyp = opt.hyp,
            single_cls = opt.single_cls,
            project = opt.project,
            name = opt.name
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














# import argparse
# from dts.model_paths import model_defined_names
# from pathlib import Path
# from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.models.yolo import Model
# from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.torch_utils import intersect_dicts
# import yaml
# import os
# import torch
# from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.datasets import create_dataloader, LoadImages
# from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.general import colorstr
# from pathlib import Path
# from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo import val

# def get_mAP(
#         weights = None,
#         cfg = None,
#         device = 'cpu',
#         img_size = None,
#         data = None,
#         hyp = None,
#         single_cls = None,
#         project = None,
#         name = None
#     ):
    

#     ckpt = torch.load(weights, map_location=torch.device(device))

#     with open(data) as f:
#     data_dict = yaml.safe_load(f)  # data dict
    
#     # Hyperparameters
#     # if isinstance(hyp, str):
#     with open(hyp) as f:
#         hyp = yaml.safe_load(f)  # load hyps dict
    
#         nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
#     batch_size = 4
#     # WORLD_SIZE = 2
#     WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
#     imgsz = img_size
#     gs = max(int(model.stride.max()), 32)
#     nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
#     workers = 8

#         # Model parameters
#     hyp['box'] *= 3. / nl  # scale to layers
#     hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
#     hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
#     hyp['label_smoothing'] = 0.0
#     model.nc = nc  # attach number of classes to model
#     model.hyp = hyp  # attach hyperparameters to model
#     model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)

#     val_path = data_dict['val']

#     val_loader = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, gs,
#                                     hyp=hyp, rect=True, rank=-1,
#                                     workers=workers, pad=0.5,
#                                     cache = True,
#                                     prefix=colorstr('val: '))[0]

#     results, class_wise_maps, t = val.run(data_dict,
#                             batch_size=batch_size // WORLD_SIZE * 2,
#                             imgsz=imgsz,
#                             model=model,
#                             # single_cls=single_cls,
#                             dataloader=val_loader,
#                             project=project,
#                             name = name,
#                             )
    
#     return results, class_wise_maps, t

# def main(opt):

#     results, class_wise_maps, t = get_mAP(
#             weights = opt.weights,
#             cfg = opt.cfg,
#             device = 'cpu',
#             img_size = opt.img_size,
#             data = opt.data,
#             hyp = opt.hyp,
#             single_cls = opt.single_cls,
#             project = opt.project,
#             name = opt.name
#         )
#     mp, mr, map50, map, loss, = [results[i] for i in range(0,5)] 
#     from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.metrics import fitness
#     import numpy as np
#     fi = fitness(np.array([mp, mr, map50, map]).reshape(1, -1))
#     size = os.stat(opt.weights).st_size/(1024.0*1024.0)

#     mAP50, mAP, fitness, size, latency, gflops = map50, map, fi, size, t, None
#     print("class_wise_maps", class_wise_maps)
#     print("fitness_score", fitness)
#     return {'mAP50' : mAP50, 'mAP' : mAP, 'fitness' : fitness, 'size' : size, 'latency' : latency, 'GFLOPS' : gflops}
# # =====

# def temp():
#     opt.weights = str(Path(opt.weights).absolute())
#     # opt.results = str(Path(opt.results).absolute())
#     # os.makedirs(opt.results.replace('/best.pt', '')) if not os.path.exists(opt.results.replace('/best.pt', '')) else None
#     # cfg = "models/yolov5s.yaml"
#     cfg = opt.cfg
#     # hyp = 'data/hyps/hyp.scratch.yaml'
#     hyp = opt.hyp
#     # device = 'cpu'
#     device = opt.device
#     # data = 'data_widerface.yaml'
#     data = opt.data

#     with open(data) as f:
#             data_dict = yaml.safe_load(f) 
#     with open(hyp) as f:
#             hyp = yaml.safe_load(f)
#     nc = 1 # number of classes
#     # exclude = ['anchor'] # exclude keys

#     # print(model)
#     # print(model.state_dict())

#     ###PTQ
#     # create a model instance
#     model = Model(cfg = cfg , ch=3, nc=nc, anchors=hyp.get('anchors')).to(device) ###creating architecture instance
#     # model must be set to eval mode for static quantization logic to work
#     model.eval()
#     # ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
#     # csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
#     # csd = intersect_dicts(csd, model.state_dict())  # intersect
#     # model.load_state_dict(csd, strict=False) ##load checkpoints into created architecture

#     # print(model)
#     model_fp32 = model
#     # attach a global qconfig, which contains information about what kind
#     # of observers to attach. Use 'fbgemm' for server inference and
#     # 'qnnpack' for mobile inference. Other quantization configurations such
#     # as selecting symmetric or assymetric quantization and MinMax or L2Norm
#     # calibration techniques can be specified here.
#     model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

#     # Fuse the activations to preceding layers, where applicable.
#     # This needs to be done manually depending on the model architecture.
#     # Common fusions include `conv + relu` and `conv + batchnorm + relu`
#     # model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv', 'bn']])

#     # print(model_fp32_fused)

#     # Prepare the model for static quantization. This inserts observers in
#     # the model that will observe activation tensors during calibration.
#     model_fp32_prepared = torch.quantization.prepare(model_fp32)


#     # print(model_fp32_prepared)
#     dataset = LoadImages(data_dict['train'], img_size=416, stride=32)
#     for path, img, im0s, vid_cap in dataset:
#         img = torch.from_numpy(img).to(device)
#         img = img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)
#         # model_fp32_prepared(img)
#         # calibrate
#         model_fp32_prepared(img)


#     model_int8 = torch.quantization.convert(model_fp32_prepared)
#     # loading int8 ckpt
#     ckpt_PTQ = torch.load(opt.weights, map_location=device)  # load checkpoint
#     csd_PTQ = ckpt_PTQ['model']  # checkpoint state_dict as FP32
#     csd_PTQ = intersect_dicts(csd_PTQ, model_int8.state_dict())  # intersect
#     model_int8.load_state_dict(csd_PTQ, strict=False) ##load checkpoints into created architecture

#     # ckpt = {
#     #     'model' : model_int8.state_dict()
#     # }
#     # torch.save(ckpt, os.path.join(opt.results))
#     ###validation
# # # =======================================================================================
#     batch_size = 4
#     WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
#     imgsz = 416
#     val_path = data_dict['val']
#     gs = max(int(model.stride.max()), 32)
#     workers = 8
#     single_cls = False
#     # compute_loss = ComputeLoss(model)  # init loss class
#     val_loader = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
#                                         hyp=hyp, cache='true', rect=True, rank=-1,
#                                         workers=workers, pad=0.5,
#                                         prefix=colorstr('val: '))[0]
#     ###
#     if not os.path.exists(opt.project):
#         os.makedirs(opt.project)
#     results, class_wise_maps, t = val.run(data_dict,
#                                     batch_size=batch_size // WORLD_SIZE * 2,
#                                     imgsz=imgsz,
#                                     model=model_int8,
#                                     dataloader=val_loader,
#                                     project= opt.project,
#                                     name = opt.name,
#                                     # save_dir=Path(opt.results),
#                                     # iou_thres = 0.7,
#                                     compute_loss=None
#                                     )
#     # print("class_wise_maps", results)
#     # size = os.stat(os.path.join(opt.weights)).st_size/(1024.0*1024.0)
#     # print(size)
#     # shape = (batch_size, 3, imgsz, imgsz)
#     # print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)
#     # val_res = {'mAP50' : results[2], 'mAP' : results[3], 'fitness' : None, 'size' : str(size)+"MB", 'latency' : str(t[1])+"ms", 'GFLOPS' : None}
#     # print("PTQ: ", val_res)
#     # =========================================================================================
#     return results, class_wise_maps, t
#     # https://github.com/pytorch/pytorch/issues/20756


# def parse_opt(known = False):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', type=str, help='best_widerface_int8.pt')
#     parser.add_argument('--results', type=str, help='val_results')

#     # opt = parser.parse_args()
#     opt = parser.parse_known_args()[0] if known else parser.parse_args()
#     return opt

# def run(**kwargs):
#     # Usage: import train; train.run(imgsz=320, weights='yolov5m.pt')
#     opt = parse_opt(True)
#     for k, v in kwargs.items():
#         setattr(opt, k, v)
#     main(opt)


# if __name__ == "__main__":
#     opt = parse_opt()
#     results = main(opt)
#     print(results) 