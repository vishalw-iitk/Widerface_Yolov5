import argparse
from pathlib import Path
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.models.yolo import Model
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.torch_utils import intersect_dicts
import yaml
import os
import torch
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.datasets import create_dataloader, LoadImages
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.general import colorstr
from pathlib import Path
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo import val

def main(opt):
    opt.weights = str(Path(opt.weights).absolute())
    opt.results = str(Path(opt.results).absolute())
    # cfg = "models/yolov5s.yaml"
    cfg = opt.cfg
    # hyp = 'data/hyps/hyp.scratch.yaml'
    hyp = opt.hyp
    # device = 'cpu'
    device = opt.device
    # data = 'data_widerface.yaml'
    data = opt.data

    with open(data) as f:
            data_dict = yaml.safe_load(f) 
    with open(hyp) as f:
            hyp = yaml.safe_load(f)
    nc = 1 # number of classes
    exclude = ['anchor'] # exclude keys

    # print(model)
    # print(model.state_dict())

    ###PTQ
    # create a model instance
    model = Model(cfg = cfg , ch=3, nc=nc, anchors=hyp.get('anchors')).to(device) ###creating architecture instance
    # model must be set to eval mode for static quantization logic to work
    model.eval()
    ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(csd, strict=False) ##load checkpoints into created architecture

    # print(model)
    model_fp32 = model
    # attach a global qconfig, which contains information about what kind
    # of observers to attach. Use 'fbgemm' for server inference and
    # 'qnnpack' for mobile inference. Other quantization configurations such
    # as selecting symmetric or assymetric quantization and MinMax or L2Norm
    # calibration techniques can be specified here.
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Fuse the activations to preceding layers, where applicable.
    # This needs to be done manually depending on the model architecture.
    # Common fusions include `conv + relu` and `conv + batchnorm + relu`
    # model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv', 'bn']])

    # print(model_fp32_fused)

    # Prepare the model for static quantization. This inserts observers in
    # the model that will observe activation tensors during calibration.
    model_fp32_prepared = torch.quantization.prepare(model_fp32)


    # print(model_fp32_prepared)
    dataset = LoadImages(data_dict['train'], img_size=416, stride=32)
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # model_fp32_prepared(img)
        # calibrate
        model_fp32_prepared(img)
        

    model_int8 = torch.quantization.convert(model_fp32_prepared)
    torch.save(model_int8.state_dict(),os.path.join(opt.results))
    ###validation

    batch_size = 4
    WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
    imgsz = 416
    val_path = data_dict['val']
    gs = max(int(model.stride.max()), 32)
    workers = 8
    single_cls = False
    # compute_loss = ComputeLoss(model)  # init loss class
    val_loader = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
                                        hyp=hyp, cache='true', rect=True, rank=-1,
                                        workers=workers, pad=0.5,
                                        prefix=colorstr('val: '))[0]
    ###
    if not os.path.exists(opt.results):
        os.makedirs(opt.results)
    results, class_wise_maps, t = val.run(data_dict,
                                    batch_size=batch_size // WORLD_SIZE * 2,
                                    imgsz=imgsz,
                                    model=model_int8,
                                    single_cls=False,
                                    dataloader=val_loader,
                                    save_dir=Path(opt.results),
                                    iou_thres = 0.7,
                                    compute_loss=None
                                    )
    print("class_wise_maps", results)
    size = os.stat(os.path.join(opt.results,"PTQ_Int8_widerface.pt")).st_size/(1024.0*1024.0)
    print(size)
    shape = (batch_size, 3, imgsz, imgsz)
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)
    val_res = {'mAP50' : results[2], 'mAP' : results[3], 'fitness' : None, 'size' : str(size)+"MB", 'latency' : str(t[1])+"ms", 'GFLOPS' : None}
    print("PTQ: ", val_res)
    # return val_res
    # https://github.com/pytorch/pytorch/issues/20756


def parse_opt(known = False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='best_widerface_f32.pt')
    parser.add_argument('--results', type=str, help='val_results')
    
    # opt = parser.parse_args()
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def run(**kwargs):
    # Usage: import train; train.run(imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    results = main(opt)
    print(results)