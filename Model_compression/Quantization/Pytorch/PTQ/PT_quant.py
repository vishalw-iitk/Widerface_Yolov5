import argparse
import yaml
import torch
from pathlib import Path
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.models.yolo import Model
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.torch_utils import intersect_dicts
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.utils.datasets import  LoadImages

def main(opt):
    """creates a Quantized model using Post Training Quantization approach.

    Arguments:
        weights (str): name of float32 or floart16 model, i.e. 'best.pt'.
        results (str): name of directory to store  quantized int8 model.

    """

    if '/best.pt' in opt.results:
        opt.results = opt.results.replace('/best.pt', '')
    elif r'\best.pt' in opt.results:
        opt.results = opt.results.replace(r'\best.pt', '')
    
    # Get requied paths of files/directory.
    weights = Path(opt.weights)
    store_result = Path(opt.results)
    print("print opt results", opt.results)

    store_result.mkdir(parents=True, exist_ok=True)
    cfg = opt.cfg
    hyp = opt.hyp
    device = opt.device
    data = opt.data

    with open(data) as f:
            data_dict = yaml.safe_load(f) 
    with open(hyp) as f:
            hyp = yaml.safe_load(f)
    nc = 1 # number of classes
    
    # *************************************PTQ************************************
    # create a model instance
    model = Model(cfg = cfg , ch=3, nc=nc, anchors=hyp.get('anchors')).to(device) ###creating architecture instance
    # model must be set to eval mode for static quantization logic to work
    model.eval()
    ckpt = torch.load(weights, map_location=device)     # load checkpoint
    csd = ckpt['model'].float().state_dict()            # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict())      # intersect
    model.load_state_dict(csd, strict=False)            # load checkpoints into created architecture

    model_fp32 = model

    """
    Step 1:
        attach a global qconfig, which contains information about what kind of observers to attach. Use 'fbgemm' for server inference and
        'qnnpack' for mobile inference. Other quantization configurations such as selecting symmetric or assymetric quantization and MinMax or L2Norm
        calibration techniques can be specified here.
    """

    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    """
    Step 2: (optional)
        Fuse the activations to preceding layers, where applicable. This needs to be done manually depending on the model architecture.
        Common fusions include `conv + relu` and `conv + batchnorm + relu`.
        # model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv', 'bn']])
    """
    
    """
    Step 3:
        Prepare the model for static quantization. This inserts observers in the model that will observe activation tensors during calibration.

    """

    model_fp32_prepared = torch.quantization.prepare(model_fp32)


    """
    Step 4:
        calibrate the prepared model to determine quantization parameters for activations in a real world setting, the calibration would be done with a representative dataset
    """

    dataset = LoadImages(data_dict['train'], img_size=416, stride=32)
    for img in dataset[0:100]:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        model_fp32_prepared(img)
    
    """
    Step 5:
        Convert the observed model to a quantized model. This does several things:
        quantizes the weights, computes and stores the scale and bias value to be used with each activation tensor, and replaces key operators with quantized implementations.
    """

    model_int8 = torch.quantization.convert(model_fp32_prepared)

    # Saving converted model
    ckpt = {
        'model' : model_int8.state_dict()
    }
    best = store_result / 'best.pt'
    torch.save(ckpt, best)


def parse_opt(known = False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='best_widerface_f32.pt')
    parser.add_argument('--results', type=str, help='val_results')
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