import sys
import argparse
from pathlib import Path
import torch
import os

from dts.utils.load_the_models import load_the_model
# from dts.model_paths import framework_path
from yolov5.utils.general import colorstr, check_img_size, check_requirements, file_size


def export_onnx(
    model,
    img,
    file,
    opset_version = 12,
    train = False,
    dynamic = False,
    simplify = False):

    # ONNX model export
    prefix = colorstr('ONNX:')
    try:
        check_requirements(('onnx', 'onnx-simplifier'))
        import onnx

        print(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        f = file.with_suffix('.onnx')
        p =torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL
        print(p)
        torch.onnx.export(model, img, f, verbose=False, opset_version=opset_version,
                          training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=not train,
                          input_names=['images'],
                          output_names=['output'],
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                        'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                                        } if dynamic else None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # print(onnx.helper.printable_graph(model_onnx.graph))  # print

        # Simplify
        if simplify:
            try:
                import onnxsim

                print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={'images': list(img.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print(f'{prefix} simplifier failure: {e}')
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'{prefix} export failure: {e}')


def get_modelutils(weights, img_size = (416, 416), batch_size = 4, half = False, train = False, device = 'cpu'):
    rel_path = '../../..'
    
    file = Path(os.path.join(weights))

    MLmodel = load_the_model('cpu')
    
    
    framework = 'Pytorch'
    model_type = 'QAT quantized'
    model_name_user_defined = "QAT quantized pytorch qint8 model"
    MLmodel.load_pytorch(
        model_path = weights,
        model_name_user_defined = model_name_user_defined,
        cfg = os.path.join(rel_path, 'yolov5/models/yolov5s.yaml'),
        imgsz = 416,
        data = os.path.join(rel_path, 'dts/data.yaml'),
        hyp = os.path.join(rel_path, 'yolov5/data/hyps/hyp.scratch.yaml'),
        single_cls = False,
        model_class = model_type
    )

    print(MLmodel.statement)
    model = MLmodel.model

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    img_size = [check_img_size(x, gs) for x in img_size]  # verify img_size are gs-multiples
    img = torch.zeros(batch_size, 3, *img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    if half:
        img, model = img.half(), model.half()  # to FP16
    model.train() if train else model.eval()  # training mode = no Detect() layer grid construction

    return model, img, file






def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../Quantization/Pytorch/QAT/runs/train/QAT/yolov5s_results6/weights/best.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[416, 416], help='image (height, width)')
    parser.add_argument('--opset-version', type=int, default=12, help='ONNX: opset version')
    parser.add_argument('--dynamic', action='store_true', help='ONNX: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--train', action='store_true', help='model.train() mode')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(opt.img_size)
    model, img, file = get_modelutils(opt.weights, opt.img_size, opt.batch_size, opt.half, opt.train, opt.device)
    export_onnx(model, img, file, opt.opset_version, opt.train, opt.dynamic, opt.simplify)

def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
