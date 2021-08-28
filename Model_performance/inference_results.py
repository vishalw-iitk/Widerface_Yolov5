import os
import argparse

from dts.Model_performance.Inference_results_store.Regular.Pytorch import infer_regular
from dts.Model_performance.Inference_results_store.Quantization.Tflite import infer_tfl
from dts.Model_performance.Inference_results_store.Quantization.Pytorch import infer_pyt
from dts.model_paths import infer_results_dictionary
from dts.model_paths import running_model_dictionary
from dts.model_paths import model_defined_names
from pathlib import Path

class model_performance_Results:
    def __init__(self):
        pass

class Regular(model_performance_Results):
    def __init__(self):
        model_performance_Results.__init__(self)

class Quantization(model_performance_Results):
    def __init__(self):
        model_performance_Results.__init__(self)

class Pruning(model_performance_Results):
    def __init__(self):
        model_performance_Results.__init__(self)


# Regular
class PytorchR(Regular):
    def __init__(self):
        Regular.__init__(self)

    def metrics(self, **kwargs):
        results_dictionary = infer_regular.run(**kwargs)
        return results_dictionary

class Tfl_fp32_R(Regular):
    def __init__(self):
        Regular.__init__(self)

    def metrics(self, **kwargs):
        results_dictionary = infer_tfl.run(**kwargs)
        return results_dictionary



# Quantization
class PytorchQ(Quantization):
    def __init__(self):
        Quantization.__init__(self)

class TfliteQ(Quantization):
    def __init__(self):
        Quantization.__init__(self)


# Pytorch Quantization
class QAT_PyQ(PytorchQ):
    def __init__(self):
        PytorchQ.__init__(self)
    def metrics(self, **kwargs):
        results_dictionary = infer_pyt.run(**kwargs)
        return results_dictionary
        
class PTQ_PyQ(PytorchQ):
    def __init__(self):
        PytorchQ.__init__(self)
    def metrics(self, **kwargs):
        # results_dictionary = PTQ_infer.run(**kwargs)
        results_dictionary = infer_pyt.run(**kwargs)
        return results_dictionary


# Tflite Quantization
class Tfl_fp16_Q(Quantization):
    def __init__(self):
        Quantization.__init__(self)

    def metrics(self, **kwargs):
        results_dictionary = infer_tfl.run(**kwargs)
        return results_dictionary

class Tfl_int8_Q(Quantization):
    def __init__(self):
        Quantization.__init__(self)

    def metrics(self, **kwargs):
        results_dictionary = infer_tfl.run(**kwargs)
        return results_dictionary


# Pruning
class PruningP(Pruning):
    def __init__(self):
        Pruning.__init__(self)

class TfliteP(Pruning):
    def __init__(self):
        Pruning.__init__(self)


# Pytorch pruning     
class P1_PyP(PruningP):
    def __init__(self):
        PruningP.__init__(self)
    def metrics(self, **kwargs):
        results_dictionary = infer_regular.run(**kwargs)
        return results_dictionary
        
class P2_PyP(PruningP):
    def __init__(self):
        PruningP.__init__(self)
    def metrics(self, **kwargs):
        results_dictionary = infer_regular.run(**kwargs)
        return results_dictionary

class P4_PyP(PruningP):
    def __init__(self):
        PruningP.__init__(self)
    def metrics(self, **kwargs):
        results_dictionary = infer_regular.run(**kwargs)
        return results_dictionary

# Tflite pruning
class tflm1P(TfliteP):
    def __init__(self):
        TfliteP.__init__(self)
    def metrics(self):
        pass
        
class tflm2P(TfliteP):
    def __init__(self):
        TfliteP.__init__(self)
    def metrics(self):
        pass    

def main(opt):
    
    running_model_metrics = running_model_paths = opt.running_model_paths
    infer_paths = infer_results_dictionary()
    model_name = model_defined_names()

    # Regular pytorch model results
    project = infer_paths['Regular']['Pytorch']['fp32']
    name = model_name['Regular']['Pytorch']['fp32']
    save_dir = os.path.join(project[0], name)
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    (Path(save_dir) / 'labels' if opt.save_txt else Path(save_dir)).mkdir(parents=True, exist_ok=True)
    regularp = PytorchR()
    running_model_metrics['Regular']['Pytorch']['fp32'] = regularp.metrics(
        weights = running_model_paths['Regular']['Pytorch']['fp32'],
        cfg = opt.cfg, data = opt.data, hyp = opt.hyp,
        device = opt.device,
        img_size = opt.img_size, batch_size = opt.batch_size,
        single_cls = opt.single_cls, save_dir = Path(save_dir), save_txt = opt.save_txt,
        conf_thres = opt.conf_thres, iou_thres = opt.iou_thres
    )
    print("the required...regular pyt.")
    print(running_model_metrics['Regular']['Pytorch']['fp32'])


    # Regular model Tflite val results
    os.makedirs(infer_paths['Regular']['Tflite']['fp32']) if not os.path.exists(infer_paths['Regular']['Tflite']['fp32']) else None       
    regulartf = Tfl_fp32_R()
    running_model_metrics['Regular']['Tflite']['fp32'] = regulartf.metrics(
        weights = running_model_paths['Regular']['Tflite']['fp32'],
        data = opt.data,
        device = opt.device,
        imgsz = opt.img_size, batch_size = 1,
        single_cls = opt.single_cls, save_txt = opt.save_txt,
        conf_thres = opt.conf_thres, iou_thres = opt.iou_thres,
        
        verbose = True,
        project = infer_paths['Regular']['Tflite']['fp32'], name = model_name['Regular']['Tflite']['fp32'],
        exist_ok = True, tfl_int8 = False
    )
    print("the required...tfl fp 32.")
    print(running_model_metrics['Regular']['Tflite']['fp32'])

    
    '''*********************  Quantization  ***************'''
    # Pytorch
    # method 1
    
    project = infer_paths['Quantization']['Pytorch']['QAT']
    name = model_name['Quantization']['Pytorch']['QAT']
    save_dir = os.path.join(project, name)
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    (Path(save_dir) / 'labels' if opt.save_txt else Path(save_dir)).mkdir(parents=True, exist_ok=True) 
    qat_py = QAT_PyQ()
    running_model_metrics['Quantization']['Pytorch']['QAT'] = qat_py.metrics(
        weights = running_model_paths['Quantization']['Pytorch']['QAT'],
        cfg = opt.cfg, data = opt.data, hyp = opt.hyp,
        device = 'cpu',
        img_size = opt.img_size, batch_size = opt.batch_size,
        single_cls = opt.single_cls, save_dir = Path(save_dir), save_txt = opt.save_txt,
        conf_thres = opt.conf_thres, iou_thres = opt.iou_thres,
        
        batch_size_inferquant = opt.batch_size_inferquant,
        fuseQ = True
    )
    print("the required...QAT.")
    print(running_model_metrics['Quantization']['Pytorch']['QAT'])

    # method 2
    project = infer_paths['Quantization']['Pytorch']['PTQ']
    name = model_name['Quantization']['Pytorch']['PTQ']
    save_dir = os.path.join(project, name)
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    (Path(save_dir) / 'labels' if opt.save_txt else Path(save_dir)).mkdir(parents=True, exist_ok=True)  # make dir
    ptq_py = PTQ_PyQ()
    running_model_metrics['Quantization']['Pytorch']['PTQ'] = ptq_py.metrics(
        weights = running_model_paths['Quantization']['Pytorch']['PTQ'],
        cfg = opt.cfg, data = opt.data, hyp = opt.hyp,
        device = 'cpu',
        img_size = opt.img_size, batch_size = opt.batch_size,
        single_cls = opt.single_cls, save_dir = Path(save_dir), save_txt = opt.save_txt,
        conf_thres = opt.conf_thres, iou_thres = opt.iou_thres,

        batch_size_inferquant = opt.batch_size_inferquant,
        fuseQ = False
    )
    print("the required....PTQ")
    print(running_model_metrics['Quantization']['Pytorch']['PTQ'])
    
    # method 3
    os.makedirs(infer_paths['Quantization']['Tflite']['fp16']) if not os.path.exists(infer_paths['Quantization']['Tflite']['fp16']) else None
    fp16_Q_tf = Tfl_fp16_Q()
    running_model_metrics['Quantization']['Tflite']['fp16'] = fp16_Q_tf.metrics(
        weights = running_model_paths['Quantization']['Tflite']['fp16'],
        data = opt.data,
        device = opt.device,
        imgsz = opt.img_size, batch_size = 1,
        single_cls = opt.single_cls, save_txt = opt.save_txt,
        conf_thres = opt.conf_thres, iou_thres = opt.iou_thres,
        
        verbose = True,
        project = infer_paths['Quantization']['Tflite']['fp16'], name = model_name['Quantization']['Tflite']['fp16'],
        exist_ok = True, tfl_int8 = False
    )
    print("the required....Tflite_fp16")
    print(running_model_metrics['Quantization']['Tflite']['fp16'])

    '''*********************  Pruning  ***************'''
    # Pytorch
    # method 1

    project = infer_paths['Pruning']['Pytorch']['P1']
    name = model_name['Pruning']['Pytorch']['P1']
    save_dir = os.path.join(project, name)
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    (Path(save_dir) / 'labels' if opt.save_txt else Path(save_dir)).mkdir(parents=True, exist_ok=True)
    p1_reinit = P1_PyP()
    running_model_metrics['Pruning']['Pytorch']['P1'] = p1_reinit.metrics(
        weights = running_model_paths['Pruning']['Pytorch']['P1'],
        cfg = opt.cfg, data = opt.data, hyp = opt.hyp,
        device = opt.device,
        img_size = opt.img_size, batch_size = opt.batch_size,
        single_cls = opt.single_cls, save_dir = Path(save_dir), save_txt = opt.save_txt,
        conf_thres = opt.conf_thres, iou_thres = opt.iou_thres
    )
    print("the required...regular P1")
    print(running_model_metrics['Pruning']['Pytorch']['P1'])

    # method 2
    
    project = infer_paths['Pruning']['Pytorch']['P2']
    name = model_name['Pruning']['Pytorch']['P2']
    save_dir = os.path.join(project, name)
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    (Path(save_dir) / 'labels' if opt.save_txt else Path(save_dir)).mkdir(parents=True, exist_ok=True)
    p2_theta0 = P2_PyP()
    running_model_metrics['Pruning']['Pytorch']['P2'] = p2_theta0.metrics(
        weights = running_model_paths['Pruning']['Pytorch']['P2'],
        cfg = opt.cfg, data = opt.data, hyp = opt.hyp,
        device = opt.device,
        img_size = opt.img_size, batch_size = opt.batch_size,
        single_cls = opt.single_cls, save_dir = Path(save_dir), save_txt = opt.save_txt,
        conf_thres = opt.conf_thres, iou_thres = opt.iou_thres
    )
    print("the required...regular P2")
    print(running_model_metrics['Pruning']['Pytorch']['P2'])

    # method 3

    project = infer_paths['Pruning']['Pytorch']['P4']
    name = model_name['Pruning']['Pytorch']['P4']
    save_dir = os.path.join(project, name)
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    (Path(save_dir) / 'labels' if opt.save_txt else Path(save_dir)).mkdir(parents=True, exist_ok=True)
    p4_layerprune = P4_PyP()
    running_model_metrics['Pruning']['Pytorch']['P4'] = p4_layerprune.metrics(
        weights = running_model_paths['Pruning']['Pytorch']['P4'],
        cfg = opt.cfg, data = opt.data, hyp = opt.hyp,
        device = opt.device,
        img_size = opt.img_size, batch_size = opt.batch_size,
        single_cls = opt.single_cls, save_dir = Path(save_dir), save_txt = opt.save_txt,
        conf_thres = opt.conf_thres, iou_thres = opt.iou_thres
    )
    print("the required...regular P4")
    print(running_model_metrics['Pruning']['Pytorch']['P4'])

    del running_model_metrics['Regular']['Pytorch']['fp16']
    del running_model_metrics['Quantization']['Tflite']['int8']
    del running_model_metrics['Pruning']['Tflite']
    
    return running_model_metrics

    

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--batch-size', type=int, default=128, help='training batch size')
    parser.add_argument('--img-size', type=int, default = 416, help = 'Image size suitable for feeding to the model and train, val image size (pixels)')
    
    parser.add_argument('--cfg', type=str, default='../yolov5/models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='../yolov5/data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    running_model_paths = running_model_dictionary()
    opt.running_model_paths = running_model_paths

    return opt

def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    return main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
