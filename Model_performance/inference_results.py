import argparse

from dts.model_paths import infer_results_dictionary
from dts.model_paths import running_model_dictionary
from dts.model_paths import model_defined_names
from dts.Model_performance.Inference_results_store.infer_common import *



def main(opt):
    
    running_model_metrics = opt.running_model_paths
    infer_paths = infer_results_dictionary()
    model_names = model_defined_names()

    '''**************************************************************************************************'''
    '''*******************  Regular  ********************'''

    '''************  Regular Pytorch fp32  **************'''

    # classmethod(PytorchR(opt, 'Regular', 'Pytorch', 'fp32').explicit_inference(infer_paths, model_names, running_model_metrics))
    regularp = PytorchR(opt, 'Regular', 'Pytorch', 'fp32')
    regularp.explicit_inference(infer_paths, model_names, running_model_metrics)


    '''***********  Regular Tflite fp32  ***************'''

    regulartf = Tfl_fp32_R(opt, 'Regular', 'Tflite', 'fp32')
    regulartf.explicit_inference(infer_paths, model_names, running_model_metrics)


    '''**************************************************************************************************'''
    '''************************* Quantization  ****************************'''

    '''*************  Pytorch Quantization  ****************'''
    # Pytorch
    Pytorch_Quantization = [PytorchQ(opt, 'Quantization', 'Pytorch', 'QAT'),\
                            PytorchQ(opt, 'Quantization', 'Pytorch', 'PTQ')]
    for obj in Pytorch_Quantization:
        obj.explicit_inference(infer_paths, model_names, running_model_metrics)


    '''************  Tflite Quantization  ****************'''
    # method 3
    fp16_Q_tf = TfliteQ(opt, 'Quantization', 'Tflie', 'fp16')
    fp16_Q_tf.explicit_inference(infer_paths, model_names, running_model_metrics)



    '''**************************************************************************************************'''
    '''************************* Pruning **********************************'''
    # Pytorch
    Pytorch_Pruning = [Pruning(opt, 'Pruning', 'Pytorch', 'P1'),\
                       Pruning(opt, 'Pruning', 'Pytorch', 'P2'),\
                       Pruning(opt, 'Pruning', 'Pytorch', 'P4')]
    for obj in Pytorch_Pruning:
        obj.explicit_inference(infer_paths, model_names, running_model_metrics)


    '''**************************************************************************************************'''
    '''***********  filtering the plot-keys  ***************************'''
    model_performance_Results.unused_plot_keys(
        running_model_metrics['Regular']['Pytorch']['fp16'],
        running_model_metrics['Quantization']['Tflite']['int8'],
        running_model_metrics['Pruning']['Tflite']
    )

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
