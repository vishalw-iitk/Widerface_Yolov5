'''
To perform face detection over WIDERFACE dataset and get the trained model and \
    then optimize the obtained model using Pruning and Quantization.
Model architechture  : Yolov5s
Implementation       :
    -Model trained   :   Pytorch fp16 and fp32 models         |   Tflite fp32 model via export
    -ONNX export     :   Pytorch->ONNX->tf_pb_keras->TFLITE
    -Quantization:
        Pytorch  :   Quantize Aware training(fp32 -> int8)    |   Static Post Training Quantization(PTQ)(fp32 -> int8)
        Tflite   :    fp32 -> fp16(PTQ)                       |   fp32 -> int8(PTQ) (Inference not implemented yet)
    -Pruning     :   Global Unstructured (P1, P2 here)    |   Channel-wise Structured (called P4 here)
'''

''' Importing the libraries '''
import sys
import argparse

''' Adding the file import root to the parent of Pipeline.py '''
sys.path.append('..')

''' Files and functions imports '''
from dts.utils.pipeline_utils import *
from dts.Model_performance import plot_the_performance
from dts.Requirements import requirements


def main(opt):
    '''
    To install the requirements if not already installed.
    This step can be easily avoided if all the requirements are pre-installed beforehand
    '''
    requirements.run()

    '''Data preparation step'''
    data_preparation(opt).run()

    '''
    Now, as the system is setup with Repository, requiremnts and dataset,\
    we can import the yolov5 repo libraries to use in the codes ahead
    '''

    '''Getting the model names and initial model paths'''
    paths = get_paths(opt)

    '''Training the model. Either from scratch, or by using the widerface pre-trained weights'''
    if opt.skip_training == False:
        regular_train(opt, paths).run()

        model_type_conversion(opt, paths).run()

    # Not to use fp16 path onwards
    # Use fp32 path
    '''
    Basic model export step
    Once the pytorch model is trained, we can export it to ONNX -> tf_pb_keras -> TFLITE model
    Same is done with pre-trained models if pre-trained weights are being used
    '''

    model_exportation(opt, paths).run()

    '''
    ****************  PRUNING  *******************
    Two major section of pruning are implemented
    1) Unstructured with different initialization schemes
    2) Structured
    '''
    Pruning_(opt, paths).run()

    '''
    ****************  QUANTIZATION  *******************
    Four Quantization schemes implemented
    1) Quantize aware training
    2) Static Post training Quantization(Static PTQ)
    3) Tflite fp32 PTQ
    4) Tflite int8 PTQ
    '''
    Quantization_(opt, paths).run()

    # Not implemented yet # model_conversion.run('Quantized')

    # Not implemented yet in the pipeline # test_the_models() #detect.py

    '''
    Inference on every model which have been implemented above.
    Getting the inference results stored inside Model performance folder and also return the \
    performance dictionary so as to plot the performance results
    Inference not yet available for Tflite int8 quantized model
    '''
    plot_results = inferencing(opt, paths).run()
    print(plot_results)
    
    # Not implemented yet
    # arrnagement of results like segregation and density_threshold using val.py
    
    '''
    Getting the model performance plots stored inside plot_metrics folder
    '''
    plot_the_performance.run(plot_results=plot_results, save_dir = opt.save_metrics_dir)




def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolov5-repo-name', default = 'yolov5', help='Better not to pass this argument unless the name of the repo itself is changed\
                        Not using this argument and keeping it default is completely fine. yolov5 repo at ../workdir will be deleted to allow cloning\
                        and to deal with old-ultralytics version')
    parser.add_argument('--results-folder_path', default= '../runs', help='training results will be stored inside ..runs/ directory')
    parser.add_argument('--clone-updated-yolov5', action='store_true', help='clone the updated yolov5 repository. This may not work if updates in the original yolv5 repo become incompatible with our setup')
    
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    '''Dataset paths'''
    parser.add_argument('--raw-dataset-path', type=str, default = '../RAW_DATASET', help='Path of the raw dataset which was just arranged from the downloaded dataset')
    parser.add_argument('--arranged-data-path', type=str, default = '../ARRANGED_DATASET', help='Path of the arranged dataset')

    '''Partial dataset selection'''
    parser.add_argument('--partial-dataset', action='store_true', help='willing to select custom percentage of dataset')
    parser.add_argument('--percent-traindata', type=int, default=100, help=' percent_of_the_train_data_required')
    parser.add_argument('--percent-validationdata', type=int, default=100, help=' percent_of_the_validation_data_required')
    parser.add_argument('--percent-testdata', type=int, default=100, help='percent_of_the_test_data_required')

    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--batch-size', type=int, default=128, help='training batch size')

    parser.add_argument('--epochs', type=int, default=250, help='training epochs')   
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer') 
    parser.add_argument('--img-size', type=int, default = 416, help = 'Image size suitable for feeding to the model and train, val image size (pixels)')

    parser.add_argument('--skip-training', action='store_true', help='skip the time taking regular training')
    parser.add_argument('--retrain-on-pre-trained', action='store_true', help= 'Retrain using the pre-trained weights')

    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    
    parser.add_argument('--cfg', type=str, default='../yolov5/models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='../yolov5/data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    
    '''Inference only'''
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')

    '''Pytorch QAT training only'''
    parser.add_argument('--skip-QAT-training', action='store_true', help='skip the time taking Quantizze Aware training training')
    parser.add_argument('--batch-size-QAT', type=int, default=64, help='training batch size for Quantize aware training')
    parser.add_argument('--QAT-epochs', type=int, default=50, help='QAT training epochs')
    
    '''Pytorch QAT and PTQ inference'''
    parser.add_argument('--batch-size-inferquant', type=int, default=32, help='batch size for quantization inference')

    '''Tflite int8 Only'''
    parser.add_argument('--repr-images', type=str, default='../ARRANGED_DATASET/images/validation/', help='path of representative dataset')
    parser.add_argument('--imgtf', nargs='+', type=int, default=[416, 416], help='image size')  # height, width
    parser.add_argument('--ncalib', type=int, default=100, help='number of calibration images')

    '''Pruning'''
    parser.add_argument('--skip-pruning', action='store_true', help='skip the time taking Pruning training')
    parser.add_argument('--skip-P1-training', action='store_true', help='skip the time taking Pruning m1 training')
    parser.add_argument('--skip-P2-training', action='store_true', help='skip the time taking Pruning m2 training')
    parser.add_argument('--skip-P4-training', action='store_true', help='skip the time taking Pruning m4 training')  
    parser.add_argument('--prune-infer-on-pre-pruned-only', action='store_true', help='pruning inference on pre-pruned stored model only and not on recently pruned in pipeline')
    parser.add_argument('--prune-iterations', type=int, default=5, help='prune+retrain total number of iterations') 
    parser.add_argument('--prune-retrain-epochs', type=int, default=100, help=' number of retrain epochs after pruning')
    parser.add_argument('--prune-perc', type=int, default=30, help='initial pruning percentage')
    parser.add_argument('--P4-epochs', type=int, default=1500, help='number of epochs for structured pruning')
    parser.add_argument('--sparsity-training', action='store_true',default=True, help='train with L1 sparsity normalization')
    parser.add_argument('--sparsity-rate', type=float, default=0.001, help='L1 normal sparse rate')

    '''Save metrics dir'''
    parser.add_argument('--save-metrics-dir', type=str, default='../plot_metrics', help='path to save metric plots')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
