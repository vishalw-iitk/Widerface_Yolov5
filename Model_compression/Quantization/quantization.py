'''
    ****************  QUANTIZATION  *******************
    Four Quantization schemes implemented
    1) Quantize aware training  2) Static Post training Quantization(Static PTQ)
    3) Tflite fp16 PTQ          4) Tflite int8 PTQ
'''

''' Importing the libraries '''
import argparse

''' File imports '''
from dts.model_paths import running_model_dictionary
from dts.model_paths import pre_trained_model_dictionary
from dts.model_paths import frameworks
from dts.model_paths import train_results_dictionary
from dts.model_paths import model_defined_names
from dts.Model_compression.Quantization.quantize_common import *


def main(opt):
    '''
    Implementation of four Quantization schemes
    '''
    train_results_paths = train_results_dictionary()
    model_names = model_defined_names()



    '''Pytorch'''

    '''Pytorch Quantize Aware training'''
    if opt.skip_QAT_training == False:
        qat_py = QAT(opt, 'Quantization', 'Pytorch', 'QAT')
        qat_py.quantize(train_results_paths, model_names)

    '''Pytorch Static Post Training Quantization(PTQ)'''
    ptq_py = PTQ(opt, 'Quantization', 'Pytorch', 'PTQ')
    ptq_py.quantize()



    '''Tflite'''

    '''Tflite fp32->fp16 PTQ'''
    tfl_fp16 = TFL_fp16(opt, 'Quantization', 'Tflite', 'fp16')
    tfl_fp16.quantize(model_names)

    '''Tflite fp32->int8 PTQ'''
    tfl_int8 = TFL_int8(opt, 'Quantization', 'Tflite', 'int8')
    tfl_int8.quantize(model_names)
        

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    '''Arguments modifiable from Pipeline.py as well'''

    '''For Pytroch QAT and PTQ'''
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml', help='hyperparameters path')

    '''For Pytorch QAT only'''
    parser.add_argument('--skip-QAT-training', action='store_true', help='skip the time taking Quantizze Aware training training')
    parser.add_argument('--QAT-epochs', type=int, default=30)
    parser.add_argument('--batch-size-QAT', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')

    '''For Pytorch PTQ only'''
    parser.add_argument('--trained-weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--ptq-model-store', type=str, default='../infer_res', help='location where inference results will be stored')


    '''For Tflite fp32->fpint8 Only'''
    parser.add_argument('--repr-images', type=str, default='../ARRANGED_DATASET/images/validation/', help='path of representative dataset')
    parser.add_argument('--imgtf', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--ncalib', type=int, default=100, help='number of calibration images')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    ''' If directly executing quantization.py(Not yet tested) '''
    running_model_paths =  running_model_dictionary()
    pre_trained_model_paths =  pre_trained_model_dictionary()
    opt.framework_path = frameworks(opt.skip_QAT_training, running_model_paths, pre_trained_model_paths)

    return opt


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


