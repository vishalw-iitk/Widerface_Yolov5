'''
    ****************  QUANTIZATION  *******************
    Four Quantization schemes implemented
    1) Quantize aware training  2) Static Post training Quantization(Static PTQ)
    3) Tflite fp16 PTQ          4) Tflite int8 PTQ
'''

''' Importing the libraries '''
import argparse

''' File imports '''
from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo import train
from dts.Model_compression.Quantization.Pytorch.PTQ import PT_quant
from dts.Model_conversion import model_export
from dts.model_paths import running_model_dictionary
from dts.model_paths import pre_trained_model_dictionary
from dts.model_paths import frameworks
from dts.model_paths import train_results_dictionary
from dts.model_paths import model_defined_names

class Quantization(object):
    def __init__(self):
        pass

class Pytorch(Quantization):
    def __init__(self):
        Quantization.__init__(self)

class Tflite(Quantization):
    def __init__(self):
        Quantization.__init__(self)


'''Pytorch Quantize Aware training'''
class QAT(Pytorch):
    def __init__(self):
        Pytorch.__init__(self)
    def quantize(self, **kwargs):
        train.run(**kwargs)

'''Pytorch Static Post Training Quantization(PTQ)'''    
class PTQ(Pytorch):
    def __init__(self):
        Pytorch.__init__(self)
    def quantize(self, **kwargs):
        PT_quant.run(**kwargs)


'''Tflite fp32->fp16 PTQ'''
class TFL_fp16(Tflite):
    def __init__(self):
        Tflite.__init__(self)
    def quantize(self, **kwargs):
        model_export.run(**kwargs)

'''Tflite fp32->int8 PTQ'''
class TFL_int8(Tflite):
    def __init__(self):
        Tflite.__init__(self)
    def quantize(self, **kwargs):
        model_export.run(**kwargs)




def main(opt):
    '''
    Implementation of four Quantization schemes
    '''

    train_results_paths = train_results_dictionary()
    model_names = model_defined_names()

    running_model_paths = opt.running_model_paths
    framework_path = opt.framework_path
    
    '''Pytorch'''

    '''Pytorch Quantize Aware training'''
    if opt.skip_QAT_training == False:
        qat_py = QAT()
        qat_py.quantize(
                weights = opt.weights,
                
                cfg = opt.cfg, data = opt.data, hyp = opt.hyp,
                batch_size_QAT = opt.batch_size_QAT, QAT_epochs = opt.QAT_epochs,
                img_size = opt.img_size, cache_images = opt.cache_images,
                device = opt.device,
                single_cls = opt.single_cls, adam = opt.adam, workers = opt.workers,

                project = train_results_paths['Quantization']['Pytorch']['QAT'],
                name = model_names['Quantization']['Pytorch']['QAT'],
                )

    '''Pytorch Static Post Training Quantization(PTQ)'''
    ptq_py = PTQ()
    ptq_py.quantize(
        weights = opt.weights,
        cfg = opt.cfg, hyp = opt.hyp, device = 'cpu', data = opt.data,
        results = running_model_paths['Quantization']['Pytorch']['PTQ'],
    )



    '''Tflite'''

    '''Tflite fp32->fp16 PTQ'''
    tfl_fp16 = TFL_fp16()
    tfl_fp16.quantize(
        model_type_for_export = model_names['Quantization']['Tflite']['fp16'],
        framework_path = framework_path,
        model_names = model_names
        )

    '''Tflite fp32->int8 PTQ'''
    tfl_int8 = TFL_int8()
    tfl_int8.quantize(
        model_type_for_export = model_names['Quantization']['Tflite']['int8'],
        framework_path = framework_path,
        model_names = model_names,
        repr_images = opt.repr_images, imgtf = opt.imgtf, ncalib = opt.ncalib
    )
        

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

    '''For Tflite fp32->fpint8 Only'''
    parser.add_argument('--repr-images', type=str, default='../ARRANGED_DATASET/images/validation/', help='path of representative dataset')
    parser.add_argument('--imgtf', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--ncalib', type=int, default=100, help='number of calibration images')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    ''' If directly executing quantization.py(Not yet tested) '''
    running_model_paths =  running_model_dictionary()
    pre_trained_model_paths =  pre_trained_model_dictionary()
    opt.framework_path = frameworks(opt.skip_QAT_training, running_model_paths, pre_trained_model_paths)
    opt.running_model_paths = running_model_paths

    return opt


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


