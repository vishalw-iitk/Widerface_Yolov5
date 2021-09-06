from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo import train
from dts.Model_compression.Quantization.Pytorch.PTQ import PT_quant
from dts.Model_conversion import model_export

class Quantization:
    def __init__(self, technique, framework, model_name):
        self.technique, self.framework, self.model_name = technique, framework, model_name

    def model_args(self):
        self.__dict__.update({k:v if type(v) is not tuple else v[0] for k,v in self.__dict__.items()})

class Pytorch(Quantization):
    def __init__(self, opt, technique, framework, model_name):
        Quantization.__init__(self, technique, framework, model_name)        
        self.cfg = opt.cfg
        self.data = opt.data
        self.hyp = opt.hyp


'''Pytorch Quantize Aware training'''
class QAT(Pytorch):
    def __init__(self, opt, technique, framework, model_name):
        Pytorch.__init__(self, opt, technique, framework, model_name)
        self.device = opt.device
        self.batch_size_QAT = opt.batch_size_QAT
        self.QAT_epochs = opt.QAT_epochs
        self.weights = opt.weights  
        self.img_size = opt.img_size
        self.cache_images = opt.cache_images
        self.single_cls = opt.single_cls
        self.adam = opt.adam
        self.workers = opt.workers

    def set_project_and_name(self, train_results_paths, model_names):
        self.project = train_results_paths[self.technique][self.framework][self.model_name]
        self.name = model_names[self.technique][self.framework][self.model_name]


    def quantize(self, train_results_paths, model_names):
        self.set_project_and_name(train_results_paths, model_names)
        Quantization.model_args(self)
        train.run(**self.__dict__)

'''Pytorch Static Post Training Quantization(PTQ)'''    
class PTQ(Pytorch):
    def __init__(self, opt, technique, framework, model_name):
        Pytorch.__init__(self, opt, technique, framework, model_name)
        self.device = 'cpu'
        self.weights = opt.trained_weights
        self.results = opt.ptq_model_store
    def quantize(self):
        Quantization.model_args(self)
        PT_quant.run(**self.__dict__)

class Tflite(Quantization):
    def __init__(self, opt, technique, framework, model_name):
        Quantization.__init__(self, technique, framework, model_name)
        self.framework_path = opt.framework_path
        self.cfg = opt.cfg,
        self.data = opt.data,
        self.batch_size = opt.batch_size
    # def set_name_only(self):
        # self.name = self.model_names[self.technique][self.framework][self.model_name]

    def quantize(self, model_names):
        self.model_names = model_names
        self.model_type_for_export = self.model_names[self.technique][self.framework][self.model_name]
        # Tflite.set_name_only(self)
        Quantization.model_args(self)
        model_export.run(**self.__dict__)


'''Tflite fp32->fp16 PTQ'''
class TFL_fp16(Tflite):
    def __init__(self, opt, technique, framework, model_name):
        Tflite.__init__(self, opt, technique, framework, model_name)


'''Tflite fp32->int8 PTQ'''
class TFL_int8(Tflite):
    def __init__(self, opt, technique, framework, model_name):
        Tflite.__init__(self, opt, technique, framework, model_name)
        self.repr_images = opt.repr_images
        self.imgtf = opt.imgtf
        self.ncalib = opt.ncalib


