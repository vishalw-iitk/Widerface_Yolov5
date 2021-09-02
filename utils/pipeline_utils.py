from dts.utils import begin
from dts.Data_preparation import data_prep_yolo
from dts.model_paths \
    import \
        running_model_dictionary, train_results_dictionary, pre_trained_model_dictionary,\
        train_results_dictionary, model_defined_names, prune_with_pre_trained_only, frameworks, \
        update_to_running_paths_with_pretrianed
from yolov5 import train
from dts.Model_conversion import fp_type_conversion
from dts.Model_conversion import model_export
from dts.Model_conversion import model_export
from dts.Model_compression.Quantization import quantization
from dts.Model_compression.Pruning import pruning
from dts.Model_performance import inference_results

class pipeline:
    def __init__(self, opt):
        self.img_size = opt.img_size
    def model_args(self):
            self.__dict__.update({k:v if type(v) is not tuple else v[0] for k,v in self.__dict__.items()})


class ultralytics(pipeline):
    def __init__(self, opt):
        self.yolov5_repo_name = opt.yolov5_repo_name
        self.results_folder_path = opt.results_folder_path
        self.clone_updated_yolov5 = opt.clone_updated_yolov5
    def run(self):
        pipeline.model_args(self)
        begin.run(**self.__dict__)

class get_paths:
    def __init__(self, opt):
        self.running_model_paths = running_model_dictionary()
        self.pre_trained_model_paths = pre_trained_model_dictionary()
        self.framework_path = frameworks(opt.skip_training, self.running_model_paths, self.pre_trained_model_paths)
        self.train_results_paths = train_results_dictionary()
        self.model_names = model_defined_names()

        '''The paths updated here will be taken for inference results'''
        self.running_model_paths = update_to_running_paths_with_pretrianed(self.running_model_paths, self.pre_trained_model_paths, \
                                    opt.skip_training, opt.skip_QAT_training, \
                                    opt.skip_pruning, opt.skip_P1_training, opt.skip_P2_training, opt.skip_P4_training)




class data_preparation(pipeline):
    def __init__(self, opt):
        super().__init__(opt)
        self.raw_dataset_path = opt.raw_dataset_path
        self.arranged_data_path = opt.arranged_data_path
        self.partial_dataset = opt.partial_dataset
        self.percent_traindata = opt.percent_traindata
        self.percent_validationdata = opt.percent_validationdata
        self.percent_testdata = opt.percent_testdata
    def run(self):
        pipeline.model_args(self)
        data_prep_yolo.run(**self.__dict__)



class regular_train(pipeline):
    def __init__(self, opt, paths):
        super().__init__(opt)
        self.opt = opt
        self.paths = paths
        self.cfg = self.opt.cfg
        self.data = self.opt.data
        self.hyp = self.opt.hyp
        self.device = self.opt.device
        self.cache_images = self.opt.cache_images
    def train_n_quant(self):
        self.adam = self.opt.adam
        self.workers = self.opt.workers
    def prun_n_quant(self):
        self.weights = self.paths.running_model_paths['Regular']['Pytorch']['fp32'] if self.opt.retrain_on_pre_trained else self.opt.weights
    def prun_quant_infer(self):
        self.running_model_paths = self.paths.running_model_paths
    def quant_infer(self):
        self.batch_size_inferquant = self.opt.batch_size_inferquant
    def train_prun_infer(self):
        self.batch_size = self.opt.batch_size
    def train_quant_infer(self):
        self.single_cls = self.opt.single_cls

    def run(self):
        self.epochs = self.opt.epochs
        self.prun_n_quant()
        self.project = self.paths.train_results_paths['Regular']['Pytorch']['fp32']
        self.name = self.paths.model_names['Regular']['Pytorch']['fp32']
        self.train_n_quant()
        self.train_prun_infer()
        self.train_quant_infer()
        self.weights = self.paths.pre_trained_model_paths['Regular']['Pytorch']['fp32'] if self.opt.retrain_on_pre_trained else self.opt.weights
        for attr in ('opt','paths'):
            self.__dict__.pop(attr,None)
        pipeline.model_args(self)
        train.run(**self.__dict__)
    

class model_type_conversion(pipeline):
    def __init__(self, opt, paths):
        super().__init__(opt)
        self.conversion_type = 'fp16_to_fp32' if opt.clone_updated_yolov5 == True else 'fp32_to_fp16'
        self.weights = paths.running_model_paths['Regular']['Pytorch']['fp16'] if self.conversion_type == 'fp16_to_fp32'\
                else paths.running_model_paths['Regular']['Pytorch']['fp32']
    def run(self):
        pipeline.model_args(self)
        fp_type_conversion.run(**self.__dict__)



class model_exportation(regular_train):
    def __init__(self, opt, paths):
        super().__init__(opt, paths)
        self.model_type_for_export = paths.model_names['Regular']['Pytorch']['fp32']
        self.framework_path = paths.framework_path
        self.model_names = paths.model_names
    def run(self):
        pipeline.model_args(self)
        model_export.run(**self.__dict__)

class Pruning_(regular_train):
    def __init__(self, opt, paths):
        regular_train.__init__(self, opt, paths)
        self.skip_pruning = self.opt.skip_pruning,
        self.skip_P1_training = self.opt.skip_P1_training
        self.skip_P2_training= self.opt.skip_P2_training
        self.skip_P4_training = self.opt.skip_P4_training
        self.pre_trained_model_paths = self.paths.pre_trained_model_paths
        self.prune_retrain_epochs = self.opt.prune_retrain_epochs
        self.num_iterations = self.opt.prune_iterations
        self.prune_perc = self.opt.prune_perc
        self.theta0_weights = self.paths.pre_trained_model_paths['Pruning']['Pytorch']['theta0']
        self.P1_saved = self.paths.running_model_paths['Pruning']['Pytorch']['P1']
        self.P2_saved = self.paths.running_model_paths['Pruning']['Pytorch']['P2']
        self.P4_epochs = self.opt.P4_epochs
        self.sparsity_training = self.opt.sparsity_training
        self.sparsity_rate = self.opt.sparsity_rate
        
    def run(self):
        regular_train.prun_n_quant(self)
        regular_train.prun_quant_infer(self)
        ''' running_model_paths_modification for pruned model with pre-trained/pruned stored weights'''
        if self.opt.prune_infer_on_pre_pruned_only == True:
            self.running_model_paths = prune_with_pre_trained_only(self.running_model_paths, self.pre_trained_model_paths)
        regular_train.train_prun_infer(self)
        for attr in ('opt','paths'):
            self.__dict__.pop(attr,None)
        pipeline.model_args(self)
        pruning.run(**self.__dict__)


class Quantization_(regular_train):
    def __init__(self, opt, paths):
        regular_train.__init__(self, opt, paths)
        self.skip_QAT_training = self.opt.skip_QAT_training
        self.batch_size_QAT = self.opt.batch_size_QAT
        self.QAT_epochs = self.opt.QAT_epochs
        self.framework_path = self.paths.framework_path
        self.trained_weights = self.paths.running_model_paths['Regular']['Pytorch']['fp32']
        self.ptq_model_store = self.paths.running_model_paths['Quantization']['Pytorch']['PTQ']
        self.repr_images = self.opt.repr_images
        self.imgtf = self.opt.imgtf
        self.ncalib = self.opt.ncalib

    def run(self):
        regular_train.train_n_quant(self)
        regular_train.prun_n_quant(self)
        regular_train.prun_quant_infer(self)
        regular_train.quant_infer(self)
        regular_train.train_quant_infer(self)
        for attr in ('opt','paths'):
            self.__dict__.pop(attr,None)
        pipeline.model_args(self)
        quantization.run(**self.__dict__)

class inferencing(Quantization_, Pruning_):
    def __init__(self, opt, paths):
        super().__init__(opt, paths)
        self.save_txt = self.opt.save_txt
        self.conf_thres = self.opt.conf_thres
        self.iou_thres = self.opt.iou_thres
    def run(self):
        regular_train.prun_quant_infer(self)
        regular_train.quant_infer(self)
        regular_train.train_prun_infer(self)
        regular_train.train_quant_infer(self)
        for attr in ('opt','paths'):
            self.__dict__.pop(attr,None)
        pipeline.model_args(self)
        return inference_results.run(**self.__dict__)

