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
    def __init__(self, opt):
        super().__init__(opt)
        self.cfg = opt.cfg
        self.data = opt.data
        self.hyp = opt.hyp
        self.device = opt.device
        self.cache_images = opt.cache_images
    def train_n_quant(self, opt):
        self.adam = opt.adam
        self.workers = opt.workers
    def train_prun_n_quant(self, opt, paths):
        self.weights = paths.running_model_paths['Regular']['Pytorch']['fp32'] if opt.retrain_on_pre_trained else opt.weights
    def prun_quant_infer(self, paths):
        self.running_model_paths = paths.running_model_paths
    def quant_infer(self, opt):
        self.batch_size_inferquant = opt.batch_size_inferquant
    def train_prun_infer(self, opt):
        self.batch_size = opt.batch_size
    def train_quant_infer(self, opt):
        self.single_cls = opt.single_cls

    def run(self, opt, paths):
        self.epochs = opt.epochs
        self.train_prun_n_quant(opt, paths)
        self.project = paths.train_results_paths['Regular']['Pytorch']['fp32']
        self.name = paths.model_names['Regular']['Pytorch']['fp32']
        self.train_n_quant(opt)
        self.train_prun_infer(opt)
        self.train_quant_infer(opt)
        pipeline.model_args(self)
        train.run(**self.__dict__)
    

class model_type_conversion(regular_train):
    def __init__(self, opt, paths):
        super().__init__(opt)
        self.clone_updated_yolov5 = opt.clone_updated_yolov5
        self.single_cls = opt.single_cls
        self.img_size = opt.img_size
        self.running_model_paths = paths.running_model_paths
    def run(self):
        pipeline.model_args(self)
        fp_type_conversion.run(**self.__dict__)



class model_exportation(regular_train):
    def __init__(self, opt, paths):
        super().__init__(opt)
        self.model_type_for_export = paths.model_names['Regular']['Pytorch']['fp32']
        self.framework_path = paths.framework_path
        self.model_names = paths.model_names
    def run(self):
        pipeline.model_args(self)
        model_export.run(**self.__dict__)

class Pruning_(regular_train):
    def __init__(self, opt, paths):
        regular_train.__init__(self, opt)
        self.skip_pruning = opt.skip_pruning,
        self.skip_P1_training = opt.skip_P1_training
        self.skip_P2_training= opt.skip_P2_training
        self.skip_P4_training = opt.skip_P4_training
        self.pre_trained_model_paths = paths.pre_trained_model_paths
        self.prune_retrain_epochs = opt.prune_retrain_epochs
        self.num_iterations = opt.prune_iterations
        self.prune_perc = opt.prune_perc
        self.theta0_weights = paths.pre_trained_model_paths['Pruning']['Pytorch']['theta0']
        self.P1_saved = paths.running_model_paths['Pruning']['Pytorch']['P1']
        self.P2_saved = paths.running_model_paths['Pruning']['Pytorch']['P2']
        self.P4_epochs = opt.P4_epochs
        self.sparsity_training = opt.sparsity_training
        self.sparsity_rate = opt.sparsity_rate
        
    def run(self, opt, paths):
        regular_train.train_prun_n_quant(self, opt, paths)
        regular_train.prun_quant_infer(self, paths)
        ''' running_model_paths_modification for pruned model with pre-trained/pruned stored weights'''
        if opt.prune_infer_on_pre_pruned_only == True:
            self.running_model_paths = prune_with_pre_trained_only(self.running_model_paths, self.pre_trained_model_paths)
        regular_train.train_prun_infer(self, opt)
        pipeline.model_args(self)
        pruning.run(**self.__dict__)


class Quantization_(regular_train):
    def __init__(self, opt, paths):
        regular_train.__init__(self, opt)
        self.skip_QAT_training = opt.skip_QAT_training
        self.batch_size_QAT = opt.batch_size_QAT
        self.QAT_epochs = opt.QAT_epochs
        self.framework_path = paths.framework_path
        self.trained_weights = paths.running_model_paths['Regular']['Pytorch']['fp32']
        self.ptq_model_store = paths.running_model_paths['Quantization']['Pytorch']['PTQ']
        self.repr_images = opt.repr_images
        self.imgtf = opt.imgtf
        self.ncalib = opt.ncalib

    def run(self, opt, paths):
        regular_train.train_n_quant(self, opt)
        regular_train.train_prun_n_quant(self, opt, paths)
        regular_train.prun_quant_infer(self, paths)
        regular_train.quant_infer(self, opt)
        regular_train.train_quant_infer(self, opt)
        pipeline.model_args(self)
        quantization.run(**self.__dict__)

class inferencing(Quantization_, Pruning_):
    def __init__(self, opt, paths):
        super().__init__(opt, paths)
        self.save_txt = opt.save_txt
        self.conf_thres = opt.conf_thres
        self.iou_thres = opt.iou_thres
    def run(self, opt, paths):
        regular_train.prun_quant_infer(self, paths)
        regular_train.quant_infer(self, opt)
        regular_train.train_prun_infer(self, opt)
        regular_train.train_quant_infer(self, opt)
        pipeline.model_args(self)
        return inference_results.run(**self.__dict__)

