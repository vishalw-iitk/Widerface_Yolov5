import os
from pathlib import Path

from dts.Model_performance.Inference_results_store.Regular.Pytorch import infer_regular
from dts.Model_performance.Inference_results_store.Quantization.Tflite import infer_tfl
from dts.Model_performance.Inference_results_store.Quantization.Pytorch import infer_pyt

# Inference for all
class model_performance_Results:
    def __init__(self, opt, technique, framework, model_name):
        self.technique = technique
        self.framework = framework
        self.model_name = model_name
        self.data = opt.data
        self.img_size = opt.img_size
        self.single_cls = opt.single_cls
        self.save_txt = opt.save_txt
        self.conf_thres = opt.conf_thres
        self.iou_thres = opt.iou_thres
        self.running_model_paths = opt.running_model_paths

    def set_project_and_name(self, infer_paths, model_names):
        self.project = infer_paths[self.technique][self.framework][self.model_name]
        self.name = model_names[self.technique][self.framework][self.model_name]

    def path_to_save_dir(self):
        self.save_dir = os.path.join(self.project, self.name)

    def creation_of_save_dir(self):
        os.makedirs(self.save_dir) if not os.path.exists(self.save_dir) else None
    
    def creation_of_labels_dir(self):
        (Path(self.save_dir) / 'labels' if self.save_txt else Path(self.save_dir)).mkdir(parents=True, exist_ok=True)
    
    def infer_dir_complete(self, infer_paths, model_names):
        self.set_project_and_name(infer_paths, model_names)
        self.path_to_save_dir()
        self.creation_of_save_dir()
        self.creation_of_labels_dir()
    
    def set_weights(self):
        self.weights = self.running_model_paths[self.technique][self.framework][self.model_name]

    def model_args(self):
        self.__dict__.update({k:v if type(v) is not tuple else v[0] for k,v in self.__dict__.items()})
    
    def model_args_fullset(self):
        self.set_weights()
        self.save_dir = Path(self.save_dir)
        self.model_args()
    
    def metrics(self):
        self.model_args_fullset()
        return infer_regular.run(**self.__dict__)

    def get_results_dictionary(self, running_model_metrics):
        running_model_metrics[self.technique][self.framework][self.model_name] = self.metrics()

    def result_prints(self, running_model_metrics):
        print("\nThe "+ self.technique + " " +self.framework + " " +self.model_name + " results are as follows :")
        print(running_model_metrics[self.technique][self.framework][self.model_name],'\n')
    
    def explicit_inference(self, infer_paths, model_names, running_model_metrics):
        self.infer_dir_complete(infer_paths, model_names)
        self.get_results_dictionary(running_model_metrics)
        self.result_prints(running_model_metrics)
        
    @staticmethod
    def unused_plot_keys(running_model_metrics):
        del running_model_metrics['Regular']['Pytorch']['fp16']
        del running_model_metrics['Quantization']['Tflite']['int8']
        del running_model_metrics['Pruning']['Tflite']


# Regular
class Regular(model_performance_Results):
    def __init__(self, opt, technique, framework, model_name):
        model_performance_Results.__init__(self, opt, technique, framework, model_name)


# Quantization
class Quantization(model_performance_Results):
    def __init__(self, opt, technique, framework, model_name):
        model_performance_Results.__init__(self, opt, technique, framework, model_name)

# Pruning
class Pruning(model_performance_Results):
    def __init__(self, opt, technique, framework, model_name):
        model_performance_Results.__init__(self, opt, technique, framework, model_name)
        self.device = opt.device
        self.cfg = opt.cfg
        self.hyp = opt.hyp
        self.batch_size = opt.batch_size



# Pytorch Regular
class PytorchR(Regular):
    def __init__(self, opt, technique, framework, model_name):
        Regular.__init__(self, opt, technique, framework, model_name)
        self.device = opt.device
        self.cfg = opt.cfg
        self.hyp = opt.hyp
        self.batch_size = opt.batch_size

# Tflite Regular
class Tfl_fp32_R(Regular):
    def __init__(self, opt, technique, framework, model_name):
        Regular.__init__(self, opt, technique, framework, model_name)
        self.batch_size = 1
        self.verbose = True
        self.exist_ok = True
        self.tfl_int8 = False

    def creation_of_save_dir(self):
        os.makedirs(self.project) if not os.path.exists(self.project) else None

    def model_args_fullset(self):
        model_performance_Results.set_weights(self)
        model_performance_Results.model_args(self)

    def metrics(self):
        self.model_args_fullset()
        return infer_tfl.run(**self.__dict__)

    def explicit_inference(self, infer_paths, model_names, running_model_metrics):
        model_performance_Results.set_project_and_name(self, infer_paths, model_names)
        self.creation_of_save_dir() # project = save_dir in this case
        model_performance_Results.get_results_dictionary(self, running_model_metrics)
        model_performance_Results.result_prints(self, running_model_metrics)



# Pytorch Quantization
class PytorchQ(Quantization):
    def __init__(self, opt, technique, framework, model_name):
        Quantization.__init__(self, opt, technique, framework, model_name)
        self.cfg = opt.cfg
        self.hyp = opt.hyp
        self.device = 'cpu'
        self.batch_size_inferquant = opt.batch_size_inferquant
    
    def set_fuse(self):
        self.fuseQ = True if self.model_name == 'QAT' else False

    def metrics(self):
        model_performance_Results.model_args_fullset(self)
        return infer_pyt.run(**self.__dict__)

    def get_results_dictionary(self, running_model_metrics):
        self.set_fuse()
        running_model_metrics[self.technique][self.framework][self.model_name] = self.metrics()



# Tflite Quantization
class TfliteQ(Quantization, Tfl_fp32_R):
    def __init__(self, opt, technique, framework, model_name):
        Quantization.__init__(self, opt, technique, framework, model_name)
        Tfl_fp32_R.__init__(self, opt, technique, framework, model_name)
        self.tfl_int8 = False
