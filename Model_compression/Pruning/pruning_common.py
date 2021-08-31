from dts.Model_compression.Pruning.Pytorch import prune_train
from dts.Model_compression.Pruning.Pytorch.P4 import train

class Pruning__:
    def __init__(self, opt, technique, framework, model_name, train_results_paths, model_names):
        self.technique, self.framework, self.model_name = technique, framework, model_name
        self.train_results_paths = train_results_paths
        self.model_names = model_names
        self.data=opt.data
        self.cfg=opt.cfg 
        self.hyp=opt.hyp
        self.device=opt.device
        self.batch_size=opt.batch_size
        self.imgsz=opt.img_size
        self.cache_images = opt.cache_images
        self.exist_ok= opt.exist_ok
        self.prune_on_weights = opt.weights
    def model_args(self):
        self.__dict__.update({k:v if type(v) is not tuple else v[0] for k,v in self.__dict__.items()})
    def set_project_and_name(self):
        self.project = self.train_results_paths[self.technique][self.framework][self.model_name]
        self.name = self.model_names[self.technique][self.framework][self.model_name]



class Unstructured(Pruning__):
    def __init__(self, opt, technique, framework, model_name, train_results_paths, model_names):
        super().__init__(opt, technique, framework, model_name, train_results_paths, model_names)
        self.epochs=opt.prune_retrain_epochs
        self.prune_perc=opt.prune_perc
        self.random_reinit = False
        self.theta0_reinit = True
        self.num_iterations = opt.num_iterations
    def set_weights(self, i):
        self.weights=self.prune_on_weights if i==0 else self.P_saved
    def give_iter_prints(self, i):
        print("++++++    Starting iteration " + str(i) + " of "+ self.model_name + " iterative pruning    +++++++++")
    def prune(self, i):
        self.prune_iter=i
        self.set_weights(i)
        self.give_iter_prints(i)
        Pruning__.set_project_and_name(self)
    def iterator(self):
        for i in range(self.num_iterations):
            self.prune(i)
            Pruning__.model_args(self)
            prune_train.run(**self.__dict__)





# Pytorch
class P1(Unstructured):
    def __init__(self, opt, technique, framework, model_name, train_results_paths, model_names):
        super().__init__(opt, technique, framework, model_name, train_results_paths, model_names)
        self.P_saved = opt.P1_saved
        

class P2(Unstructured):
    def __init__(self, opt, technique, framework, model_name, train_results_paths, model_names):
        super().__init__(opt, technique, framework, model_name, train_results_paths, model_names)
        self.random_reinit = not self.random_reinit
        self.theta0_reinit = not self.theta0_reinit
        self.theta0_weights = opt.theta0_weights
        self.P_saved = opt.P2_saved




class Structured(Pruning__):
    def __init__(self, opt, technique, framework, model_name, train_results_paths, model_names):
        super().__init__(opt, technique, framework, model_name, train_results_paths, model_names)
        self.weights = self.prune_on_weights
        self.epochs = opt.P4_epochs
        self.st = opt.st
        self.sr = opt.sr

class P4(Structured):
    def __init__(self, opt, technique, framework, model_name, train_results_paths, model_names):
        super().__init__(opt, technique, framework, model_name, train_results_paths, model_names)
        
    def prune(self):
        Pruning__.set_project_and_name(self)
        Pruning__.model_args(self)
        train.run(**self.__dict__)

