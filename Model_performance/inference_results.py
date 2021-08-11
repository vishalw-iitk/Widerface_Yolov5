import os
# from dts.Model_performance.
from dts.Model_performance.Inference_results_store.Quantization.Tflite import infer_tfl
from dts.Model_performance.Inference_results_store.Quantization.Pytorch import infer_pyt
from dts.model_paths import infer_results_dictionary
from dts.model_paths import model_defined_names

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

    def metrics(self, model_path):
        model = load_model(model_path)
        results = val.run(model)
        self.mAP05, self.mAP0595, self.fitness
        self.size = check_size(model_path)
        self.latency = latency.run(model)
        self.gflops = gflops.run(model)
        return {'mAP0.5' : self.mAP05, 'mAP0595' : self.mAP0595, 'fitness' : self.fitness, 'size' : self.size, 'latency' : self.latency, 'gflops' : self.gflops}

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
    def metrics(self):
        pass


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
    def metrics(self):
        pass
        
class P2_PyP(PruningP):
    def __init__(self):
        PruningP.__init__(self)
    def metrics(self):
        pass



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



def run(opt, running_model_paths):
    running_model_metrics = running_model_paths
    infer_paths = infer_results_dictionary()
    model_name = model_defined_names()

    # Regular model Pytroch val results
    # infer_paths['Regular']['Pytorch']['fp32']
    # model_name['Regular']['Pytorch']['fp32']
    # os.makedirs(infer_paths['Regular']['Pytorch']['fp32']) if not os.path.exists(infer_paths['Regular']['Pytorch']['fp32']) else None
    # regularp = PytorchR()
    # running_model_metrics['Regular']['Pytorch']['fp32'] = regularp.metrics()

    # Regular model Tflite val results
    os.makedirs(infer_paths['Regular']['Tflite']['fp32']) if not os.path.exists(infer_paths['Regular']['Tflite']['fp32']) else None       
    regulartf = Tfl_fp32_R()
    running_model_metrics['Regular']['Tflite']['fp32'] = regulartf.metrics(
        data = 'data.yaml',
        weights = running_model_paths['Regular']['Tflite']['fp32'],
        batch_size = 1,
        imgsz = 416,
        conf_thres = 0.001,
        iou_thres = 0.6,
        task = 'val',
        device = 'cpu',
        single_cls = True,
        augment = False,
        verbose = True,
        save_txt = True,
        save_hybrid = False,
        save_conf = False,
        save_json = False,
        project = infer_paths['Regular']['Tflite']['fp32'],
        name = model_name['Regular']['Tflite']['fp32'],
        exist_ok = True,
        half = False,
        tfl_int8 = False
        )






    '''*********************  Quantization  ***************'''
    # Pytorch
    # 1) QAT
    # infer_paths['Quantization']['Pytorch']['QAT']
    # model_name['Quantization']['Pytorch']['QAT']
    os.makedirs(infer_paths['Quantization']['Pytorch']['QAT']) if not os.path.exists(infer_paths['Quantization']['Pytorch']['QAT']) else None
    qat_py = QAT_PyQ()
    running_model_metrics['Quantization']['Pytorch']['QAT'] = qat_py.metrics(
        weights = running_model_paths['Quantization']['Pytorch']['QAT'],
        cfg = opt.cfg,
        device = opt.device,
        img_size = opt.img_size,
        data = opt.data,
        hyp = opt.hyp
    )
    print("the required....")
    print(running_model_metrics['Quantization']['Pytorch']['QAT'])

    
    # 2) PTQ
    # infer_paths['Quantization']['Pytorch']['PTQ']
    # model_name['Quantization']['Pytorch']['PTQ']
    # ptq_py = PTQ_PyQ()
    # running_model_metrics['Quantization']['Pytorch']['PTQ'] = ptq_py.metrics()


    # Tflite
    # 1) fp16
    # infer_paths['Quantization']['Tflite']['fp16']
    # model_name['Quantization']['Tflite']['fp16']
    os.makedirs(infer_paths['Quantization']['Tflite']['fp16']) if not os.path.exists(infer_paths['Quantization']['Tflite']['fp16']) else None
    fp16_Q_tf = Tfl_fp16_Q()
    running_model_metrics['Quantization']['Tflite']['fp16'] = fp16_Q_tf.metrics(
        data = 'data.yaml',
        weights = running_model_paths['Quantization']['Tflite']['fp16'],
        batch_size = 1,
        imgsz = 416,
        conf_thres = 0.001,
        iou_thres = 0.6,
        task = 'val',
        device = 'cpu',
        single_cls = True,
        augment = False,
        verbose = True,
        save_txt = True,
        save_hybrid = False,
        save_conf = False,
        save_json = False,
        project = infer_paths['Quantization']['Tflite']['fp16'],
        name = model_name['Quantization']['Tflite']['fp16'],
        exist_ok = True,
        half = False,
        tfl_int8 = False
        )

    # int8
    # infer_paths['Quantization']['Tflite']['int8']
    # model_name['Quantization']['Tflite']['int8']
    int8_Q_tf = Tfl_int8_Q()
    running_model_metrics['Quantization']['Tflite']['int8'] = int8_Q_tf.metrics(
        data = 'data.yaml',
        weights = running_model_paths['Quantization']['Tflite']['int8'],
        # weights = running_model_paths['Regular']['Tflite']['fp32'],
        batch_size = 1,
        imgsz = 416,
        conf_thres = 0.001,
        iou_thres = 0.6,
        task = 'val',
        device = 'cpu',
        single_cls = True,
        augment = False,
        verbose = True,
        save_txt = True,
        save_hybrid = False,
        save_conf = False,
        save_json = False,
        project = infer_paths['Quantization']['Tflite']['int8'],
        name = model_name['Quantization']['Tflite']['int8'],
        exist_ok = True,
        half = False,
        tfl_int8 = True
        )
    



    '''*********************  Pruning  ***************'''
    # Pytorch
    # method 1
    # infer_paths['Pruning']['Pytorch']['P1']
    # model_name['Pruning']['Pytorch']['P1']
    os.makedirs(infer_paths['Pruning']['Pytorch']['P1']) if not os.path.exists(infer_paths['Pruning']['Pytorch']['P1']) else None
    pr1_py = P1_PyP()
    running_model_metrics['Pruning']['Pytorch']['P1'] = pr1_py.metrics()

    # method 2
    # infer_paths['Pruning']['Pytorch']['P2']
    # model_name['Pruning']['Pytorch']['P2']
    pr2_py = P2_PyP()
    running_model_metrics['Pruning']['Pytorch']['P2'] = pr2_py.metrics()

    # Tflite
    # method 1
    # infer_paths['Pruning']['Tflite']['P1']
    # model_name['Pruning']['Tflite']['P1']
    os.makedirs(infer_paths['Pruning']['Tflite']['P1']) if not os.path.exists(infer_paths['Pruning']['Tflite']['P1']) else None
    tflm1p = tflm1P()
    running_model_metrics['Pruning']['Tflite']['P1'] = tflm1p.metrics()

    # method 2
    # infer_paths['Pruning']['Tflite']['P2']
    # model_name['Pruning']['Tflite']['P2']
    tflm2p = tflm2P()
    running_model_metrics['Pruning']['Tflite']['P2'] = tflm2p.metrics()

    del running_model_metrics['Regular']['Pytorch']['fp16']
    return running_model_metrics
