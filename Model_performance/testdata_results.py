import os
from dts.model_paths import model_defined_names
from dts.model_paths import test_results_dictionary


class Test_data_results:
    def __init__(self):
        pass

class Regular(Test_data_results):
    def __init__(self):
        Test_data_results.__init__(self)

class Quantization(Test_data_results):
    def __init__(self):
        Test_data_results.__init__(self)

class Pruning(Test_data_results):
    def __init__(self):
        Test_data_results.__init__(self)


# Regular
class PytorchR(Regular):
    def __init__(self):
        Regular.__init__(self)

    def get_testdata_results(self, model_path):
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

    def get_testdata_results(self, **kwargs):
        results_dictionary = infer.run(**kwargs)
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
    def get_testdata_results(self):
        pass
        
class PTQ_PyQ(PytorchQ):
    def __init__(self):
        PytorchQ.__init__(self)
    def get_testdata_results(self):
        pass


# Tflite Quantization
class Tfl_fp16_Q(Quantization):
    def __init__(self):
        Quantization.__init__(self)

    def get_testdata_results(self, **kwargs):
        results_dictionary = infer.run(**kwargs)
        return results_dictionary

class Tfl_int8_Q(Quantization):
    def __init__(self):
        Quantization.__init__(self)

    def get_testdata_results(self, **kwargs):
        results_dictionary = infer.run(**kwargs)
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
    def get_testdata_results(self):
        pass
        
class P2_PyP(PruningP):
    def __init__(self):
        PruningP.__init__(self)
    def get_testdata_results(self):
        pass



# Tflite pruning
class tflm1P(TfliteP):
    def __init__(self):
        TfliteP.__init__(self)
    def get_testdata_results(self):
        pass
        
class tflm2P(TfliteP):
    def __init__(self):
        TfliteP.__init__(self)
    def get_testdata_results(self):
        pass    

        

def run(running_model_paths):
    # regular model test results
    test_results_paths = test_results_dictionary()
    model_name = model_defined_names()

    # Regular model Pytroch val results
    # test_results_paths['Regular']['Pytorch']['fp32']
    # model_name['Regular']['Pytorch']['fp32']
    os.makedirs(test_results_paths['Regular']['Pytorch']['fp32']) if not os.path.exists(test_results_paths['Regular']['Pytorch']['fp32']) else None
    regularp = PytorchR()
    regularp.get_testdata_results()

    # Regular model Tflite val results
    os.makedirs(test_results_paths['Regular']['Tflite']['fp32']) if not os.path.exists(test_results_paths['Regular']['Tflite']['fp32']) else None       
    regulartf = Tfl_fp32_R()
    regulartf.get_testdata_results(
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
        project = test_results_paths['Regular']['Tflite']['fp32'],
        name = model_name['Regular']['Tflite']['fp32'],
        exist_ok = True,
        half = False,
        tfl_int8 = False
        )






    '''*********************  Quantization  ***************'''
    # Pytorch
    # 1) QAT
    # test_results_paths['Quantization']['Pytorch']['QAT']
    # model_name['Quantization']['Pytorch']['QAT']
    os.makedirs(test_results_paths['Quantization']['Pytorch']['QAT']) if not os.path.exists(test_results_paths['Quantization']['Pytorch']['QAT']) else None
    qat_py = QAT_PyQ()
    qat_py.get_testdata_results()

    # 2) PTQ
    # test_results_paths['Quantization']['Pytorch']['PTQ']
    # model_name['Quantization']['Pytorch']['PTQ']
    ptq_py = PTQ_PyQ()
    ptq_py.get_testdata_results()


    # Tflite
    # 1) fp16
    # test_results_paths['Quantization']['Tflite']['fp16']
    # model_name['Quantization']['Tflite']['fp16']
    os.makedirs(test_results_paths['Quantization']['Tflite']['fp16']) if not os.path.exists(test_results_paths['Quantization']['Tflite']['fp16']) else None
    fp16_Q_tf = Tfl_fp16_Q()
    fp16_Q_tf.get_testdata_results(
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
        project = test_results_paths['Quantization']['Tflite']['fp16'],
        name = model_name['Quantization']['Tflite']['fp16'],
        exist_ok = True,
        half = False,
        tfl_int8 = False
        )

    # int8
    # test_results_paths['Quantization']['Tflite']['int8']
    # model_name['Quantization']['Tflite']['int8']
    int8_Q_tf = Tfl_int8_Q()
    int8_Q_tf.get_testdata_results(
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
        project = test_results_paths['Quantization']['Tflite']['int8'],
        name = model_name['Quantization']['Tflite']['int8'],
        exist_ok = True,
        half = False,
        tfl_int8 = True
        )
    



    '''*********************  Pruning  ***************'''
    # Pytorch
    # method 1
    # test_results_paths['Pruning']['Pytorch']['P1']
    # model_name['Pruning']['Pytorch']['P1']
    os.makedirs(test_results_paths['Pruning']['Pytorch']['P1']) if not os.path.exists(test_results_paths['Pruning']['Pytorch']['P1']) else None
    pr1_py = P1_PyP()
    pr1_py.get_testdata_results()

    # method 2
    # test_results_paths['Pruning']['Pytorch']['P2']
    # model_name['Pruning']['Pytorch']['P2']
    pr2_py = P2_PyP()
    pr2_py.get_testdata_results()

    # Tflite
    # method 1
    # test_results_paths['Pruning']['Tflite']['P1']
    # model_name['Pruning']['Tflite']['P1']
    os.makedirs(test_results_paths['Pruning']['Tflite']['P1']) if not os.path.exists(test_results_paths['Pruning']['Tflite']['P1']) else None
    tflm1p = tflm1P()
    tflm1p.get_testdata_results()

    # method 2
    # test_results_paths['Pruning']['Tflite']['P2']
    # model_name['Pruning']['Tflite']['P2']
    tflm2p = tflm2P()
    tflm2p.get_testdata_results()
