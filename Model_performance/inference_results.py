import os
from dts.Model_performance.Inference_results_store.Regular.Pytorch import infer_regular
from dts.Model_performance.Inference_results_store.Quantization.Tflite import infer_tfl
from dts.Model_performance.Inference_results_store.Quantization.Pytorch import infer_pyt
# from dts.Model_performance.Inference_results_store.Quantization.Pytorch.PTQ import PTQ_infer
from dts.model_paths import infer_results_dictionary
from dts.model_paths import model_defined_names
from pathlib import Path

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

    def metrics(self, **kwargs):
        results_dictionary = infer_regular.run(**kwargs)
        return results_dictionary

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
    def metrics(self, **kwargs):
        # results_dictionary = PTQ_infer.run(**kwargs)
        results_dictionary = infer_pyt.run(**kwargs)
        return results_dictionary


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
    def metrics(self, **kwargs):
        results_dictionary = infer_regular.run(**kwargs)
        return results_dictionary
        
class P2_PyP(PruningP):
    def __init__(self):
        PruningP.__init__(self)
    def metrics(self, **kwargs):
        results_dictionary = infer_regular.run(**kwargs)
        return results_dictionary

class P4_PyP(PruningP):
    def __init__(self):
        PruningP.__init__(self)
    def metrics(self, **kwargs):
        results_dictionary = infer_regular.run(**kwargs)
        return results_dictionary

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
    
    project = infer_paths['Regular']['Pytorch']['fp32'],
    name = model_name['Regular']['Pytorch']['fp32']
    save_dir = os.path.join(project[0], name)
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    (Path(save_dir) / 'labels' if opt.save_txt else Path(save_dir)).mkdir(parents=True, exist_ok=True)  # make dir
    # os.makedirs(infer_paths['Regular']['Pytorch']['fp32']) if not os.path.exists(infer_paths['Regular']['Pytorch']['fp32']) else None
    regularp = PytorchR()
    running_model_metrics['Regular']['Pytorch']['fp32'] = regularp.metrics(
        weights = running_model_paths['Regular']['Pytorch']['fp32'],
        cfg = opt.cfg,
        device = opt.device,
        img_size = opt.img_size,
        batch_size = opt.batch_size,
        data = opt.data,
        hyp = opt.hyp,
        single_cls = opt.single_cls,
        save_dir = Path(save_dir),
        save_txt = opt.save_txt
    )
    print("the required...regular pyt.")
    print(running_model_metrics['Regular']['Pytorch']['fp32'])

    # Regular model Tflite val results
    os.makedirs(infer_paths['Regular']['Tflite']['fp32']) if not os.path.exists(infer_paths['Regular']['Tflite']['fp32']) else None       
    regulartf = Tfl_fp32_R()
    running_model_metrics['Regular']['Tflite']['fp32'] = regulartf.metrics(
        data = opt.data,
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
    print("the required...tfl fp 32.")
    print(running_model_metrics['Regular']['Tflite']['fp32'])





    '''*********************  Quantization  ***************'''
    # Pytorch
    # 1) QAT
    # infer_paths['Quantization']['Pytorch']['QAT']
    # model_name['Quantization']['Pytorch']['QAT']
    
    project = infer_paths['Quantization']['Pytorch']['QAT']
    name = model_name['Quantization']['Pytorch']['QAT']
    # print("project", project, type(project), project[0])
    save_dir = os.path.join(project, name)
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    (Path(save_dir) / 'labels' if opt.save_txt else Path(save_dir)).mkdir(parents=True, exist_ok=True)  # make dir
    # os.makedirs(infer_paths['Quantization']['Pytorch']['QAT']) if not os.path.exists(infer_paths['Quantization']['Pytorch']['QAT']) else None
    qat_py = QAT_PyQ()
    running_model_metrics['Quantization']['Pytorch']['QAT'] = qat_py.metrics(
        weights = running_model_paths['Quantization']['Pytorch']['QAT'],
        cfg = opt.cfg,
        device = 'cpu',
        img_size = opt.img_size,
        batch_size_inferquant = opt.batch_size_inferquant,
        data = opt.data,
        hyp = opt.hyp,
        conf_thres = 0.001,
        iou_thres = 0.6,
        single_cls = False,
        save_dir = Path(save_dir),
        fuse = True,
        save_txt = opt.save_txt
    )
    print("the required...QAT.")
    print(running_model_metrics['Quantization']['Pytorch']['QAT'])

    
    # 2) PTQ
    # infer_paths['Quantization']['Pytorch']['PTQ']
    # model_name['Quantization']['Pytorch']['PTQ']
    
    project = infer_paths['Quantization']['Pytorch']['PTQ']
    name = model_name['Quantization']['Pytorch']['PTQ']
    # print("project", project, type(project), project[0])
    save_dir = os.path.join(project, name)
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    (Path(save_dir) / 'labels' if opt.save_txt else Path(save_dir)).mkdir(parents=True, exist_ok=True)  # make dir
    ptq_py = PTQ_PyQ()
    running_model_metrics['Quantization']['Pytorch']['PTQ'] = ptq_py.metrics(
        weights = running_model_paths['Quantization']['Pytorch']['PTQ'],
        cfg = opt.cfg,
        device = 'cpu',
        img_size = opt.img_size,
        batch_size_inferquant = opt.batch_size_inferquant,
        data = opt.data,
        hyp = opt.hyp,
        conf_thres = 0.001,
        iou_thres = 0.6,
        single_cls = False,
        save_dir = Path(save_dir),
        fuse = False,
        save_txt = opt.save_txt
    )
    print("the required....PTQ")
    print(running_model_metrics['Quantization']['Pytorch']['PTQ'])


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
    ##int8_Q_tf = Tfl_int8_Q()
    # running_model_metrics['Quantization']['Tflite']['int8'] = int8_Q_tf.metrics(
    #     data = 'data.yaml',
    #     weights = running_model_paths['Quantization']['Tflite']['int8'],
    #     # weights = running_model_paths['Regular']['Tflite']['fp32'],
    #     batch_size = 1,
    #     imgsz = 416,
    #     conf_thres = 0.001,
    #     iou_thres = 0.6,
    #     task = 'val',
    #     device = 'cpu',
    #     single_cls = True,
    #     augment = False,
    #     verbose = True,
    #     save_txt = True,
    #     save_hybrid = False,
    #     save_conf = False,
    #     save_json = False,
    #     project = infer_paths['Quantization']['Tflite']['int8'],
    #     name = model_name['Quantization']['Tflite']['int8'],
    #     exist_ok = True,
    #     half = False,
    #     tfl_int8 = True
    #     )
    



    '''*********************  Pruning  ***************'''
    # Pytorch
    # method 1

    project = infer_paths['Pruning']['Pytorch']['P1']
    name = model_name['Pruning']['Pytorch']['P1']
    # print("project", project, type(project), project[0])
    save_dir = os.path.join(project, name)
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    (Path(save_dir) / 'labels' if opt.save_txt else Path(save_dir)).mkdir(parents=True, exist_ok=True)  # make dir
    # os.makedirs(infer_paths['Pruning']['Pytorch']['P1']) if not os.path.exists(infer_paths['Pruning']['Pytorch']['P1']) else None
    p1_reinit = P1_PyP()
    running_model_metrics['Pruning']['Pytorch']['P1'] = p1_reinit.metrics(
        weights = running_model_paths['Pruning']['Pytorch']['P1'],
        cfg = opt.cfg,
        device = opt.device,
        img_size = opt.img_size,
        batch_size = opt.batch_size,
        data = opt.data,
        hyp = opt.hyp,
        single_cls = opt.single_cls,
        save_dir = Path(save_dir),
        save_txt = opt.save_txt
    )
    print("the required...regular P1")
    print(running_model_metrics['Pruning']['Pytorch']['P1'])

    # method 2
    
    project = infer_paths['Pruning']['Pytorch']['P2']
    name = model_name['Pruning']['Pytorch']['P2']
    save_dir = os.path.join(project, name)
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    (Path(save_dir) / 'labels' if opt.save_txt else Path(save_dir)).mkdir(parents=True, exist_ok=True)  # make dir
    # os.makedirs(infer_paths['Pruning']['Pytorch']['P2']) if not os.path.exists(infer_paths['Pruning']['Pytorch']['P2']) else None
    p2_theta0 = P2_PyP()
    running_model_metrics['Pruning']['Pytorch']['P2'] = p2_theta0.metrics(
        weights = running_model_paths['Pruning']['Pytorch']['P2'],
        cfg = opt.cfg,
        device = opt.device,
        img_size = opt.img_size,
        batch_size = opt.batch_size,
        data = opt.data,
        hyp = opt.hyp,
        single_cls = opt.single_cls,
        save_dir = Path(save_dir),
        save_txt = opt.save_txt
    )
    print("the required...regular P2")
    print(running_model_metrics['Pruning']['Pytorch']['P2'])


    project = infer_paths['Pruning']['Pytorch']['P4']
    name = model_name['Pruning']['Pytorch']['P4']
    save_dir = os.path.join(project, name)
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    (Path(save_dir) / 'labels' if opt.save_txt else Path(save_dir)).mkdir(parents=True, exist_ok=True)  # make dir
    # os.makedirs(infer_paths['Pruning']['Pytorch']['P4']) if not os.path.exists(infer_paths['Pruning']['Pytorch']['P4']) else None
    p4_layerprune = P4_PyP()
    running_model_metrics['Pruning']['Pytorch']['P4'] = p4_layerprune.metrics(
        weights = running_model_paths['Pruning']['Pytorch']['P4'],
        cfg = opt.cfg,
        device = opt.device,
        img_size = opt.img_size,
        batch_size = opt.batch_size,
        data = opt.data,
        hyp = opt.hyp,
        single_cls = opt.single_cls,
        save_dir = Path(save_dir),
        save_txt = opt.save_txt
    )
    print("the required...regular P4")
    print(running_model_metrics['Pruning']['Pytorch']['P4'])


    # Tflite
    # method 1
    # infer_paths['Pruning']['Tflite']['P1']
    # model_name['Pruning']['Tflite']['P1']
    ##os.makedirs(infer_paths['Pruning']['Tflite']['P1']) if not os.path.exists(infer_paths['Pruning']['Tflite']['P1']) else None
    ##tflm1p = tflm1P()
    ##running_model_metrics['Pruning']['Tflite']['P1'] = tflm1p.metrics()

    # method 2
    # infer_paths['Pruning']['Tflite']['P2']
    # model_name['Pruning']['Tflite']['P2']
    ##tflm2p = tflm2P()
    ##running_model_metrics['Pruning']['Tflite']['P2'] = tflm2p.metrics()

    del running_model_metrics['Regular']['Pytorch']['fp16']
    del running_model_metrics['Quantization']['Tflite']['int8']
    del running_model_metrics['Pruning']['Tflite']
    
    return running_model_metrics
