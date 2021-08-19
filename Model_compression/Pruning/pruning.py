import argparse
import os
# from dts.model_paths import running_model_dictionary
# from dts.model_paths import pre_trained_model_dictionary
# from dts.model_paths import frameworks
from dts.Model_conversion import model_export
from dts.Model_compression.Pruning.Pytorch import prune_train
from dts.Model_compression.Pruning.Pytorch.P4 import train
from dts.model_paths import train_results_dictionary
from dts.model_paths import model_defined_names
from dts.model_paths import pre_trained_model_dictionary


class Pruning(object):
    def __init__(self):
        # base_model_path
        pass

class Pytorch(Pruning):
    def __init__(self):
        Pruning.__init__(self)
        # new model definition

class Tflite(Pruning):
    def __init__(self):
        Pruning.__init__(self)


# Pytorch
class P1(Pytorch):
    def __init__(self):
        Pytorch.__init__(self)
        # to save path
    def prune(self, **kwargs):
        prune_train.run(**kwargs)
        
class P2(Pytorch):
    def __init__(self):
        Pytorch.__init__(self)
        # to save path
    def prune(self, **kwargs):
        prune_train.run(**kwargs)

class P3(Pytorch):
    def __init__(self):
        Pytorch.__init__(self)
        # to save path
    def prune(self, **kwargs):
        prune_train.run(**kwargs)
        
class P4(Pytorch):
    def __init__(self):
        Pytorch.__init__(self)
        # to save path
    def prune(self, **kwargs):
        train.run(**kwargs)


# Tflite
class TFP1(Tflite):
    def __init__(self):
        Tflite.__init__(self)
    def prune(self, **kwargs):
        model_export.run(**kwargs)

class TFP2(Tflite):
    def __init__(self):
        Tflite.__init__(self)
    def prune(self, **kwargs):
        model_export.run(**kwargs)


def main(opt):
    # running_model_paths = opt.running_model_paths
    # framework_path = opt.framework_path
    # weights = running_model_paths['Regular']['Pytorch']['fp32']
    
    train_results_paths = train_results_dictionary()
    model_names = model_defined_names()
    pre_trained_model_paths = pre_trained_model_dictionary()
    # running_model_paths = opt.running_model_paths
    # framework_path = opt.framework_path

    if opt.skip_P1_training == False:
    #random re-init
        p1_py = P1()
        for i in range(opt.num_iterations):
            p1_py.prune(weights=opt.weights,
                        batch_size=opt.batch_size, imgsz=opt.img,epochs=opt.prune_retrain_epochs,
                        project = train_results_paths['Pruning']['Pytorch']['P1'],
                        name=model_names['Pruning']['Pytorch']['P1'],     
                        prune_perc=opt.prune_perc,prune_iter=i,
                        random_reinit=False, theta0_reinit=False,
                        data=opt.data,cfg=opt.cfg, 
                        hyp=opt.hyp,exist_ok=True,cache_images=True,device=opt.device)


    if opt.skip_P2_training == False:
        #theta0 re-init
        p2_py = P2()
        for i in range(opt.num_iterations):
            p2_py.prune(weights=opt.weights, 
                    batch_size=opt.batch_size, imgsz=opt.img,epochs=opt.prune_retrain_epochs,
                    project = train_results_paths['Pruning']['Pytorch']['P2'],
                    name=model_names['Pruning']['Pytorch']['P2'],     
                    prune_perc=opt.prune_perc,prune_iter=i,
                    random_reinit=False, theta0_reinit=True,theta0_weights=pre_trained_model_paths['Pruning']['Pytorch']['theta0'],
                    data=opt.data,cfg=opt.cfg, 
                    hyp=opt.hyp,exist_ok=True)

    if opt.skip_P3_training == False:
        #no reinit
        p3_py = P3()
        for i in range(opt.num_iterations):
            p3_py.prune(weights=opt.weights, 
                    batch_size=opt.batch_size, imgsz=opt.img,epochs=opt.prune_retrain_epochs,
                    project = train_results_paths['Pruning']['Pytorch']['P3'],
                    name=model_names['Pruning']['Pytorch']['P3'],     
                    prune_perc=opt.prune_perc,prune_iter=i,
                    random_reinit=False, theta0_reinit=False,
                    data=opt.data,cfg=opt.cfg,
                    hyp=opt.hyp,exist_ok=True)
    
    if opt.skip_Pruning_P4 == False:
        p4_py = P4()
        p4_py.prune(
            pre_trained_model_paths = opt.pre_trained_model_paths,
            weights = opt.weights,
            # weights = opt.pre_trained_model_paths['Regular']['Pytorch']['fp32'] if opt.retrain_on_pre_trained else opt.weights,
            data = opt.data,
            cfg = opt.cfg,
            hyp = opt.hyp,
            device = opt.device,
            cache_images = True,
            project = train_results_paths['Pruning']['Pytorch']['P4'],
            name = model_names['Pruning']['Pytorch']['P4'],
            st = opt.st,
            sr = opt.sr
            )

    # p2_py = P2()
    # p2_py.prune()

    # Tflite
    # tfl_p1 = TFP1()
    # tfl_p1.prune(
        # model_type_for_export = 'pruned_tfl_fp16',
        # framework_path = framework_path
        # )

    # tfl_p2 = TFP2()
    # tfl_p2.prune(
        # model_type_for_export = 'pruned_tfl_int8',
        # framework_path = framework_path,
        # repr_images = opt.repr_images,
        # img = opt.img,
        # ncalib = opt.ncalib
    # )
        

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-train', action='store_true', help='skip the time taking regular training')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--qat-project', default='../runs_QAT/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--qat-name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def run(**kwargs):
    # Usage: import train; train.run(imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


