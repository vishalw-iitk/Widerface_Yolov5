import argparse
from dts.model_paths import train_results_dictionary
from dts.model_paths import model_defined_names
from dts.Model_compression.Pruning.pruning_common import *


def main(opt):
    
    if opt.skip_pruning == True:
        return
    
    train_results_paths = train_results_dictionary()
    model_names = model_defined_names()

    '''
    P1 Pruning : Pruning on previously trained weights with Random Reinitialize.
    P2 Pruning : Pruning on previously trained weights with theta0 Reinitialize.
    P4 Pruning : Structured pruning 
    '''

    # random re-init
    P1(opt, 'Pruning', 'Pytorch', 'P1', train_results_paths, model_names).iterator() if opt.skip_P1_training == False else None
    

    #theta0 re-init
    P2(opt, 'Pruning', 'Pytorch', 'P1', train_results_paths, model_names).iterator() if opt.skip_P2_training == False else None

    # Structure channel-wise pruning
    P4(opt, 'Pruning', 'Pytorch', 'P4', train_results_paths, model_names).prune() if opt.skip_P4_training == False else None


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--skip-pruning', action='store_true', help='skip the time taking Pruning training')
    parser.add_argument('--skip-P1-training', action='store_true', help='skip the time taking Pruning m1 training')
    parser.add_argument('--skip-P2-training', action='store_true', help='skip the time taking Pruning m2 training')
    parser.add_argument('--skip-P4-training', action='store_true', help='skip the time taking Pruning m4 training')  
    parser.add_argument('--prune-infer-on-pre-pruned-only', action='store_true', help='pruning inference on pre-pruned stored model only and not on recently pruned in pipeline')
    parser.add_argument('--prune-iterations', type=int, default=5, help='prune+retrain total number of iterations') 
    parser.add_argument('--prune-retrain-epochs', type=int, default=100, help=' number of retrain epochs after pruning')
    parser.add_argument('--prune-perc', type=int, default=30, help=' initial pruning percentage')
    parser.add_argument('--st', action='store_true',default=True, help='train with L1 sparsity normalization')
    parser.add_argument('--sr', type=float, default=0.001, help='L1 normal sparse rate')

    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--qat-name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
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


