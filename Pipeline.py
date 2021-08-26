'''
To perform face detection over WIDERFACE dataset and get the trained model and \
    then optimize the obtained model using Pruning and Quantization.
Model architechture  : Yolov5s
Implementation       :
    -Model trained   :   Pytorch fp16 and fp32 models         |   Tflite fp32 model via export
    -ONNX export     :   Pytorch->ONNX->tf_pb_keras->TFLITE
    -Quantization:
        Pytorch  :   Quantize Aware training(fp32 -> int8)    |   Static Post Training Quantization(PTQ)(fp32 -> int8)
        Tflite   :    fp32 -> fp16(PTQ)                       |   fp32 -> int8(PTQ) (Inference not implemented yet)
    -Pruning     :   Global Unstructured (P1, P2 here)    |   Channel-wise Structured (called P4 here)
'''

''' Importing the libraries '''
import sys
import argparse

''' Adding the file import root to the parent of Pipeline.py '''
sys.path.append('..')

''' Files and functions imports '''
from dts.utils import begin
from dts.model_paths \
    import \
        running_model_dictionary, train_results_dictionary, pre_trained_model_dictionary,\
        train_results_dictionary, model_defined_names, prune_with_pre_trained_only, frameworks, \
        update_to_running_paths_with_pretrianed


def main(opt):
    '''
    To clone the Ultralytics-repository's latest version or\
    to continue working on the version which we already have inside this repo.
    '''
    begin.run(
        yolov5_repo_name = opt.yolov5_repo_name,
        results_folder_path = opt.results_folder_path,
        clone_updated_yolov5 = opt.clone_updated_yolov5,
    )

    '''
    To install the requirements if not already installed.
    This step can be easily avoided if all the requirements are pre-installed beforehand
    '''
    from dts.Requirements import requirements
    requirements.run()

    '''Data preparation step'''
    from dts.Data_preparation import data_prep_yolo
    data_prep_yolo.run(
        raw_dataset_path = opt.raw_dataset_path, arranged_data_path = opt.arranged_data_path, img_size = opt.img_size,
        partial_dataset = opt.partial_dataset,
        percent_traindata = opt.percent_traindata, percent_validationdata = opt.percent_validationdata, percent_testdata = opt.percent_testdata
    )

    '''
    Now, as the system is setup with Repository, requiremnts and dataset,\
    we can import the yolov5 repo libraries to use in the codes ahead
    '''
    from yolov5 import train
    from dts.Model_conversion import model_export
    from dts.Model_compression.Quantization import quantization
    from dts.Model_compression.Pruning import pruning
    from dts.Model_performance import inference_results, plot_the_performance
    from dts.Model_conversion.fp_type_conversion import fp_type_conversion

    '''Getting the model names and initial model paths'''
    running_model_paths = running_model_dictionary()
    pre_trained_model_paths = pre_trained_model_dictionary()
    framework_path = frameworks(opt.skip_training, running_model_paths, pre_trained_model_paths)
    train_results_paths = train_results_dictionary()
    model_names = model_defined_names()

    '''The paths updated here will be taken for inference results'''
    running_model_paths = update_to_running_paths_with_pretrianed(running_model_paths, pre_trained_model_paths, opt.skip_training, opt.skip_QAT_training, opt.skip_pruning, opt.skip_P1_training, opt.skip_P2_training, opt.skip_P4_training)

    '''Training the model. Either from scratch, or by using the widerface pre-trained weights'''
    if opt.skip_training == False:
        train.run(
            weights = pre_trained_model_paths['Regular']['Pytorch']['fp32'] if opt.retrain_on_pre_trained else opt.weights,
            cfg = opt.cfg, data = opt.data, hyp = opt.hyp,
            rect = False, resume = False, nosave = False, noval = False, noautoanchor = False, evolve = 0, #doubt
            bucket = False, cache_images = True, image_weights = False, device = opt.device, multi_scale = False, single_cls = False,
            adam = opt.adam, sync_bn = False, workers = 8, entity = None,
            project = train_results_paths['Regular']['Pytorch']['fp32'],
            name = model_names['Regular']['Pytorch']['fp32'],
            exist_ok = False, quad = False, linear_lr = False, label_smoothing = 0.0, upload_dataset = False,
            bbox_interval = -1, save_period = -1, artifact_alias = 'latest', local_rank = -1, freeze = 0
        )
        if opt.clone_updated_yolov5 == True:
            conversion_type = 'fp16_to_fp32'
        else:
            conversion_type = 'fp32_to_fp16'
        fp_type_conversion(conversion_type, opt.device, opt.cfg, opt.data, opt.hyp, opt.single_cls, opt.img_size, running_model_paths)

    # Not to use fp16 path onwards
    # Use fp32 path
    '''
    Basic model export step
    Once the pytorch model is trained, we can export it to ONNX -> tf_pb_keras -> TFLITE model
    Same is done with pre-trained models if pre-trained weights are being used
    '''
    model_export.run(
        model_type_for_export = model_names['Regular']['Pytorch']['fp32'],
        framework_path = framework_path,
        model_names = model_names
    )

    '''
    ****************  PRUNING  *******************
    Two major section of pruning are implemented
    1) Unstructured with different initialization schemes
    2) Structured
    '''
    pruning.run(
        skip_pruning = opt.skip_pruning,
        skip_P1_training = opt.skip_P1_training, skip_P2_training= opt.skip_P2_training, skip_P4_training = opt.skip_P4_training,
        running_model_paths = running_model_paths, pre_trained_model_paths = pre_trained_model_paths,
        weights = running_model_paths['Regular']['Pytorch']['fp32'] if opt.retrain_on_pre_trained else opt.weights,
        prune_retrain_epochs = opt.prune_retrain_epochs, num_iterations = opt.prune_iterations, prune_perc = opt.prune_perc,
        data = opt.data, cfg = opt.cfg, hyp = opt.hyp, img = opt.img_size, device = opt.device, batch_size = opt.batch_size,
        cache_images = True,
        st = True, sr = 0.0001
        )

    '''
    ****************  QUANTIZATION  *******************
    Four Quantization schemes implemented
    1) Quantize aware training
    2) Static Post training Quantization(Static PTQ)
    3) Tflite fp32 PTQ
    4) Tflite int8 PTQ
    '''
    quantization.run(
        skip_QAT_training = opt.skip_QAT_training,
        running_model_paths = running_model_paths, framework_path = framework_path,
        weights = running_model_paths['Regular']['Pytorch']['fp32'] if opt.retrain_on_pre_trained else opt.weights,
        repr_images = opt.repr_images, batch_size_QAT = opt.batch_size_QAT, img = opt.img, ncalib = opt.ncalib,
        cfg = opt.cfg, data = opt.data, hyp = opt.hyp,
        rect = False, resume = False, nosave = False, noval = False, noautoanchor = False, evolve = 0, #doubt
        bucket = False, cache_images = True,
        image_weights = False, device = opt.device, multi_scale = False,
        single_cls = False, adam = opt.adam,
        sync_bn = False, workers = 8,
        entity = None, exist_ok = False, quad = False, linear_lr = False, label_smoothing = 0.0, upload_dataset = False,
        bbox_interval = -1, save_period = -1, artifact_alias = 'latest', local_rank = -1, freeze = 0
        ) 

    # Not implemented yet # model_conversion.run('Quantized')

    # Not implemented yet in the pipeline # test_the_models() #detect.py

    ''' running_model_paths_modification for pruned model with pre-trained/pruned stored weights'''
    if opt.prune_infer_on_pre_pruned_only == True:
        running_model_paths = prune_with_pre_trained_only(running_model_paths, pre_trained_model_paths)
    
    '''
    Inference on every model which have been implemented above.
    Getting the inference results stored inside Model performance folder and also return the \
    performance dictionary so as to plot the performance results
    Inference not yet available for Tflite int8 quantized model
    '''
    plot_results = inference_results.run(opt, running_model_paths) #mAP0.5, mAP0.5:0.95, fitness_score, latency, GFLOPs, Size
    print(plot_results)
    
    # Not implemented yet
    # arrnagement of results like segregation and density_threshold using val.py
    
    '''
    Getting the model performance plots stored inside plot_metrics folder
    '''
    plot_the_performance.run(plot_results,save_dir = '../plot_metrics')




    



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolov5-repo-name', default = 'yolov5', help='Better not to pass this argument unless the name of the repo itself is changed\
        Not using this argument and keeping it default is completely fine. yolov5 repo at ../workdir will be deleted to allow cloning and to deal with old-ultralytics version')
    parser.add_argument('--results-folder_path', default= '../runs', help='training results will be stored inside ..runs/ directory')
    parser.add_argument('--clone-updated-yolov5', action='store_true', help='clone the updated yolov5 repository. This may not work if updates in the original yolv5 repo become incompatible with our setup')
    
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--raw-dataset-path', type=str, default = '../RAW_DATASET', help='Path of the raw dataset which was just arranged from the downloaded dataset')
    parser.add_argument('--arranged-data-path', type=str, default = '../ARRANGED_DATASET', help='Path of the arranged dataset')

    parser.add_argument('--partial-dataset', action='store_true', help='willing to select custom percentage of dataset')
    parser.add_argument('--percent-traindata', type=int, default=100, help=' percent_of_the_train_data_required')
    parser.add_argument('--percent-validationdata', type=int, default=100, help=' percent_of_the_validation_data_required')
    parser.add_argument('--percent-testdata', type=int, default=100, help='percent_of_the_test_data_required')

    parser.add_argument('--batch-size', type=int, default=128, help='training batch size')
    parser.add_argument('--batch-size-QAT', type=int, default=64, help='training batch size for Quantize aware training')

    parser.add_argument('--epochs', type=int, default=300, help='')   
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer') 
    parser.add_argument('--img-size', type=int, default = 416, help = 'Image size suitable for feeding to the model')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')

    parser.add_argument('--cfg', type=str, default='../yolov5/models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='../yolov5/data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    
    parser.add_argument('--retrain-on-pre-trained', action='store_true', help= 'Retrain using the pre-trained weights')
    parser.add_argument('--skip-training', action='store_true', help='skip the time taking regular training')
    parser.add_argument('--skip-QAT-training', action='store_true', help='skip the time taking Quantizze Aware training training')
    parser.add_argument('--qat-project', default='../runs_QAT/train', help='save to project/name')
    parser.add_argument('--qat-name', default='exp', help='save to project/name')
    parser.add_argument('--QAT-epochs', type=int, default=50, help='')
    parser.add_argument('--batch-size-inferquant', type=int, default=32, help='batch size for quantization inference')

    parser.add_argument('--repr-images', type=str, default='../ARRANGED_DATASET/images/validation/', help='path of representative dataset')
    parser.add_argument('--img', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--ncalib', type=int, default=100, help='number of calibration images')

    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--local-rank', type=int, default=-1, help='DDP parameter, do not modify')

    parser.add_argument('--skip-pruning', action='store_true', help='skip the time taking Pruning training')
    parser.add_argument('--skip-P1-training', action='store_true', help='skip the time taking Pruning m1 training')
    parser.add_argument('--skip-P2-training', action='store_true', help='skip the time taking Pruning m2 training')
    parser.add_argument('--skip-P4-training', action='store_true', help='skip the time taking Pruning m4 training')
    parser.add_argument('--prune-infer-on-pre-pruned-only', action='store_true', help='pruning inference on pre-pruned stored model only and not on recently pruned in pipeline')
    parser.add_argument('--prune-iterations', type=int, default=5, help='prune+retrain total number of iterations') 
    parser.add_argument('--prune-retrain-epochs', type=int, default=100, help=' number of retrain epochs after pruning')
    parser.add_argument('--prune-perc', type=int, default=30, help=' initial pruning percentage')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
