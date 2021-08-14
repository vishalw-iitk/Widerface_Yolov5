import sys
sys.path.append('..')

from dts.utils import begin
from dts.model_paths import running_model_dictionary, train_results_dictionary
from dts.model_paths import pre_trained_model_dictionary
from dts.model_paths import train_results_dictionary
from dts.model_paths import model_defined_names
# from dts.Data_preparation import df_percent
from dts.Data_preparation import data_prep_yolo

from dts.Requirements import requirements
# from dts.model_paths import  update_the_paths




from dts.model_paths import frameworks
# from dts.Model_compression.Pruning import pruning
from dts.model_paths import update_to_running_paths_with_pretrianed
from dts.Model_conversion.fp_type_conversion import fp_type_conversion


import argparse

def main(opt):

    begin.run(
        yolov5_repo_name = opt.yolov5_repo_name,
        results_folder_path = opt.results_folder_path,
        clone_updated_yolov5 = opt.clone_updated_yolov5,
    )

    requirements.run(
        device = opt.device
    )

    data_prep_yolo.run(
        raw_dataset_path = opt.raw_dataset_path,
        partial_dataset = opt.partial_dataset,
        percent_traindata = opt.percent_traindata,
        percent_validationdata = opt.percent_validationdata,
        percent_testdata = opt.percent_testdata,
        arranged_data_path = opt.arranged_data_path,
        img_size = opt.img_size
    )

    from yolov5 import train
    from dts.Model_conversion import model_export
    from dts.Model_compression.Quantization import quantization
    from dts.Model_performance import inference_results
    running_model_paths = running_model_dictionary()
    pre_trained_model_paths = pre_trained_model_dictionary()
    framework_path = frameworks(opt.skip_training, running_model_paths, pre_trained_model_paths)
    train_results_paths = train_results_dictionary()
    model_names = model_defined_names()

    if opt.skip_training == True: # Use Pre-trained weighths
        running_model_paths = update_to_running_paths_with_pretrianed(running_model_paths, pre_trained_model_paths)
    else: # train the model
        # print(weights)
        train.run(
            weights = pre_trained_model_paths['Regular']['Pytorch']['fp32'] if opt.retrain_on_pre_trained else opt.weights,
            cfg = opt.cfg,
            data = opt.data,
            hyp = opt.hyp,
            rect = False,
            resume = False,
            nosave = False,
            noval = False,
            noautoanchor = False,
            evolve = 0, #doubt
            bucket = False,
            cache_images = True,
            image_weights = False,
            device = opt.device,
            multi_scale = False,
            single_cls = False,
            adam = False,
            sync_bn = False,
            workers = 8,
            project = train_results_paths['Regular']['Pytorch']['fp32'],
            entity = None,
            name = model_names['Regular']['Pytorch']['fp32'],
            exist_ok = False,
            quad = False,
            linear_lr = False,
            label_smoothing = 0.0,
            upload_dataset = False,
            bbox_interval = -1,
            save_period = -1,
            artifact_alias = 'latest',
            local_rank = -1,
            freeze = 0
        )
        if opt.clone_updated_yolov5 == True:
            fp_type_conversion('fp16_to_fp32', opt.device, running_model_paths)
        else:
            fp_type_conversion('fp132_to_fp16', opt.device, running_model_paths)
    
    # Not to use fp16 path onwards
    # Use fp32 path
    model_export.run(
        model_type_for_export = model_names['Regular']['Pytorch']['fp32'],
        framework_path = framework_path,
        model_names = model_names
    )

    # Pruning.run(opt.skip_training)

    quantization.run(       #Quantization.py
        skip_QAT_training = opt.skip_QAT_training,
        running_model_paths = running_model_paths,
        framework_path = framework_path,
        weights = pre_trained_model_paths['Regular']['Pytorch']['fp32'] if opt.retrain_on_pre_trained else opt.weights,
        # weights = running_model_paths['Regular']['Pytorch']['fp32'],
        repr_images = opt.repr_images,
        img = opt.img,
        ncalib = opt.ncalib,
        # weights = 'runs/train/yolov5s_results/weights/best.pt',
        cfg = opt.cfg,
        data = opt.data,
        hyp = opt.hyp,
        rect = False,
        resume = False,
        nosave = False,
        noval = False,
        noautoanchor = False,
        evolve = 0, #doubt
        bucket = False,
        cache_images = True,
        image_weights = False,
        device = opt.device,
        multi_scale = False,
        single_cls = False,
        adam = False,
        sync_bn = False,
        workers = 8,
        project = train_results_paths['Quantization']['Pytorch']['QAT'],
        entity = None,
        name = model_names['Quantization']['Pytorch']['QAT'],
        exist_ok = False,
        quad = False,
        linear_lr = False,
        label_smoothing = 0.0,
        upload_dataset = False,
        bbox_interval = -1,
        save_period = -1,
        artifact_alias = 'latest',
        local_rank = -1,
        freeze = 0
        ) 

    # model_conversion.run('Quantized')

    # test_the_models() #detect.py

    plot_results = inference_results.run(opt, running_model_paths) #mAP0.5, mAP0.5:0.95, fitness_score, latency, GFLOPs, Size
    # arrnagement of results like segregation and density_threshold using val.py

    # plot_the_performance.run(plot_results)
    



    



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolov5-repo-name', default= 'yolov5', help='')
    parser.add_argument('--results-folder_path', default= '../runs', help='')
    parser.add_argument('--clone-updated-yolov5', action='store_true', help='clone the updated yolov5 repository. This may not work if updates in the original yolv5 repo become incompatible with our setup')
    
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--raw-dataset-path', type=str, default = '../RAW_DATASET', help='Path of the raw dataset which was just arranged from the downloaded dataset')
    parser.add_argument('--arranged-data-path', type=str, default = '../ARRANGED_DATASET', help='Path of the arranged dataset')

    parser.add_argument('--partial-dataset', action='store_true', help='willing to select custom percentage of dataset')
    parser.add_argument('--percent-traindata', type=int, help=' percent_of_the_train_data_required')
    parser.add_argument('--percent-validationdata', type=int, help=' percent_of_the_validation_data_required')
    parser.add_argument('--percent-testdata', type=int, help=' percent_of_the_test_data_required')

    parser.add_argument('--batch-size', type=int, default=128, help='')
    parser.add_argument('--epochs', type=int, default=300, help='')    
    parser.add_argument('--img-size', type=int, default = 416, help = 'Image size suitable for feeding to the model')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')

    parser.add_argument('--cfg', type=str, default='../yolov5/models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='../yolov5/data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    
    parser.add_argument('--retrain-on-pre-trained', action='store_true', help= 'Retrain using the pre-trained weights')
    parser.add_argument('--skip-training', action='store_true', help='skip the time taking regular training')
    parser.add_argument('--skip-QAT-training', action='store_true', help='skip the time taking Quantizze Aware training training')
    parser.add_argument('--qat-project', default='../runs_QAT/train', help='save to project/name')
    parser.add_argument('--qat-name', default='exp', help='save to project/name')
    parser.add_argument('--QAT-epochs', type=int, default=50, help='')
    parser.add_argument('--repr-images', type=str, default='../ARRANGED_DATASET/images/validation/', help='path of representative dataset')
    parser.add_argument('--img', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--ncalib', type=int, default=100, help='number of calibration images')

    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--local-rank', type=int, default=-1, help='DDP parameter, do not modify')

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



# Pytorch
# Regular_training
    # Test_the_models
    # inference results
    # export
        #Pytorch
            # Quantization
                # Test_the_models
                # Quantized model export
                # inference results
            # Pruning
                # Test_the_models
                # Pruned model exports
                # inference results
        #Tflite
            # Quantization
                # Test_the_models
                # inference results
            # Pruning
                # Test_the_models
                # inference results
        