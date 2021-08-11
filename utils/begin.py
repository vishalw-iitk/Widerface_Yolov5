import shutil
import os
import stat
import argparse
from os import path

# from dts.model_paths import running_model_dictionary


def remove_access_denied_folders(folder):
    # deleting already modified cloned repository
    if os.path.exists(os.path.join(folder)):
        for root, dirs, files in os.walk("./"+folder):  
            for dir in dirs:
                os.chmod(path.join(root, dir), stat.S_IRWXU)
            for file in files:
                os.chmod(path.join(root, file), stat.S_IRWXU)
        shutil.rmtree('./'+folder)


def main(opt):
    yolov5_repo_name = opt.yolov5_repo_name
    cloned_yolov5_path = '../'+yolov5_repo_name
    worked_on_yolov5_path = os.path.join('Worked_on_yolov5', yolov5_repo_name)
    results_folder_path = opt.results_folder_path
    clone_updated_yolov5 = opt.clone_updated_yolov5

    # running_model_paths = running_model_dictionary()

    remove_access_denied_folders(cloned_yolov5_path)
    # remove_access_denied_folders(results_folder_path)
    
    # considering only freshy cloned yolov5 repository
    if clone_updated_yolov5:
        # os.system('rm '+yolov5_repo_name)
        os.system('git clone https://github.com/ultralytics/yolov5.git')
        # del running_model_paths['Regular']['Pytorch']['fp32']
        # del running_model_paths['Regular']['Tflite']['fp32']
        os.system('mv '+cloned_yolov5_path[3:]+' ../')
    else:
        os.system('cp -r '+worked_on_yolov5_path+' ../')

    # return running_model_paths


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolov5_repo_name', default= 'yolov5', help='')
    parser.add_argument('--results_folder_path', default= '../runs', help='')
    parser.add_argument('--clone-updated-yolov5', action='store_true', help='clone the updated yolov5 repository. This may not work if updates in the original yolv5 repo become incompatible with our setup')
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

