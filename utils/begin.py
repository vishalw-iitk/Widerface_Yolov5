'''
To clone the Ultralytics-repository's latest version or\
    to continue working it's older version which we already have inside this repo.
'''

# Import libraries
import shutil
import os
import stat
import argparse
from os import path
import shutil
from distutils.dir_util import copy_tree

def remove_access_denied_folders(folder):
    '''
    Using it to delete the yolov5 repo so that either cloned or old-ultralytics version \
        can be placed at ../Working_dir location
    ARGUMENTS : 
        folder : folder to be deleted recursively
    '''
    if os.path.exists(os.path.join(folder)):
        for root, dirs, files in os.walk("./"+folder):  
            for dir in dirs:
                os.chmod(path.join(root, dir), stat.S_IRWXU)
            for file in files:
                os.chmod(path.join(root, file), stat.S_IRWXU)
        shutil.rmtree('./'+folder)


def main(opt):
    yolov5_repo_name = opt.yolov5_repo_name
    cloned_yolov5_path = os.path.join('..', yolov5_repo_name)
    worked_on_yolov5_path = os.path.join('Worked_on_yolov5', yolov5_repo_name)
    results_folder_path = opt.results_folder_path
    clone_updated_yolov5 = opt.clone_updated_yolov5

    # calling the functions to remove the yolov5/ and runs/ folder to avoid conflict
    remove_access_denied_folders(cloned_yolov5_path)
    remove_access_denied_folders(results_folder_path)
    
    if clone_updated_yolov5: # considering only freshly cloned yolov5 repository
        os.system('git clone https://github.com/ultralytics/yolov5.git')
        shutil.move(cloned_yolov5_path[3:], os.path.join('..', 'yolov5'))
    else: # To continue working with older ultralytic yolov5 repo version
        copy_tree(worked_on_yolov5_path, os.path.join('..', 'yolov5'))

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolov5-repo-name', default = 'yolov5', help='Better not to pass this argument unless the name of the repo itself is changed\
                        Not using this argument and keeping it default is completely fine. yolov5 repo at ../workdir will be deleted to allow cloning\
                        and to deal with old-ultralytics version')
    parser.add_argument('--results-folder_path', default= '../runs', help='training results will be stored inside ..runs/ directory')
    parser.add_argument('--clone-updated-yolov5', action='store_true', help='clone the updated yolov5 repository. This may not work if updates in the original yolv5 repo become incompatible with our setup')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


# if __name__ == "__main__":
opt = parse_opt(True)
main(opt)

