import shutil
import os
import stat
from os import path

# name of the cloned repository
cloned_yolov5 = 'yolov5'
infer_folder = 'runs'

def remove_access_denied_folders(folder):
    # deleting already modified cloned repository
    if os.path.exists(os.path.join(folder)):
        for root, dirs, files in os.walk("./"+folder):  
            for dir in dirs:
                os.chmod(path.join(root, dir), stat.S_IRWXU)
            for file in files:
                os.chmod(path.join(root, file), stat.S_IRWXU)
        shutil.rmtree('./'+folder)

# considering only freshy cloned yolov5 repository
os.system('git clone https://github.com/ultralytics/yolov5.git')

if os.path.exists('runs/'):
    os.remove('runs/')