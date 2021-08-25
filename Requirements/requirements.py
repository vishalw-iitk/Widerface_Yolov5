''' To install the requirements if not already installed.\
    This step can be easily avoided if all the requirements are pre-installed beforehand '''

# Library imports
import os
import shutil

# now we need to install the packages from yolov5 repository
def run():
    '''
    Installs the requiremnts as mentioned in ultralytics yolov5 repo
    Installs the requiremnts which we have mentioned explicitly
    Installs cuda's torch library for CUDA support
    '''
    # Installs the requiremnts as mentioned in ultralytics yolov5 repo
    shutil.copy(os.path.join('..', 'yolov5', 'requirements.txt'), os.path.join('..', 'requirements.txt'))
    os.system('pip3 install -qr ../requirements.txt')

    os.remove('../requirements.txt')

    # Installs the requiremnts which we have mentioned explicitly
    shutil.copy(os.path.join('Requirements', 'requirements.txt'), os.path.join('..', 'requirements.txt'))
    os.system('pip3 install -qr ../requirements.txt')
    
    # Installs torch library for CUDA support
    os.system('pip3 install --no-cache torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html')
