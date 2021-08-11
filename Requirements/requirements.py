import os
import argparse

CUDA_compatible_command = 'pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html'

# try:
#     # if requirements.txt is absent or it is blank then we get error while uninstalling it
#     # if os.path.exists('requirements.txt'):
#         # os.system('cp requirements.txt gpu_requirements.txt')
#     os.system('pip3 freeze > ../requirements.txt && pip3 uninstall -r ../requirements.txt -y 2> /dev/null') #not displaying error message if requirements.txt is empty
#     os.remove('../requirements.txt')
# except:
#     #if we don't get uninstallation error. It means we have uninstallation step was done correctly
#     pass

# now we need to install the packages from yolov5 repository
def main(opt):
    device = opt.device
    if opt.device == device:
        os.system('cp ../yolov5/requirements.txt ../requirements.txt')
    else:
        os.system('cp gpu_requirements.txt ../requirements.txt')
    
    os.system('pip3 install -qr ../requirements.txt')

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
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
