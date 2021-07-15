import os

CUDA_compatible_command = 'pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html'


if os.path.exists('gpu_requirements.txt'):
    os.system('cp gpu_requirements.txt requirements.txt')
    os.remove('gpu_requirements.txt')

# os.system('pip3 install -qr requirements.txt')
file1 = open('requirements.txt', 'r')
Lines = file1.readlines()

# skip = False
skip = True

if not skip:
    for line in Lines:
        if line[0]!='#':
            os.system('pip3 install '+line)

    os.system(CUDA_compatible_command)

