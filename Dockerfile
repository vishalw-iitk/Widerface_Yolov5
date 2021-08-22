# FROM python:3

# WORKDIR /usr/src/app

# # RUN git clone ......
# RUN mkdir dts

# WORKDIR /usr/src/app/dts

# COPY Requirements/requirements.txt /usr/src/app/dts
# RUN pip3 install -qr requirements.txt

# # Comment it if cloning inside docker
# COPY . /usr/src/app/dts
# =================================
# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.05-py3

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

RUN python -m pip install --upgrade pip

RUN pip3 uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof

RUN pip3 install --no-cache torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

COPY Worked_on_yolov5/yolov5/requirements.txt .
RUN pip3 install --no-cache -qr requirements.txt

COPY Requirements/requirements.txt .
RUN pip3 install --no-cache -qr requirements.txt


# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN mkdir dts
WORKDIR /usr/src/app/dts

COPY . /usr/src/app/dts
# =================================
# FROM nvidia/cuda:11.1-base

# CMD nvidia-smi
#set up environment
# RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
# RUN apt-get install unzip
# RUN apt-get -y install python3
# RUN apt-get -y install python3-pip

# RUN apt-get install ffmpeg libsm6 libxext6  -y

# RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# WORKDIR /usr/src/app

# RUN git clone ......
# RUN mkdir dts

# WORKDIR /usr/src/app/dts

# Comment it if cloning inside docker
# COPY . /usr/src/app/dts

# RUN pip3 install -qr Requirements/requirements.txt

# RUN pip3 install -qr Worked_on_yolov5/yolov5/requirements.txt

# ===================================
# FROM python:3

# CMD nvidia-smi

# RUN apt-get update

# RUN apt-get install ffmpeg libsm6 libxext6  -y

# RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# WORKDIR /usr/src/app

# RUN git clone ......
# RUN mkdir dts

# WORKDIR /usr/src/app/dts

# Comment it if cloning inside docker
# COPY . /usr/src/app/dts

# RUN pip3 install -qr Requirements/requirements.txt

# RUN pip3 install -qr Worked_on_yolov5/yolov5/requirements.txt


# =======================

# ENTRYPOINT [ "python" ]

# CMD ["jupyter notebook"]
# CMD [ "dts/Pipeline.py", "--batch-size", "4", "--img-size", "416", "--epochs", "1", "--skip-training", "--skip-pruning", "skip-QAT-training" ]

#docker build -t widerface_image .
# docker run --name widerf_container -p 8888:8888 -it widerface_image bash
# Volume mounting :
# docker run -v "C:\Users\Vishal Waghmare\3D Objects\Mirafra_docs\Work\Training\Submission_example\Project\Colab:/usr/src/app" -v /usr/src/app/dts --name widerf_container -p 8888:8888 -it widerface bash

# jupyter notebook --allow-root --ip='0.0.0.0'
# 
# docker run --name widerf_container -p 8888:8888 -it widerface_image bash