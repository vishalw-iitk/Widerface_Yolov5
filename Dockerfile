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
