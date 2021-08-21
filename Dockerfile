# FROM python:3

# WORKDIR /usr/src/app

# # RUN git clone ......
# RUN mkdir dts

# WORKDIR /usr/src/app/dts

# COPY Requirements/requirements.txt /usr/src/app/dts
# RUN pip3 install -qr requirements.txt

# # Comment it if cloning inside docker
# COPY . /usr/src/app/dts
# ===================================
FROM python:3

WORKDIR /usr/src/app

# RUN git clone ......
RUN mkdir dts

WORKDIR /usr/src/app/dts

# Comment it if cloning inside docker
COPY . /usr/src/app/dts

RUN pip3 install -qr Requirements/requirements.txt

RUN pip3 install -qr Worked_on_yolov5/yolov5/requirements.txt

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

# =======================

# ENTRYPOINT [ "python" ]

# CMD ["jupyter notebook"]
# CMD [ "dts/Pipeline.py", "--batch-size", "4", "--img-size", "416", "--epochs", "1", "--skip-training", "--skip-pruning", "skip-QAT-training" ]

#docker build -t widerface_image .
# docker run --name widerf_container -p 8888:8888 -it widerface_image bash
# Volume mounting :
# docker run -v "C:\Users\Vishal Waghmare\3D Objects\Mirafra_docs\Work\Training\Submission_example\Project\Colab:/usr/src/app" -v /usr/src/app/dts --name widerf_container -p 8888:8888 -it widerface_image bash

# jupyter notebook --allow-root --ip='0.0.0.0'
# 
# docker run --name widerf_container -p 8888:8888 -it widerface_image bash