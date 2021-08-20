# <div align="center"> YOLOv5 🚀 WIDER FACE </div>
<div align="center">
<p>
YOLOv5 🚀 is a family of object detection architectures and models pretrained on the COCO dataset and repo <a href="https://github.com/ultralytics/yolov5.git">YOLOv5 🚀 in PyTorch</a>. We have taken reference from that repo and trained a model on Wider Face data. We have further experiment with model compression techniques Quantization and Pruning. 
</p></div>

## <div align="center"> SETUP </div>
***
<b>STEP 1 :</b>
- Download dataset from : http://shuoyang1213.me/WIDERFACE/index.html
    - Train : 12880
    - Validation : 3226
    - Test : 16097

<b>STEP 2 :</b>
Rearrange the downloaded dataset as below :
- RAW_DATASET
    - train
        - images
            - images_folders_of_various_classes
        - labels
            - .mat and .txt file with img_names and labels
     - validation
         - images
            - images_folders_of_various_classes
        - labels
            - labels in .mat and .txt file with img_names and labels
     - test
         - images
            - images_folders_of_various_classes
         - labels
            - labels in .mat and .txt file with img_names ONLY

<b>STEP 3 :</b>
```bash
$ mkdir Project
$ cd Project
```

<b>STEP 4 :</b>
```bash
$ mv Path{RAW_DATASET} .
```

<b>STEP 5 :</b>
```bash
$ git clone https://github.com/kuchlous/dts.git
$ git checkout complete
```
<b>STEP 6 :</b>
```bash
$ cd dts/
```

## <div align="center"> Pipeline </div>
***
Every pipeline run, you will get the summary of the model performance in the form of well analysed plots in the plot_metrics folder.

<details open>
<summary>Arranging Repo </summary>
Arranging the repo is integrated into the pipeline.
You can always attempt to get all the results with the latest ultralytics repository changes
If the recent version of the ultralytics repository is not compatible with dts repository codes, then we can skip cloning and use the older version of ultralytics repo.

```python 
'''args 
    --yolov5-repo-name # default= 'yolov5' 
'''
# python utils/begin.py
```
</details>
<details open>
<summary>Preparing Data </summary>

```python 
'''args 
    --device  #cuda device, i.e. 0 or 0,1,2,3 or cpu
    --raw-dataset-path  # default = '../RAW_DATASET' Path of the raw dataset which was just arranged from the downloaded dataset
    --arranged-data-path # default = '../ARRANGED_DATASET' 'Path of the arranged dataset
    --partial-dataset  # willing to select custom percentage of dataset
    --percent-traindata'  # percent_of_the_train_data_required 1=1%
    --percent-validationdata # percent_of_the_validation_data_required 1=1%
    .--percent-testdata # percent_of_the_test_data_required
'''
# python Pipeline.py --clone-updated-yolov5 --partial-dataset --percent-traindata 1 --percent-validationdata 1 --percent-testdata 1

# python Data_preparation/data_prep_yolo.py
```
</details>
<details open>
<summary>Skip Traning and Retraning </summary>

```python 
'''args 
    --batch-size 128 --img-size 416 --epochs 1 --device '0' 
    --skip-training  # Take pre-trained weights 
    --retrain-on-pre-trained # Fine-Tuning or retrain on pre-trained weights
'''
# Sctrach 
python Pipeline.py --batch-size 32 --img-size 416 --epochs 1 --device '0' --adam --prune-iterations 1 --prune-retrain-epochs 1

#Finetune training
python Pipeline.py --batch-size 32 --img-size 416 --epochs 1 --device '0' --prune-iterations 1 --prune-retrain-epochs 1 --retrain-on-pre-trained
```
</details>
<details open>
<summary>Model Export </summary>
</details>
<details open>
<summary>Pruning</summary>

```python 
'''args 
    -- skip-pruning 
    --skip-P1-training #random re-init
    --skip-P2-training #theta0 re-init
    --skip-P3-training #no reinit
    --skip-P4-training 
'''
python Pipeline.py --skip-P2-training --skip-P3-training --prune-iterations 5 --prune-retrain-epochs 1 --prune-perc 30 --img-size 416 --batch-size 1
```
</details>
<details open>
<summary>Quantization</summary>
<details open>
<summary>PTQ</summary>

```python 
'''args 
    --skip-QAT-training  
'''
python Pipeline.py --clone-updated-yolov5 --partial-dataset --percent-traindata 1 --percent-validationdata 1 --percent-testdata 1 --batch-size 4 --epochs 1 --retrain-on-pre-trained --single-cls --skip-training --skip-QAT-training
```
</details>
<details open>
<summary>QAT</summary>

```python 
# perform short training and do inference on all the models(fine-tuning/pre-trained)
python Pipeline.py --batch-size 32 --img-size 416 --epochs 1 --device '0' --adam --prune-iterations 1 --prune-retrain-epochs 1 --retrain-on-pre-trained --prune-infer-on-pre-pruned-only
```
</details>
</details>

## <div align="center"> Results </div>
***

<div align="center">

|Model |size<br><sup>(pixels) |dtype |mAP<sup>val<br>0.5 |mAP<sup>val<br>0.5:0.95 |fitness<sup>val |latency<br><sup>(ms) | |GFOPs<br><sup>416 |Size<br><sup> (Mb)
|---                    |---  |---  |---      |---      |---      |---     |---|---   |---
|YOLOv5s      |416  |fp16  |0.605|0.31 |-     |**-** |   |6.92   |14.4
|             |     |     |         |         |         |        |   |      |
|Pruning_1(G)     |416 |fp16     |-     |-     |-     |-    |   |-  |-
|Pruning_2(P)     |416 |fp16     |-     |-     |-     |-    |   |-  |-
|                 |     |     |         |         |         |        |   |      |
|Q_tf             |416 |fp16     |-     |-     |-     |-    |   |-  |-
|PTQ     |416 |int8     |-     |-     |-     |-    |   |-  |-
|QAT     |416 |int8     |-     |-     |-     |-    |   |-  |-
|                       |     |     |         |         |         |        |   |      |
</div>
<details>
<summary>Table Notes (<b>Reproduce</b>)</summary>

</details>
