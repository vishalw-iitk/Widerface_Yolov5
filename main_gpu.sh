python begin.py
python gpu_req.py
python data_prep.py
python ../yolov5/train.py --img 416 --batch 4 --epochs 2 --data 'data.yaml' --cfg ../yolov5/models/yolov5s.yaml --hyp ../yolov5/data/hyps/hyp.scratch.yaml --project ../runs/train --weights '' --name yolov5s_results  --cache
python ../yolov5/detect.py --source ../ARRANGED_DATASET_TEST/images/test/ --weights ../runs/train/yolov5s_results/weights/best.pt --project ../runs/detect --conf-thres 0.015 --iou-thres 0.3
