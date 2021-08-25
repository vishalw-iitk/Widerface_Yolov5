import os
import sys
sys.path.append('../..')
# from dts.model_paths import 

rel_path = '../../..'
Project_folder = 'Project'
add_path = os.path.join(Project_folder)

def frameworks(skip_training, running_model_paths, pre_trained_model_paths):
    if skip_training == False: #training
        framework_path = {
            "Pytorch":{
                "Regular_fp32": running_model_paths['Regular']['Pytorch']['fp32'],
                "Regular_fp16": running_model_paths['Regular']['Pytorch']['fp16']
                # "QAT quantized": 'dts/Model_compression/Quantization/Pytorch/QAT/runs/train/QAT/yolov5s_results6/weights/best.pt'
            },
            "ONNX":{
                "Regular_fp32": '../runs/train/yolov5s_results/weights/fp32/best.onnx'
                # 'Quantized_tfl_fp16' : '../runs/train/yolov5s_results/weights/fp32/best.onnx',
                # 'Quantized_tfl_int8' : '../runs/train/yolov5s_results/weights/fp32/best.onnx'
                # "QAT quantized": 'dts/Model_compression/Quantization/Pytorch/QAT/runs/train/QAT/yolov5s_results6/weights/best.onnx'
            },
            "tf_pb":{
                "Regular_fp32": '../runs/train/yolov5s_results/weights/fp32/best.pb/saved_model.pb',
                'Quantized_tfl_fp16' : "../runs/train/yolov5s_results/weights/fp32/best.pb/saved_model.pb",
                'Quantized_tfl_int8' : "../runs/train/yolov5s_results/weights/fp32/best.pb/saved_model.pb"
                # "QAT quantized": 'dts/Model_compression/Quantization/Pytorch/QAT/runs/train/QAT/yolov5s_results6/weights/best.pb/saved_model.pb'
            },
            "tflite":{
                "Regular_fp32": '../runs/train/yolov5s_results/weights/fp32/best.tflite',
                'Quantized_tfl_fp16' : "../Model_compression/Quantization/Tflite/fp16/best.tflite",
                'Quantized_tfl_int8' : "../Model_compression/Quantization/Tflite/int8/best.tflite"
                # "QAT quantized": 'dts/Model_compression/Quantization/Pytorch/QAT/runs/train/QAT/yolov5s_results6/weights/best.tflite'
            }
        }
    else:
        framework_path = {
            "Pytorch":{
                "Regular_fp32": pre_trained_model_paths['Regular']['Pytorch']['fp32'],
                "Regular_fp16": pre_trained_model_paths['Regular']['Pytorch']['fp16']
                # "QAT quantized": 'dts/Model_compression/Quantization/Pytorch/QAT/runs/train/QAT/yolov5s_results6/weights/best.pt'
            },
            "ONNX":{
                "Regular_fp32": '../Pre_trained_model/Regular/Pytorch/fp32/best.onnx'
                # 'Quantized_tfl_fp16' : '../runs/train/yolov5s_results/weights/fp32/best.onnx',
                # 'Quantized_tfl_int8' : '../runs/train/yolov5s_results/weights/fp32/best.onnx'
                # "QAT quantized": 'dts/Model_compression/Quantization/Pytorch/QAT/runs/train/QAT/yolov5s_results6/weights/best.onnx'
            },
            "tf_pb":{
                "Regular_fp32": '../Pre_trained_model/Regular/Pytorch/fp32/best.pb/saved_model.pb',
                'Quantized_tfl_fp16' : "../Pre_trained_model/Regular/Pytorch/fp32/best.pb/saved_model.pb",
                'Quantized_tfl_int8' : "../Pre_trained_model/Regular/Pytorch/fp32/best.pb/saved_model.pb"
                # "QAT quantized": 'dts/Model_compression/Quantization/Pytorch/QAT/runs/train/QAT/yolov5s_results6/weights/best.pb/saved_model.pb'
            },
            "tflite":{
                "Regular_fp32": '../Pre_trained_model/Regular/Pytorch/fp32/best.tflite',
                'Quantized_tfl_fp16' : "../Model_compression/Quantization/Tflite/fp16/best.tflite",
                'Quantized_tfl_int8' : "../Model_compression/Quantization/Tflite/int8/best.tflite"
                # "QAT quantized": 'dts/Model_compression/Quantization/Pytorch/QAT/runs/train/QAT/yolov5s_results6/weights/best.tflite'
            }
        }
    return framework_path
