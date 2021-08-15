import sys
# sys.path.append("../..")
# sys.path.append(path.join(path.dirname(__file__), '..'))
# from yolov5.models.yolo import Model


# from dts.Model_compression.model_conversion.model_paths import framework_path

import torch
import onnx
import os

rel_path = '../..'

class load_the_model:
    def __init__(self, device_name):
        # self.model_path = model_path
        self.device_name = device_name
        # self.model_name_user_defined = model_name_user_defined
    def load_pytorch(self, model_path, model_name_user_defined, cfg, imgsz, data, hyp, single_cls, model_class = 'any'):
        if model_class == 'Regular':
            # attempt load (loads the normal model in fp32)
            from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.models.experimental import attempt_load
            self.model = attempt_load(model_path, map_location=self.device_name)  # load FP32 model
            
        elif model_class == 'QAT quantized':
            # infer_qat load (load the QAT quantized qint8 model)
            # from Model_compression.Quantization.Pytorch.QAT import load_and_infer
            # from dts.Model_compression.Quantization.Pytorch.QAT.yolov5_repo.load_and_infer import quantized_load
            self.model = quantized_load(model_path, cfg, self.device_name, imgsz, data, hyp, single_cls)
            # self.model = infer_qat.run(model_path, map_location=torch.device(self.device_name))
            
        else:
            self.model = torch.load(model_path, map_location = torch.device(self.device_name))
            pass
        # try:
        #     self.model_architechture = model_class(cfg)
        #     self.model_info = torch.load(self.model_path, map_location=torch.device(self.device_name))
        # except:
        #     self.model_architechture = model_class(cfg)
        #     self.model = torch.load(self.model_path, map_location=torch.device(self.device_name))
        self.statement = model_name_user_defined + " has been loaded"
    
    def load_onnx(self, model_path, model_name_user_defined):
        self.model = onnx.load(model_path)
        self.statement = model_name_user_defined + " has been loaded"

    def load_tf_pb(self, model_path, model_name_user_defined):
        import tensorflow as tf
        # we are probably not loading the tf_pb model in the tf_pb format
        # self.statement = self.model_name_user_defined + " has been loaded"
        pass
    
    def load_tf_pb_as_tflite_converter(self, model_path, model_name_user_defined):
        import tensorflow as tf
        self.converter =  tf.compat.v1.lite.TFLiteConverter.from_saved_model(model_path)
        self.statement = model_name_user_defined + " has been loaded"
    
    def load_tflite(self, model_path, model_name_user_defined):
        import tensorflow as tf
        with open(model_path, 'rb') as fid:
            self.model = fid.read()
        self.statement = model_name_user_defined + " has been loaded"

# MLmodel = load_the_model('cpu')

# framework = 'Pytorch'
# model_type = 'Proper'
# model_name_user_defined = "Properly trained pytorch fp32 model"
# MLmodel.load_pytorch(
#     model_path = os.path.join(rel_path, framework_path[framework][model_type]),
#     model_name_user_defined = model_name_user_defined,
#     cfg = os.path.join(rel_path, 'yolov5/models/yolov5s.yaml'),
#     imgsz = 416,
#     data = os.path.join(rel_path, 'dts/data.yaml'),
#     hyp = os.path.join(rel_path, 'yolov5/data/hyps/hyp.scratch.yaml'),
#     single_cls = False,
#     model_class = model_type
# )
# print(MLmodel.statement)
# # print(MLmodel.model)

# framework = 'Pytorch'
# model_type = 'QAT quantized'
# model_name_user_defined = "QAT quantized pytorch qint8 model"
# MLmodel.load_pytorch(
#     model_path = os.path.join(rel_path, framework_path[framework][model_type]),
#     model_name_user_defined = model_name_user_defined,
#     cfg = os.path.join(rel_path, 'yolov5/models/yolov5s.yaml'),
#     imgsz = 416,
#     data = os.path.join(rel_path, 'dts/data.yaml'),
#     hyp = os.path.join(rel_path, 'yolov5/data/hyps/hyp.scratch.yaml'),
#     single_cls = False,
#     model_class = model_type
# )
# print(MLmodel.statement)
# # print(MLmodel.model)

# framework = 'ONNX'
# model_type = 'Proper'
# model_name_user_defined = 'ONNX properly trained fp32 model'
# MLmodel.load_onnx(
#     model_path = os.path.join(rel_path, framework_path[framework][model_type]),
#     model_name_user_defined = model_name_user_defined,
# )
# print(MLmodel.statement)
# # print(MLmodel.model)

# # framework = 'ONNX'
# # model_type = 'QAT quantized'
# # model_name_user_defined = 'ONNX QAT trained qint8 model'
# # MLmodel.load_onnx(
# #     model_path = os.path.join(rel_path, framework_path[framework][model_type]),
# #     model_name_user_defined = model_name_user_defined,
# # )
# # print(MLmodel.statement)
# # print(MLmodel.model)

# framework = 'tf_pb'
# model_type = 'Proper'
# model_name_user_defined = 'Tf_pb fp32 model as tflite converter'
# MLmodel.load_tf_pb_as_tflite_converter(
#     model_path = os.path.join(rel_path, framework_path[framework][model_type]),
#     model_name_user_defined = model_name_user_defined,
# )
# print(MLmodel.statement)
# # print(MLmodel.model)

# # framework = 'tf_pb'
# # model_type = 'QAT quantized'
# # model_name_user_defined = 'Tf_pb qint8 model as tflite converter'
# # MLmodel.load_tf_pb_as_tflite_converter(
# #     model_path = os.path.join(rel_path, framework_path[framework][model_type]),
# #     model_name_user_defined = model_name_user_defined,
# # )
# # print(MLmodel.statement)
# # print(MLmodel.model)

# framework = 'tflite'
# model_type = 'Proper'
# model_name_user_defined = 'Tflite - properly trained fp32 model'
# MLmodel.load_tflite(
#     model_path = os.path.join(rel_path, framework_path[framework][model_type]),
#     model_name_user_defined = model_name_user_defined,
# )
# print(MLmodel.statement)
# # print(MLmodel.model)

# # framework = 'QAT quantized'
# # model_type = 'Proper'
# # model_name_user_defined = 'Tflite - QAT trained qint8 model'
# # MLmodel.load_tflite(
# #     model_path = os.path.join(rel_path, framework_path[framework][model_type]),
# #     model_name_user_defined = model_name_user_defined,
# # )
# # print(MLmodel.statement)
# # print(MLmodel.model)

