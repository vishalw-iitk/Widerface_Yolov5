'''
Loading the models on different frameworks such as pytorch, ONNX, tf_pb and tflite
'''

''' Importing the libraries '''
import onnx
import torch

class load_the_model:
    
    '''
    To load the models such as :
        Pytorch normal/regular model
        ONNX model
        Tf_pb_keras (loading as a converter)
        Loading the tflite model
    '''

    def __init__(self, device_name):
        '''Device on which the model will be loaded'''
        self.device_name = device_name


    def load_pytorch(self, model_path, model_name_user_defined, model_class = 'any'):
        if model_class == 'Regular':
            '''Loading the Pytorch model using attemot load so fp32 model version/dtype is loaded'''
            from yolov5.models.experimental import attempt_load
            self.model = attempt_load(model_path, map_location=self.device_name)  # load FP32 model
        else:
            '''Trying to load any other pytorch model weights'''
            self.model = torch.load(model_path, map_location = torch.device(self.device_name))
        self.statement = model_name_user_defined + " has been loaded"


    def load_onnx(self, model_path, model_name_user_defined):
        '''Loading the ONNX model using the pytorch model path'''
        self.model = onnx.load(model_path)
        self.statement = model_name_user_defined + " has been loaded"


    def load_tf_pb(self, model_path, model_name_user_defined):
        '''Not loading the tf_pb keras model'''
        import tensorflow as tf
        pass


    def load_tf_pb_as_tflite_converter(self, model_path, model_name_user_defined):
        '''Loading the tf_pb as tflite_converter'''
        import tensorflow as tf
        self.converter =  tf.lite.TFLiteConverter.from_saved_model(model_path)
        self.statement = model_name_user_defined + " has been loaded"


    def load_tflite(self, model_path, model_name_user_defined):
        '''Loading the tflite model'''
        import tensorflow as tf
        with open(model_path, 'rb') as fid:
            self.model = fid.read()
        self.statement = model_name_user_defined + " has been loaded"

# =========================================================================
# In case you want to experiment on loading the model separately


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

