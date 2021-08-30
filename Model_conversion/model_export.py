import sys

from cv2 import data


from dts.utils.load_the_models import load_the_model
from dts.model_paths import model_defined_names, running_model_dictionary
from dts.model_paths import pre_trained_model_dictionary
from yolov5 import export
from pathlib import Path

from dts.Model_conversion import onnx_export
from yolov5.utils.datasets import LoadImages


from dts.model_paths import frameworks
from onnx_tf.backend import prepare
import os
import argparse
import numpy as np
import tensorflow as tf



 
class transform_the_model:
    def __init__(self, framework_path, model_names):
        self.MLmodel = load_the_model('cpu')
        self.framework_path = framework_path
        self.model_names = model_names

    def pytorch_to_onnx(self, framework_from, model_type, pytorch_name_user_defined, onnx_name_user_defined,  framework_to):
        if model_type == self.model_names['Regular']['Pytorch']['fp32']:
            export.run(
                weights = os.path.join(self.framework_path[framework_from][model_type]),
                img_size = (416, 416),
                include = ['onnx']
                # half = half
            )
            self.statement = pytorch_name_user_defined + " has been transformed to " + onnx_name_user_defined
        elif model_type == 'Quantized':
            weights = os.path.join(self.framework_path[framework_from][model_type])
            model, img, file = onnx_export.get_modelutils(weights)
            print("file************", file)
            onnx_export.export_onnx(model, img, file)
            self.statement = pytorch_name_user_defined + " has been transformed to " + onnx_name_user_defined
        else:
            self.statement = "pytorch to onnx already available"

    def onnx_to_tfpb(self, framework_from, model_type, onnx_name_user_defined, tf_pb_name_user_defined, framework_to):
        if model_type == self.model_names['Regular']['Pytorch']['fp32']:            
            tf_pb_model_storage_path = os.path.join(self.framework_path[framework_to][model_type])
            
            self.MLmodel.load_onnx(
                model_path = os.path.join(self.framework_path[framework_from][model_type]),
                model_name_user_defined = onnx_name_user_defined
            )
            print("********************************")
            print(self.MLmodel.statement)
            print("********************************")

            tf_rep = prepare(self.MLmodel.model)
            print("tf pb storage", tf_pb_model_storage_path)
            tf_rep.export_graph(tf_pb_model_storage_path)
            self.statement = onnx_name_user_defined + " has been transformed to " + tf_pb_name_user_defined
        else:
            self.statement = "onnx to tfpb graph already available"

    
        
    #load tfpb as tflite converter
    def tfpbconverter_to_tflite(self, framework_from, model_type, tf_pb_name_user_defined, tflite_name_user_defined, framework_to, representative_dataset_gen = None):
        tflite_model_storage_path = os.path.join(self.framework_path[framework_to][model_type])

        # tflite converters without optimizers
        # here self.model is tflite converter create from tfpb model

        self.MLmodel.load_tf_pb_as_tflite_converter(
            model_path = os.path.join(self.framework_path[framework_from][model_type]),
            model_name_user_defined = tf_pb_name_user_defined
        )
        print("********************************")
        print(self.MLmodel.statement)
        print("********************************")
        if model_type == self.model_names['Regular']['Pytorch']['fp32']:
            converter = self.MLmodel.converter
        elif model_type == self.model_names['Quantization']['Tflite']['fp16']:
            converter = self.MLmodel.converter
            converter.optimizations = [tf.lite.Optimize.DEFAULT]       
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            converter.target_spec.supported_types = [tf.float16]
            converter.allow_custom_ops = False
            converter.experimental_new_converter = True
            converter.experimental_new_quantizer = False
        elif model_type == self.model_names['Quantization']['Tflite']['int8']:
            #int8 conversion using representative dataset
            converter = self.MLmodel.converter
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset_gen
            converter.allow_custom_ops = False
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.SELECT_TF_OPS]
            converter.inference_input_type = tf.uint8  # or tf.int8
            converter.inference_output_type = tf.uint8  # or tf.int8
            converter.experimental_new_converter = True
            converter.experimental_new_quantizer = False

        tflite_model = converter.convert()
        if '/best.tflite' in tflite_model_storage_path:
            tflite_model_storage_path = tflite_model_storage_path.replace('/best.tflite', '')
        elif r'\best.tflite' in tflite_model_storage_path:
            tflite_model_storage_path = tflite_model_storage_path.replace(r'\best.tflite', '')
        
        if not os.path.exists(tflite_model_storage_path):
            os.makedirs(tflite_model_storage_path)
        
        tflite_model_storage_path = Path(tflite_model_storage_path)
        tflite_model_storage_path = tflite_model_storage_path / 'best.tflite'

        open(tflite_model_storage_path, "wb").write(tflite_model)
        self.statement = tf_pb_name_user_defined + " has been transformed to " + tflite_name_user_defined

        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("input_details of "+model_type)
        print(input_details)
        print("output_details of "+model_type)
        print(output_details)

def main(opt):
    model_type = opt.model_type_for_export

    transform = transform_the_model(opt.framework_path, opt.model_names)
    transform.pytorch_to_onnx(framework_from = 'Pytorch', model_type = model_type, \
        pytorch_name_user_defined = 'Pytorch '+model_type+ ' model', \
        onnx_name_user_defined = 'Onnx '+model_type, framework_to = 'ONNX')

    print("********************************")
    print(transform.statement)
    print("********************************")


    transform.onnx_to_tfpb(framework_from = 'ONNX', model_type = model_type, \
        onnx_name_user_defined = 'Onnx '+model_type, \
            tf_pb_name_user_defined = 'tf_pb '+model_type, framework_to = 'tf_pb')
    print("********************************")
    print(transform.statement)
    print("********************************")


    def representative_dataset_gen():
        # Representative dataset for use with converter.representative_dataset(int8 quantization)
        n = 0
        for path, img, im0s, vid_cap in dataset:
            # Get sample input data as a numpy array in a method of your choosing.
            n += 1
            inp = np.transpose(img, [0, 1, 2])
            print(inp.shape)
            inp = np.expand_dims(inp, axis=0).astype(np.float32)
            inp /= 255.0
            yield [inp]
            if n >= ncalib:
                break
    
    def repr_im():
        try:
            return opt.ncalib, LoadImages(opt.repr_images, img_size=opt.imgtf), representative_dataset_gen
        except:
            return None, None, None
    ncalib, dataset, representative_dataset_gen = repr_im()
    

    transform.tfpbconverter_to_tflite(framework_from = 'tf_pb', model_type = model_type, \
        tf_pb_name_user_defined = 'tf_pb '+model_type, \
            tflite_name_user_defined = 'tflite '+model_type, framework_to = 'tflite', \
                representative_dataset_gen = representative_dataset_gen)
    print("********************************")
    print(transform.statement)
    print("********************************")

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type_for_export', default= 'Regular_fp32', help='')
    parser.add_argument('--skip-training', action='store_true', help='skip the time taking regular training')
    parser.add_argument('--repr-images', type=str, default='../ARRANGED_DATASET/images/validation/', help='path of representative dataset')
    parser.add_argument('--imgtf', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--ncalib', type=int, default=100, help='number of calibration images')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    running_model_paths = running_model_dictionary()
    pre_trained_model_paths = pre_trained_model_dictionary()
    opt.framework_path = frameworks(opt.skip_training, running_model_paths, pre_trained_model_paths)

    opt.model_names = model_defined_names()
    return opt

def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)