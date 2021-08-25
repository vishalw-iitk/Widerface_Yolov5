# import argparse
# from dts.model_paths import running_model_dictionary
# from dts.model_paths import pre_trained_model_dictionary
# from dts.Model_conversion.framework_paths import frameworks

# import os

def update_to_running_paths_with_pretrianed(running_model_paths, pre_trained_model_paths):
    running_model_paths['Regular']['Pytorch']['fp32'] = pre_trained_model_paths['Regular']['Pytorch']['fp32']
    running_model_paths['Regular']['Pytorch']['fp16'] = pre_trained_model_paths['Regular']['Pytorch']['fp16']
    running_model_paths['Regular']['Tflite']['fp32'] = pre_trained_model_paths['Regular']['Tflite']['fp32']
    running_model_paths['Quantization']['Pytorch']['QAT'] = pre_trained_model_paths['Quantization']['Pytorch']['QAT']
    running_model_paths['Pruning']['Pytorch']['P1'] = pre_trained_model_paths['Pruning']['Pytorch']['P1']

    return running_model_paths

# def update_running_model_paths(clone_updated_yolov5, running_model_paths, framework_path, pre_trained_model_paths):
#     if clone_updated_yolov5 == False:
#         running_model_paths['Regular']['Pytorch']['fp32'] = pre_trained_model_paths['Regular']['Pytorch']['fp32']

#         frmw = os.path.join('..', 'Pre_trained_model', 'Regular', 'Pytorch')
#         fp = 'fp32'
#         m_type = 'Regular_fp32'
#         framework_path['Pytorch'][m_type] = os.path.join(frmw, fp, 'best.pt')
#         framework_path['ONNX'][m_type] = os.path.join(frmw, fp, 'best.onnx')
#         framework_path['tf_pb'][m_type] = os.path.join(frmw, fp, 'best.pb')
#         framework_path['tflite'][m_type] = os.path.join(frmw, fp, 'best.tflite')

#     running_model_paths['Regular']['Pytorch']['fp16'] = pre_trained_model_paths['Regular']['Pytorch']['fp16']
#     running_model_paths['Quantization']['Pytorch']['QAT'] = pre_trained_model_paths['Quantization']['Pytorch']['QAT']
#     running_model_paths['Pruning']['Pytorch']['P1'] = pre_trained_model_paths['Pruning']['Pytorch']['P1']

#     frmw = os.path.join('..', 'Pre_trained_model', 'Regular', 'Pytorch')
#     fp = 'fp16'
#     m_type = 'Regular_fp16'
#     framework_path['Pytorch'][m_type] = os.path.join(frmw, fp, 'best.pt')
#     framework_path['ONNX'][m_type] = os.path.join(frmw, fp, 'best.onnx')
#     framework_path['tf_pb'][m_type] = os.path.join(frmw, fp, 'best.pb')
#     framework_path['tflite'][m_type] = os.path.join(frmw, fp, 'best.tflite')

#     print(framework_path)

#     return running_model_paths, framework_path

# def update_with_single_path(clone_updated_yolov5, running_model_paths):
#     if clone_updated_yolov5 == True:
#         for f in ['Pytorch', 'Tflite']:
#             running_model_paths['Regular'][f]['model'] = running_model_paths['Regular'][f]['fp16']
#             del running_model_paths['Regular'][f]['fp16']
#     else:
#         for f in ['Pytorch', 'Tflite']:
#             running_model_paths['Regular'][f]['model'] = running_model_paths['Regular'][f]['fp32']
#             del running_model_paths['Regular'][f]['fp16']
#             del running_model_paths['Regular'][f]['fp32']
#     return running_model_paths
# def main(opt):
#     pre_trained_model_paths = pre_trained_model_dictionary()
#     try:
#         running_model_paths, framework_path = update_running_model_paths(opt.clone_updated_yolov5, opt.running_model_paths, opt.framework_path, pre_trained_model_paths)
#     except:
#         running_model_paths = running_model_dictionary()
#         framework_path = frameworks(running_model_paths)
#         running_model_paths, framework_path = update_running_model_paths(opt.clone_updated_yolov5, running_model_paths, framework_path, pre_trained_model_paths)
#     return running_model_paths, framework_path


# def parse_opt(known=False):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--clone-updated-yolov5', action='store_true', help='clone the updated yolov5 repository. This may not work if updates in the original yolv5 repo become incompatible with our setup')
#     opt = parser.parse_known_args()[0] if known else parser.parse_args()
#     return opt

# def run(**kwargs):
#     # Usage: import train; train.run(imgsz=320, weights='yolov5m.pt')
#     opt = parse_opt(True)
#     for k, v in kwargs.items():
#         setattr(opt, k, v)
#     running_model_paths, framework_path = main(opt)
#     return running_model_paths, framework_path


# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)

