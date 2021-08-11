from dts.utils.load_the_models import load_the_model
import os
import torch

def fp_type_conversion(type_conversion, device, running_model_paths):
    # rel_path = '../'
    MLmodel = load_the_model('cpu')
    framework = 'Pytorch'
    model_type = 'Regular'

    if type_conversion == 'fp16_to_fp32':
        fp = 'fp16'
    else:
        fp = 'fp32'
    if not os.path.exists(running_model_paths[model_type][framework][fp].replace('/best.pt', '')):
        os.mkdir(running_model_paths[model_type][framework][fp].replace('/best.pt', ''))        
    os.system('mv '+running_model_paths[model_type][framework][fp].replace(fp+'/', '') +' '+ running_model_paths[model_type][framework][fp])
    model_path = os.path.join(running_model_paths[model_type][framework][fp])

    model_name_user_defined = "Regular trained pytorch model"
    MLmodel.load_pytorch(
        model_path = model_path,
        model_name_user_defined = model_name_user_defined,
        cfg = os.path.join('../yolov5/models/yolov5s.yaml'),
        imgsz = 416,
        data = os.path.join('data.yaml'),
        hyp = os.path.join('../yolov5/data/hyps/hyp.scratch.yaml'),
        single_cls = False,
        model_class = model_type
    )
    print(MLmodel.statement)
    # print(MLmodel.model)

    if type_conversion == 'fp16_to_fp32':
        fp = 'fp32'
        ckpt = {'model' : MLmodel.model.float()}
    else:
        fp = 'fp16'
        ckpt = {'model' : MLmodel.model.half()}
    
    if not os.path.exists(running_model_paths[model_type][framework][fp].replace('/best.pt', '')):
        os.mkdir(running_model_paths[model_type][framework][fp].replace('/best.pt', ''))    

    
    torch.save(ckpt, running_model_paths[model_type][framework][fp])
