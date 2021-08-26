'''
Model conversion from Pytorch fp32 to fp16 if we chose to work on ultralitics version that we have in our repository
Model conversion from Pytorch fp16 to fp32 if we chose to work on cloned ultralitics version
'''

''' Importing the libraries '''
import os
import torch

''' Files imports '''
from dts.utils.load_the_models import load_the_model


def fp_type_conversion(type_conversion, device, cfg, data, hyp, single_cls, img_size, running_model_paths):
    '''
    Making the directories for fp32 and fp16
    Loading the Pytorch model
    and converiting into other dtype
    Finally storing the model
    '''
    MLmodel = load_the_model(device)
    framework = 'Pytorch'
    model_type = 'Regular'

    if type_conversion == 'fp16_to_fp32':
        fp = 'fp16'
    else:
        fp = 'fp32'
    
    if '/best.pt' in running_model_paths[model_type][framework][fp]:
        if not os.path.exists(running_model_paths[model_type][framework][fp].replace('/best.pt', '')):
            os.mkdir(running_model_paths[model_type][framework][fp].replace('/best.pt', ''))        
        os.system('mv '+running_model_paths[model_type][framework][fp].replace(fp+'/', '') +' '+ running_model_paths[model_type][framework][fp])
    elif r'\best.pt' in running_model_paths[model_type][framework][fp]:
        if not os.path.exists(running_model_paths[model_type][framework][fp].replace(r'\best.pt', '')):
            os.mkdir(running_model_paths[model_type][framework][fp].replace(r'\best.pt', ''))        
        os.system('mv '+running_model_paths[model_type][framework][fp].replace(fp+r'\best_check_string'[0], '') +' '+ running_model_paths[model_type][framework][fp])
    model_path = os.path.join(running_model_paths[model_type][framework][fp])


    model_name_user_defined = "Regular trained pytorch model"
    MLmodel.load_pytorch(
        model_path = model_path,
        model_name_user_defined = model_name_user_defined,
        cfg = cfg,
        imgsz = img_size,
        data = data,
        hyp = hyp,
        single_cls = single_cls,
        model_class = model_type
    )

    if type_conversion == 'fp16_to_fp32':
        fp = 'fp32'
        ckpt = {'model' : MLmodel.model.float()}
    else:
        fp = 'fp16'
        ckpt = {'model' : MLmodel.model.half()}
    
    if '/best.pt' in running_model_paths[model_type][framework][fp]:
        if not os.path.exists(running_model_paths[model_type][framework][fp].replace('/best.pt', '')):
            os.mkdir(running_model_paths[model_type][framework][fp].replace('/best.pt', ''))    
    elif r'\best.pt' in running_model_paths[model_type][framework][fp]:
        if not os.path.exists(running_model_paths[model_type][framework][fp].replace(r'\best.pt', '')):
            os.mkdir(running_model_paths[model_type][framework][fp].replace(r'\best.pt', ''))    


    # This saving does not harm the pre-trained fp16 weights
    torch.save(ckpt, running_model_paths[model_type][framework][fp])
