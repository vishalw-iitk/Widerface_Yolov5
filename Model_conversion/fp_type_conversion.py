'''
Model conversion from Pytorch fp32 to fp16 if we chose to work on ultralitics version that we have in our repository
Model conversion from Pytorch fp16 to fp32 if we chose to work on cloned ultralitics version
'''

''' Importing the libraries '''
from dts.model_paths import running_model_dictionary
import os
import torch
import argparse

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

def main(opt):
    if opt.clone_updated_yolov5 == True:
        opt.conversion_type = 'fp16_to_fp32'
    else:
        opt.conversion_type = 'fp32_to_fp16'
    
    fp_type_conversion(opt.conversion_type, opt.device, opt.cfg, opt.data, opt.hyp, \
        opt.single_cls, opt.img_size, opt.running_model_paths)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--clone-updated-yolov5', action='store_true', help='clone the updated yolov5 repository. This may not work if updates in the original yolv5 repo become incompatible with our setup')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--cfg', type=str, default='../yolov5/models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='../yolov5/data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--img-size', type=int, default = 416, help = 'Image size suitable for feeding to the model and train, val image size (pixels)')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    opt.running_model_paths = running_model_dictionary()
    return opt

def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)