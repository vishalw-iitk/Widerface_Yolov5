'''
Model conversion from Pytorch fp32 to fp16 if we chose to work on ultralitics version that we have in our repository
Model conversion from Pytorch fp16 to fp32 if we chose to work on cloned ultralitics version
'''

''' Importing the libraries '''
import os
import torch
import argparse
import sys

''' Files imports '''
# sys.path.append('../..')
from dts.utils.load_the_models import load_the_model


def fp_type_conversion(type_conversion, img_size, weights):
    '''
    Making the directories for fp32 and fp16
    Loading the Pytorch model
    and converiting into other dtype
    Finally storing the model
    '''
    MLmodel = load_the_model('cpu')
    if type_conversion == 'fp32_to_fp16':
        fp = 'fp32'
    else:
        fp = 'fp16'

    
    if '/best.pt' in weights:
        if not os.path.exists(weights.replace('/best.pt', '')):
            os.mkdir(weights.replace('/best.pt', ''))        
        os.system('mv '+weights.replace(fp+'/', '') +' '+ weights)
    elif r'\best.pt' in weights:
        if not os.path.exists(weights.replace(r'\best.pt', '')):
            os.mkdir(weights.replace(r'\best.pt', ''))        
        os.system('mv '+weights.replace(fp+r'\best_check_string'[0], '') +' '+ weights)
    model_path = os.path.join(weights)


    model_name_user_defined = "Regular trained pytorch model"
    MLmodel.load_pytorch(
        model_path = model_path,
        model_name_user_defined = model_name_user_defined,
        model_class = 'Regular'
    )

    if type_conversion == 'fp16_to_fp32':
        ckpt = {'model' : MLmodel.model.float()}
        fp = 'fp32'
        weights = weights.replace('fp16', 'fp32')
    else:
        ckpt = {'model' : MLmodel.model.half()}
        fp = 'fp16'
        weights = weights.replace('fp32', 'fp16')
    
    if '/best.pt' in weights:
        if not os.path.exists(weights.replace('/best.pt', '')):
            os.mkdir(weights.replace('/best.pt', ''))    
    elif r'\best.pt' in weights:
        if not os.path.exists(weights.replace(r'\best.pt', '')):
            os.mkdir(weights.replace(r'\best.pt', ''))    


    # This saving does not harm the pre-trained fp16 weights
    torch.save(ckpt, weights)

def main(opt):
    fp_type_conversion(opt.conversion_type, opt.img_size, opt.weights)

def parse_opt(known=False):
    '''While running explicitly : model will be replaced on the spot as per type casting\
    and uncomment the sys.path.append at the top'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--conversion-type', type=str, default='fp32_to_fp16', help='choose fp16_to_fp32 or fp32_to_fp16')
    parser.add_argument('--img-size', type=int, default = 416, help = 'Image size suitable for feeding to the model and train, val image size (pixels)')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
