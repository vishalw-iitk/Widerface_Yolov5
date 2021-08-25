import argparse
import cv2
import glob
from pathlib import Path
import os
import torch
import numpy as np
import sys
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
    
def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def gt_overlay(gt_path, pred_path,source,out_path):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    p = str(Path(source).absolute())
    if os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        files = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
    n = len(images)
    i = 1
    for image in images:
        im0 = cv2.imread(image)
        filename = os.path.split(image)[1].split('.')[0]
        print("processing {0}/{1}, file {2}".format(i,n,os.path.split(image)[1]),end=' ')
        gt_labels = os.path.join(gt_path,filename+'.txt')
        pred_labels = os.path.join(pred_path,filename+'.txt')
        with open(gt_labels,"r") as gt_file:
            for line in gt_file:
                line = line.rstrip()
                xywh = [float(x) for x in line.split(" ")[1:]]
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] #reversing normalization operation performed in detect.py
                xyxy = (xywh2xyxy(torch.tensor(xywh).view(1, 4))*gn).view(-1)
                plot_one_box(xyxy, im0 , color=(0,255,0), line_thickness=3)
        #handling absence of validation_labels.txt for those images with no objects
        if not os.path.exists(pred_labels):
            print("skipping as no objects detected")
            i = i+1
            continue
        with open(pred_labels,"r") as pred_file:
            for line in pred_file:
                line = line.rstrip()
                xywh = [float(x) for x in line.split(" ")[1:]]
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                xyxy = (xywh2xyxy(torch.tensor(xywh).view(1, 4))*gn).view(-1)
                plot_one_box(xyxy, im0 , color=(0,0,255), line_thickness=3)
        cv2.imwrite(os.path.join(out_path,filename+'.jpg'), im0)
        print("Done")
        i = i+1
    

def main(opt):
    gt_overlay(**vars(opt))
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, help='gt_labels txt')
    parser.add_argument('--source', type=str, default='data/images', help='input image')
    parser.add_argument('--pred_path', type=str, help='predicted labels txt')
    parser.add_argument('--out_path', type=str, help='output images path')
    opt = parser.parse_args()
    return opt
    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)