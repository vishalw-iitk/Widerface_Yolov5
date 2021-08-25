'''
!python /content/drive/MyDrive/Wider_Face/yolov5/Segregate.py --validationImages /content/drive/MyDrive/Wider_Face/yolov5/data/wider_face_samples/images --groundTruth_TxtPath /content/drive/MyDrive/Wider_Face/yolov5/data/wider_face_samples/ground_truths --predicted_TxtPath /content/drive/MyDrive/Wider_Face/yolov5/runs/detect/exp6/labels --thrs_FilePath /content/drive/MyDrive/Wider_Face/yolov5/runs/thr.npy --good_predPath /content/drive/MyDrive/Wider_Face/yolov5/runs/good --bad_predPath /content/drive/MyDrive/Wider_Face/yolov5/runs/bad

'''

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as st
import os,shutil
import argparse


# get BBoxes for each sample 
def get_Bboxes(Txtpath, img_width, img_height):
  """creates Bounding Box coordinates. eg: #[x_cen, y_cen, w, d]

    Arguments:
      Txtpath   : output txt file path in yolo format.
      img_width : Width of the Image.
      img_height: Height of the Image
    
    Returns: gives Bounding Box coordinates.
  """
  boxes=[]
  txt_file = open(Txtpath, 'r')
  Lines = txt_file.readlines()
  for i in range(len(Lines)):
      line=Lines[i]
      data=line.split(" ")
      
      #[x_cen, y_cen, w, d]
      x_cen=float(data[1])*img_width
      y_cen=float(data[2])*img_height
      width_box=float(data[3])*img_width
      height_box=float(data[4])*img_height

      Bbox=[int(x_cen),int(y_cen),int(width_box),int(height_box)]
      boxes.append(Bbox)
  return boxes

def MeshGrid(x,y):
  """

    Specify a meshgrid which will use 100 points interpolation on each axis. (e.g. mgrid(xmin:xmax:100j))
    Define the borders

  """
  deltaX = (max(x) - min(x))/50
  deltaY = (max(y) - min(y))/50
  xmin = min(x) - deltaX
  xmax = max(x) + deltaX
  ymin = min(y) - deltaY
  ymax = max(y) + deltaY
  return np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

def fit_gaussian_kernel(xx,yy,x,y):
  """

    Fit a gaussian kernel using the scipy’s gaussian_kde method
    Returns: MeshGrid

  """
  positions = np.vstack([xx.ravel(), yy.ravel()])
  values = np.vstack([x, y])
  kernel = st.gaussian_kde(values)
  return np.reshape(kernel(positions).T, xx.shape)

def kl_divergence(p, q):
  """
  
  Calculate the KL divergence of two probability density distributions.
  Make sure that we don’t include any probabilities equal to 0 because the log of 0 is negative infinity.
  

  """
  return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def main(opt):
  """
    Creates density threshold value.

    Arguments:
      validationImages    : Directory path of validation Images.
      groundTruth_TxtPath : Directory path of Ground truths Txt files.
      predicted_TxtPath   : Directory path of predicted Txt files
      thrs_FilePath       : file path of saved result.
      good_predPath       : Directory path of Good predictions.
      bad_predPath        : Directory path of Bad predictions.


  """

  thes_value = np.load(opt.thrs_FilePath)
  good_imagesPath = os.path.join(opt.good_predPath, "images")
  print(good_imagesPath)
  if os.path.exists(good_imagesPath):
    shutil.rmtree(good_imagesPath)
  os.makedirs(good_imagesPath)
  bad_imagesPath = os.path.join(opt.bad_predPath, "images")
  if os.path.exists(bad_imagesPath):
    shutil.rmtree(bad_imagesPath)
  os.makedirs(bad_imagesPath)
  good_labelsPath = os.path.join(opt.good_predPath, "labels")
  if os.path.exists(good_labelsPath):
    shutil.rmtree(good_labelsPath)
  os.makedirs(good_labelsPath)
  bad_labelsPath = os.path.join(opt.bad_predPath, "labels")
  if os.path.exists(bad_labelsPath):
    shutil.rmtree(bad_labelsPath)
  os.makedirs(bad_labelsPath)
  
  batchImages = os.listdir(opt.validationImages)  
  for Img in batchImages:
    img_path = os.path.join(opt.validationImages, Img)
    im = Image.open(img_path)
    img_width, img_height = im.size
    txt_name = Img.split(".")[0] + ".txt"
    GroundTruth_TxtPath = os.path.join(opt.groundTruth_TxtPath, txt_name)
    Predicted_TxtPath = os.path.join(opt.predicted_TxtPath, txt_name)


    if not (os.path.exists(GroundTruth_TxtPath) and os.path.exists(Predicted_TxtPath)):
      shutil.copy(img_path, bad_imagesPath)
      if os.path.exists(Predicted_TxtPath):
        shutil.copy(Predicted_TxtPath, bad_labelsPath)
      continue


    # Get Bboxes coordinates: list of [x_cen, y_cen, w, d].
    Grd_Bboxes = get_Bboxes(GroundTruth_TxtPath, img_width, img_height) # Ground Truth Boxes coordinates.
    Prd_Bboxes = get_Bboxes(Predicted_TxtPath, img_width, img_height)   # Predicted Boxes coordinates.
    # print(Prd_Bboxes)

    # List of x_cen and y_cen for ground_truths
    grd_x = np.array([box[0] for box in Grd_Bboxes])
    grd_y = np.array([box[1] for box in Grd_Bboxes])

    if grd_x.shape[0] == 2:
      continue

    # List of x_cen and y_cen for predictions
    prd_x = np.array([box[0] for box in Prd_Bboxes])
    prd_y = np.array([box[1] for box in Prd_Bboxes])

    ####
    # get grids
    xx_grd, yy_grd = MeshGrid(grd_x,grd_y)
    xx_prd, yy_prd = MeshGrid(prd_x,prd_y)

    ####
    # get Gaussian
    f_grd = fit_gaussian_kernel(xx_grd,yy_grd,grd_x,grd_y)
    f_prd = fit_gaussian_kernel(xx_prd,yy_prd,prd_x,prd_y)


    kl_div_value = abs(kl_divergence(f_grd,f_prd))

    if kl_div_value < thes_value:
      shutil.copy(img_path, good_imagesPath)
      shutil.copy(Predicted_TxtPath, good_labelsPath)
    else:
      shutil.copy(img_path, bad_imagesPath)
      shutil.copy(Predicted_TxtPath, bad_labelsPath)

    print(Img)
    


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validationImages', type=str, help='validationImages_path')
    parser.add_argument('--groundTruth_TxtPath', type=str, help='groundTruth_TxtPath')
    parser.add_argument('--predicted_TxtPath', type=str, help='predicted_TxtPath')
    parser.add_argument('--thrs_FilePath', type=str, help='threshold_saved_file_path')
    parser.add_argument('--good_predPath', type=str, help='good predictions images and labels path')
    parser.add_argument('--bad_predPath', type=str, help='bad predictions images and labels path')
  
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
