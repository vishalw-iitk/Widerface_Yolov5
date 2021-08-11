from dts.Data_preparation.df_percent import three_dataframe_preparation as  all_df
from dts.utils.begin import remove_access_denied_folders
# from df_percent import images_folder_path
# from df_percent import dframe_imagepaths
# from df_percent import labels_folder_path
# from df_percent import dframe_labels_train_and_val
# from df_percent import images_labels_merged_df
# from df_percent import images_labels_bbox_as_grouped 

#Library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PIL import Image
from tqdm import tqdm
# import glob
import os
import sys
import argparse

# DATA_PATH = '../RAW_DATASET'
# OUTPUT_PATH = '../ARRANGED_DATASET'

# DF_FRAC_TRAIN = 0.50
# DF_FRAC_VALIDATION = 0.50
# DF_FRAC_TEST = 0.50


def yolo_type_data(opt, data, data_type = 'train'):
    
    if data_type != 'test':
        if not os.path.exists(os.path.join(opt.arranged_data_path, "images", data_type)):
            os.makedirs(os.path.join(opt.arranged_data_path, "images", data_type))
        if not os.path.exists(os.path.join(opt.arranged_data_path, "labels", data_type)):
            os.makedirs(os.path.join(opt.arranged_data_path, "labels", data_type))

    elif data_type == 'test' and not os.path.exists(os.path.join(opt.arranged_data_path + "_" + "TEST", "images", data_type)):
        os.makedirs(os.path.join(opt.arranged_data_path + "_" + "TEST", "images", data_type))

    
    
    for row in tqdm(data.iterrows()):
        
        folder_name = row[0][0] #from 0th index of multi-index at position 0
        image_name = row[0][1] #from 1st index of multi-index at position 0
        
        
        img = plt.imread(os.path.join(opt.raw_dataset_path, data_type,"images",folder_name,image_name)) #loading the image
        img_width = img.shape[1] #image width
        img_height = img.shape[0] #image height
        
        #copying the resized images to the suitable location for yolo
        from_path = os.path.join(opt.raw_dataset_path, data_type, "images", folder_name, image_name)
        from_image = Image.open(from_path)
        resized_image = from_image.resize((opt.img_size,opt.img_size))
        if data_type == 'test':
            to_path = os.path.join(opt.arranged_data_path + "_" + "TEST", "images", data_type, image_name)
        else:
            to_path = os.path.join(opt.arranged_data_path, "images", data_type, image_name)
        resized_image.save(to_path)
        
#         shutil.copyfile(
#             os.path.join(DATA_PATH, data_type, "images", folder_name, image_name),
#             os.path.join(OUTPUT_PATH, "images", data_type, image_name)
#         )
        
        if data_type != 'test':
            
            bounding_boxes = row[1][0] #(0th column) at position 1
            yolo_data = [] # To save normalized yolo type label for an image for all the bounding boxes
            for bbox in bounding_boxes:
                x_topleft = bbox[0]
                y_topleft = bbox[1]
                bbox_width = bbox[2]
                bbox_height = bbox[3]

                #normalizing
                x_center_norm = (x_topleft + bbox_width/2) / img_width
                y_center_norm = (y_topleft + bbox_height/2) / img_height
                bbox_width_norm = bbox_width/img_width
                bbox_height_norm = bbox_height/img_height

                yolo_data.append([0, x_center_norm, y_center_norm, bbox_width_norm, bbox_height_norm])
            yolo_data = np.array(yolo_data) #array conversion

            #saving the label for an image at the suitable location for yolo
            np.savetxt(
                #os.path.join(OUTPUT_PATH, f"labels/{data_type}/{image_name[:-4]}.txt"),
                os.path.join(opt.arranged_data_path, "labels", data_type, image_name[:-4]+".txt"),
                yolo_data,
                fmt = ["%d", "%f", "%f", "%f", "%f"]
            )
        
        


def main(opt):
    flag = "run all"
    if os.path.exists(opt.arranged_data_path) and os.path.exists(opt.arranged_data_path + '_TEST'):
        if opt.partial_dataset == True:
            remove_access_denied_folders(opt.arranged_data_path)
            remove_access_denied_folders(opt.arranged_data_path + '_TEST')
            print("Removed the existed data as you want partial dataset")
            # os.system('rmdir '+opt.arranged_data_path)
            # os.system('rmdir '+opt.arranged_data_path + '_TEST')
            print("recreating the dataset again.....")
        else:    
            print("**************************************************************************************")
            print("Dataset already arranged")
            print("Make sure it doesn't contain corrupted contents or format")
            print("Hopefully you have chosed the required percentage of train, validation and test set.")
            print("If not then delete the arranged dataset folder and run the code again")
            print("**************************************************************************************")
            flag = "stop run"
            # sys.exit()
    if flag == "run all":
        #To form the dataframe out of the data scattered inside folders    
        df = all_df('train', opt.percent_traindata/100, 'validation', opt.percent_validationdata/100, 'test', opt.percent_testdata/100)
        df.get_train_test_val()

        #To form the yolo-type labels and rearrage the data for yolo using the well arranged dataframe(df_full)
        yolo_type_data(opt, df.train, 'train')
        yolo_type_data(opt, df.validation, 'validation')
        yolo_type_data(opt, df.test, 'test')

def parse_opt(known = False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dataset_path', type=str, default = '../RAW_DATASET', help='Path of the raw dataset which was just arranged from the downloaded dataset')

    parser.add_argument('--partial-dataset', action='store_true', help='willing to select custom percentage of dataset')
    parser.add_argument('--percent_traindata', type=int, help=' percent_of_the_train_data_required')
    parser.add_argument('--percent_validationdata', type=int, help=' percent_of_the_validation_data_required')
    parser.add_argument('--percent_testdata', type=int, help=' percent_of_the_test_data_required')

    parser.add_argument('--arranged_data_path', type=str, default = '../ARRANGED_DATASET', help='Path of the arranged dataset')
    
    parser.add_argument('--img-size', type=int, default = 416, help = 'Image size suitable for feeding to the model')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def run(**kwargs):
    # Usage: import train; train.run(imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
