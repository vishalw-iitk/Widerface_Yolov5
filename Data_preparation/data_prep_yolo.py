'''
Data preparation steps :
Creation of three different dataframes for train, validation and test dataset repectively
Allowing users to choose the percentage of dataset required for these tasks
Arranging the dataset properly with the help of dataframe in the format required by YOLOV5
FROM FOLDER : RAW_dATASET (as described in README.md)
TO FOLDERS  : ARRANGED_dATASET (train and validation)  |   ARRANGED_DATASET_TEST (test)
'''

'''Library imports'''
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


''' Files and functions imports '''
from dts.Data_preparation.df_percent import three_dataframe_preparation as  all_df
from dts.utils.begin import remove_access_denied_folders 


def yolo_type_data(opt, data, data_type = 'train'):
    '''
    Get the dataset arranged inside the ARRANGED_DATASET and ARRANGED_DATASET_TEST folders
    The path to these created dataset then will be passed to data.yaml present in the Working directory
    Note that for test dataset only images will be there
    ARGS :
        data : 
        data_type : To know at which dataset creation step we are at right now,\
                        train, val, or test
    '''
    
    if data_type != 'test':
        '''Creating the directories for ARRANGED_DATASET'''
        if not os.path.exists(os.path.join(opt.arranged_data_path, "images", data_type)):
            os.makedirs(os.path.join(opt.arranged_data_path, "images", data_type))
        if not os.path.exists(os.path.join(opt.arranged_data_path, "labels", data_type)):
            os.makedirs(os.path.join(opt.arranged_data_path, "labels", data_type))

    elif data_type == 'test' and not os.path.exists(os.path.join(opt.arranged_data_path + "_" + "TEST", "images", data_type)):
        '''Creating the directories for ARRANGED_DATASET_TEST'''
        os.makedirs(os.path.join(opt.arranged_data_path + "_" + "TEST", "images", data_type))

    
    
    for row in tqdm(data.iterrows()):
        
        '''Arranging/Saving the images for train, validation and test dataset'''

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
        
        if data_type != 'test':

            '''Aligning and saving labels for train and validation dataset'''
            
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
                os.path.join(opt.arranged_data_path, "labels", data_type, image_name[:-4]+".txt"),
                yolo_data,
                fmt = ["%d", "%f", "%f", "%f", "%f"]
            )
        
        


def main(opt):
    flag = "run all"
    if os.path.exists(opt.arranged_data_path) and os.path.exists(opt.arranged_data_path + '_TEST'):
        '''If the dataset already exists then check whether again partial dataset is demanded or not.\
            If partial dataset is not demanded again then we already have the dataset arranged and\
                no need to create the dataset again'''
        if opt.partial_dataset == True:
            '''If partial dataset is demanded then remove the existing arranged dataset and \
                created as demanded'''
            remove_access_denied_folders(opt.arranged_data_path)
            remove_access_denied_folders(opt.arranged_data_path + '_TEST')
            print("Removed the existed data as you want partial dataset")
            print("recreating the dataset again.....")
        else:    
            print("**************************************************************************************")
            print("Dataset already arranged")
            print("Make sure it doesn't contain corrupted contents or format although corrupted images will be handled in dataloader step")
            print("Hopefully you have chosed the required percentage of train, validation and test set. (integer percentage only)")
            print("If not then select the partial dataset as required and run the code again. --partial-dataset flag is necessary to \
                recreate the dataset by deleting the present one.")
            print("**************************************************************************************")
            flag = "stop run"
    if flag == "run all":
        '''Creation of the dataset begin here....'''

        '''To form the dataframe out of the data scattered inside folders.
        Getting all the dataset in the form of dataframe called df i.e df.train, df.validation, df.test'''    
        df = all_df('train', opt.percent_traindata/100, 'validation', opt.percent_validationdata/100, 'test', opt.percent_testdata/100, opt.raw_dataset_path)
        df.get_train_test_val()

        '''To form the yolo-type labels for train and validation \
            and rearrage the data-images\
            for yolov5 using the well arranged dataframe(df)'''
        yolo_type_data(opt, df.train, 'train')
        yolo_type_data(opt, df.validation, 'validation')
        yolo_type_data(opt, df.test, 'test')

def parse_opt(known = False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dataset-path', type=str, default = '../RAW_DATASET', help='Path of the raw dataset which was just arranged from the downloaded dataset')

    parser.add_argument('--partial-dataset', action='store_true', help='willing to select custom percentage of dataset')
    parser.add_argument('--percent_traindata', type=int, help=' percent_of_the_train_data_required')
    parser.add_argument('--percent_validationdata', type=int, help=' percent_of_the_validation_data_required')
    parser.add_argument('--percent_testdata', type=int, help=' percent_of_the_test_data_required')

    parser.add_argument('--arranged-data-path', type=str, default = '../ARRANGED_DATASET', help='Path of the arranged dataset')
    
    parser.add_argument('--img-size', type=int, default = 416, help = 'Image size suitable for feeding to the model')

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
