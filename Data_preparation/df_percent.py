#Library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PIL import Image
from tqdm import tqdm
import glob
import os

DATA_PATH = '../RAW_DATASET' # Relative path for RAW dataset


imgsize = 416 # delete this line



# class to set the folder path of images, given the data_type that we deal with
class images_folder_path:
    def __init__(self, DATA_PATH,data_type):
        self.DATA_PATH = DATA_PATH
        self.data_type = data_type
    def wider_face_way(self):
        self.folders_path = os.path.join(self.DATA_PATH, self.data_type,'images')


#arrangement : folder => {folder1, fodler2, folder3, folder4,....}
#every folder contains different differnt images
#folder1 = {img1, img2, img3, ....}
# Class to create dataframe of foldername and imagename given the path of that folder
class dframe_imagepaths:
    def __init__(self, folders_path_images):
        self.df = pd.DataFrame(columns = ['folder_name', 'image_name'])
        self.folders_path_images = folders_path_images
    def images_based_df(self):
        '''formation of df based on folder and images stored inside the storage'''
        for folder in tqdm(os.listdir(self.folders_path_images)):
            for image in os.listdir(os.path.join(self.folders_path_images,folder)):
                self.df.loc[len(self.df.index)] = [folder, image]
        # return self.df

# class to set the folder path of images, given the data_type that we deal with
class labels_folder_path:
    def __init__(self, DATA_PATH,data_type):
        self.DATA_PATH = DATA_PATH
        self.data_type = data_type
    def wider_face_way(self):
        self.folders_path = os.path.join(self.DATA_PATH, self.data_type,'labels')


class dframe_labels_train_and_val:
    def __init__(self, folders_path_labels):
        self.df = pd.DataFrame(columns = ['image_name', 'bbox'])
        self.folders_path_labels = folders_path_labels
    def labels_based_df_for_matfiles(self):
        #To create image and label specific dataframe
        #loading the labels from .mat file
        for file in glob.glob(os.path.join(self.folders_path_labels,"*.mat")):
            annots = loadmat(file)
        
        '''formation of df based on label annotaions inside the .mat labels file'''
        for f, folder in enumerate(tqdm(annots['file_list'])):
            for im, image in enumerate(folder[0]):
                boxes_per_image=[]
                for box in annots['face_bbx_list'][f][0][im][0]:
                    boxes_per_image.append(box)
                self.df.loc[len(self.df.index)] = [image[0][0]+'.jpg', boxes_per_image]

#get the merged dataframe of [folder_path, image_name] and [image_name, bbox]
class images_labels_merged_df:
    def __init__(self, df1_sample, df2_sample):
        self.df1_sample = df1_sample
        self.df2_sample = df2_sample
    def merge_df(self):
        self.df_merged = self.df1_sample.merge(self.df2_sample, on='image_name', how = 'inner')

#get the dataframe as the groups of folder_name then as the groups of image_name
#this is the required df for train and validation
class images_labels_bbox_as_grouped:
    def __init__(self, df_merged, DF_FRAC):
        self.groups = df_merged.groupby(['folder_name', 'image_name'])
        self.df_frac = DF_FRAC
    def grouping(self):
        self.df_full = self.groups.first()
        self.df_required_sample = self.df_full.sample(frac= self.df_frac, random_state = 1)





class three_dataframe_preparation:
    '''To get all the 3 required dataframes that too based on percentage of dataset which is required.'''
    def __init__(self, data_type_train, train_frac, data_type_validation, validation_frac, data_type_test, test_frac):
        self.data_type_train = data_type_train
        self.train_frac = train_frac
        self.data_type_validation = data_type_validation
        self.validation_frac = validation_frac
        self.data_type_test = data_type_test
        self.test_frac = test_frac


    def get_train_test_val(self):
        self.train = self.dataframe_preparation(self.data_type_train, self.train_frac)
        self.validation = self.dataframe_preparation(self.data_type_validation, self.validation_frac)
        self.test = self.dataframe_preparation(self.data_type_test, self.test_frac)

    def dataframe_preparation(self, dtype, frac):
        folders_path_images = images_folder_path(DATA_PATH, dtype)
        folders_path_images.wider_face_way()
        df1 = dframe_imagepaths(folders_path_images.folders_path)
        df1.images_based_df()

        if dtype == 'test':
            df_grouped = images_labels_bbox_as_grouped(df1.df, frac)
            df_grouped.grouping()
            return df_grouped.df_required_sample
            
        
        else:
            folders_path_labels = labels_folder_path(DATA_PATH, dtype)
            folders_path_labels.wider_face_way()
            df2 = dframe_labels_train_and_val(folders_path_labels.folders_path)
            df2.labels_based_df_for_matfiles()

            img_lbl_df = images_labels_merged_df(df1.df, df2.df)
            img_lbl_df.merge_df()

            df_grouped = images_labels_bbox_as_grouped(img_lbl_df.df_merged, frac)
            df_grouped.grouping()
            return df_grouped.df_required_sample
