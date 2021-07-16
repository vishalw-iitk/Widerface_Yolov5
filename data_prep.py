#Library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PIL import Image
from tqdm import tqdm
import glob
import os
import sys



DATA_PATH = '../RAW_DATASET'
OUTPUT_PATH = '../ARRANGED_DATASET'

if os.path.exists(OUTPUT_PATH) and os.path.exists(OUTPUT_PATH):
    print("**************************************************************************************")
    print("Dataset already arranged. Make sure it doesn't contain corrupted contents or format")
    print("**************************************************************************************")
    sys.exit()



def our_dataset(data_type='train'):
    
    #To create folder and image specific dataframe
    df1 = pd.DataFrame(columns = ['folder_name', 'image_name'])
    folders_path = os.path.join(DATA_PATH,data_type,'images')
    '''formation of df based on folder and images stored inside the storage'''
    for folder in tqdm(os.listdir(folders_path)):
        for image in os.listdir(os.path.join(folders_path,folder)):
            df1.loc[len(df1.index)] = [folder, image]
    
    if data_type == 'test':
        groups = df1.groupby(['folder_name', 'image_name'])
    else:    
        #To create image and label specific dataframe
        #loading the labels from .mat file
        for file in glob.glob(os.path.join(DATA_PATH, data_type, "labels","*.mat")):
            annots = loadmat(file)
        df2 = pd.DataFrame(columns = ['image_name', 'bbox'])
        '''formation of df based on label annotaions inside the .mat labels file'''
        for f, folder in enumerate(tqdm(annots['file_list'])):
            for im, image in enumerate(folder[0]):
                boxes_per_image=[]
                for box in annots['face_bbx_list'][f][0][im][0]:
                    boxes_per_image.append(box)
                df2.loc[len(df2.index)] = [image[0][0]+'.jpg', boxes_per_image]


        #merging the dataframes to form (folder, image, label) dataframe
        df3 = df1.merge(df2, on='image_name', how = 'inner')

        groups = df3.groupby(['folder_name', 'image_name'])
    
    df_full = groups.first() #as images->bbox pair are all unique hence groups.first() will also contain the complete dataset
    
    return df_full






imgsize = 416
def yolo_type_data(data, data_type = 'train'):
    
    if data_type != 'test':
        if not os.path.exists(os.path.join(OUTPUT_PATH, "images", data_type)):
            os.makedirs(os.path.join(OUTPUT_PATH, "images", data_type))
        if not os.path.exists(os.path.join(OUTPUT_PATH, "labels", data_type)):
            os.makedirs(os.path.join(OUTPUT_PATH, "labels", data_type))

    elif data_type == 'test' and not os.path.exists(os.path.join(OUTPUT_PATH + "_" + "TEST", "images", data_type)):
        os.makedirs(os.path.join(OUTPUT_PATH + "_" + "TEST", "images", data_type))

    
    
    for row in tqdm(data.iterrows()):
        
        folder_name = row[0][0] #from 0th index of multi-index at position 0
        image_name = row[0][1] #from 1st index of multi-index at position 0
        
        
        img = plt.imread(os.path.join(DATA_PATH,data_type,"images",folder_name,image_name)) #loading the image
        img_width = img.shape[1] #image width
        img_height = img.shape[0] #image height
        
        #copying the resized images to the suitable location for yolo
        from_path = os.path.join(DATA_PATH, data_type, "images", folder_name, image_name)
        from_image = Image.open(from_path)
        resized_image = from_image.resize((imgsize,imgsize))
        if data_type == 'test':
            to_path = os.path.join(OUTPUT_PATH + "_" + "TEST", "images", data_type, image_name)
        else:
            to_path = os.path.join(OUTPUT_PATH, "images", data_type, image_name)
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
                os.path.join(OUTPUT_PATH, "labels", data_type, image_name[:-4]+".txt"),
                yolo_data,
                fmt = ["%d", "%f", "%f", "%f", "%f"]
            )
        
        

#To form the dataframe out of the data scattered inside folders    
df_train = our_dataset('train')
df_valid = our_dataset('validation')
df_test = our_dataset('test')


#To form the yolo-type labels and rearrage the data for yolo using the well arranged dataframe(df_full)
yolo_type_data(df_train, 'train')
yolo_type_data(df_valid, 'validation')
yolo_type_data(df_test, 'test')

