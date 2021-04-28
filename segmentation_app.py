import numpy as np 
import argparse
import imutils
import time 
import cv2 
import os 
import matplotlib.pyplot as plt 
import os 
import pathlib 
import streamlit as st

from PIL import Image

def load_image(image_file) :
    img = Image.open(image_file) 
    return img 

def save_uploaded_file(directory, img) :

    if not os.path.exists(directory) :
        os.makedirs(directory)

    filename = 'company '+datetime.now().isoformat().replace(':','-').replace('.','-')

    img.save(directory+'/'+filename+'.jpg')

    return st.success('Saved file : {} in {}'.format( filename+'.jpg', directory ))

def run_segmentation() :
    
    SET_WIDTH = int(600) 
    normalize_image = 1 / 255.0
    resize_image = (1024, 512)
    
    IMG_OUTPUT_SHAPE_START = 1
    IMG_OUTPUT_SHAPE_END = 4 

    cv_enet_model = cv2.dnn.readNet('data/enet-cityscapes/enet-model.net')
    label_values = open('data/enet-cityscapes/enet-classes.txt').read().split('\n')
    CV_ENET_SHAPE_IMG_COLORS = open('data/enet-cityscapes/enet-colors.txt').read().split('\n')
    CV_ENET_SHAPE_IMG_COLORS = np.array([  np.array(color.split(',')).astype('int') for color in CV_ENET_SHAPE_IMG_COLORS ])
    
    image_files_list = st.file_uploader('Uploader Image', type=['png', 'jpg', 'jpeg', 'JPG'], accept_multiple_files= True)
    img_list = []   
    if image_files_list is not None :
        
        for img_files in image_files_list :
            img = load_image(img_files)
            img_array = np.array(img)
            img_list.append(img_array)
            st.image(img)
         
        if st.button('Detection') :
            for i in np.arange(len(img_list)) :
                sample_img = img_list[i]
                blob_img = cv2.dnn.blobFromImage(sample_img, normalize_image, resize_image, 0, swapRB = True, crop= False)
                cv_enet_model.setInput(blob_img)
                cv_enet_model_output = cv_enet_model.forward()

                classes_num, h, w = cv_enet_model_output.shape[IMG_OUTPUT_SHAPE_START : IMG_OUTPUT_SHAPE_END]
                class_map = np.argmax(cv_enet_model_output[0], axis=0)
                
                mask_class_map = CV_ENET_SHAPE_IMG_COLORS[class_map]
                
                mask_class_map = cv2.resize(mask_class_map, (sample_img.shape[1], sample_img.shape[0]), interpolation= cv2.INTER_NEAREST)
                class_map = cv2.resize(class_map, (sample_img.shape[1], sample_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                cv_enet_model_output = ( ( 0.4 * sample_img ) + (0.6 * mask_class_map) ).astype('uint8')
                st.image(cv_enet_model_output)

            my_legend = np.zeros( ( len(label_values) * 25 , 300 , 3 ), dtype='uint8' )
            for ( i, (class_name, img_color) ) in enumerate( zip(label_values, CV_ENET_SHAPE_IMG_COLORS) ):
                color_info = [ int(color) for color in img_color ]
                cv2.putText(my_legend, class_name, (5, (i*25)+17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2 )
                cv2.rectangle(my_legend, (100, (i*25)), (300, (i*25) + 25), tuple(color_info), -1 )
            st.image(my_legend)
