import streamlit as st 
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2 
import os 
import pathlib 
import time

from PIL import Image, ImageFilter, ImageEnhance
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile
PATH_TO_LABELS = 'data/mscoco_label_map.pbtxt'
# PATH_TO_LABELS = 'C:\\Users\\admin\\Documents\\Tensorflow\\models\\research\\object_detection\\data\\mscoco_label_map.pbtxt'
# PATH_TO_LABELS = 'C:\\Users\\5-18\\Documents\\Tensorflow\\models\\research\\object_detection\\data\\mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)

def load_image(image_file) :
    img = Image.open(image_file) 
    return img 

def save_uploaded_file(directory, img) :

    if not os.path.exists(directory) :
        os.makedirs(directory)

    filename = 'company '+datetime.now().isoformat().replace(':','-').replace('.','-')

    img.save(directory+'/'+filename+'.jpg')

    return st.success('Saved file : {} in {}'.format( filename+'.jpg', directory ))
   
def load_model(model_name):
    
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name, 
        origin=base_url + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir)/"saved_model"

    model = tf.saved_model.load(str(model_dir))

    return model

def run_inference_for_single_image(model, image):
    
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]
    
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = tf.convert_to_tensor(output_dict['detection_masks'], dtype=tf.float32)
        output_dict['detection_boxes'] = tf.convert_to_tensor(output_dict['detection_boxes'], dtype=tf.float32)

        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])  
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                        tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        
    return output_dict


def show_inference(model, image_np):       
    output_dict = run_inference_for_single_image(model, image_np)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.array(output_dict['detection_boxes']),
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed',None),
        use_normalized_coordinates=True,
        line_thickness=7)
    
    st.image(image_np)

def run_ssd() : 
    st.title('SSD Detection')
    side_radio = st.sidebar.radio('Detection Select',['Image Detection', 'Video Detection'])

    if side_radio =='Image Detection' :
        st.markdown('## <Image Detection>')
        st.video('data/videos/SSD Image.mp4')
        st.markdown('#### ※ AWS EC2 프리티어를 사용하기 때문에 SSD모델을 실행시키기 어려워서 로컬에서 SSD모델을 실행 시키는 영상(전체화면 권장)')
        # image_files_list = st.file_uploader('Uploader Image', type=['png', 'jpg', 'jpeg', 'JPG'], accept_multiple_files= True)
        # img_list = []
        # if image_files_list is not None :
            
        #     for img_files in image_files_list :
        #         img = load_image(img_files)
        #         img_array = np.array(img)
        #         img_list.append(img_array)
        #         st.image(img)

        #     if st.button('Detection') :

        #         model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
        #         detection_model = load_model(model_name)
                
        #         for i in np.arange(len(img_list)) :
        #             show_inference(detection_model, img_list[i])
        
    if side_radio == 'Video Detection' :
        st.markdown('## <Video Detection>')
        st.video('data/videos/SSD.mp4')
        st.video('data/videos/SSD2.mp4')
        st.video('data/videos/SSD3.mp4')
            




 