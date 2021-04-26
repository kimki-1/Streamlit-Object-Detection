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
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


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
    # http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz
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
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = tf.convert_to_tensor(output_dict['detection_masks'], dtype=tf.float32)
        output_dict['detection_boxes'] = tf.convert_to_tensor(output_dict['detection_boxes'], dtype=tf.float32)
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])  
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                        tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        
    return output_dict

## 예측한 결과를 보여줘라 
def show_inference(model, image_np):    
    # image_np = np.array(image)
    # 이미지를 오픈으로 받아오면 변경시켜야한다 
    # image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    print(image_np)
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

    image_np = cv2.imshow('result', cv2.resize(image_np, (1600, 1000)))
    st.image(image_np)

def run_ssd() : 

    image_files_list = st.file_uploader('Uploader Image', type=['png', 'jpg', 'jpeg', 'JPG'], accept_multiple_files= True)
    img_list = []
    if image_files_list is not None :
        # 2. 각 파일을 이미지로 바꿔줘야 한다.
        # 2-1.모든 파일이 img_list에 이미지로 저장됨
        for img_files in image_files_list :
            img = load_image(img_files)
            img_list.append(img)
            st.image(img)

        print(img_list)
        
        if st.button('Detection') :
            # patch tf1 into `utils.ops`
            utils_ops.tf = tf.compat.v1
            # Patch the location of gfile
            tf.gfile = tf.io.gfile

            PATH_TO_LABELS = 'C:\\Users\\admin\\Documents\\Tensorflow\\models\\research\\object_detection\\data\\mscoco_label_map.pbtxt'
            category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)
            # print(category_index)

            model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
            detection_model = load_model(model_name)
            # print(detection_model.signatures['serving_default'].inputs)
            # print(detection_model.signatures['serving_default'].output_dtypes)
            # print(detection_model.signatures['serving_default'].output_shapes)


            # ## 함수 테스트
            # PATH_TO_TEST_IMAGE_DIR = pathlib.Path('data\\images')
            # TEST_IMAGE_PATH = sorted( list(PATH_TO_TEST_IMAGE_DIR.glob("*.jpg")) )

            # show_inference(detection_model, img_list)
            
            # # for image_path in TEST_IMAGE_PATH:
            # #     show_inference(detection_model, image_path)

            # cv2.waitKey()
            # cv2.destroyAllWindows()



 