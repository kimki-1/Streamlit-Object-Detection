import time 
import cv2
import numpy as np 
import os 
import pathlib 
import streamlit as st
 

from PIL import Image
from yolo.model.yolo_model import YOLO

def process_image(img) :
    """ 이미지 리사이즈 하고, 차원확장 
    img : 원본 이미지
    결과는 (64, 64, 3)으로 프로세싱된 이미지 변환 """
    
    image_org = cv2.resize(img, (416, 416), interpolation=cv2.INTER_CUBIC)
    image_org = np.array(image_org, dtype='float32')
    image_org = image_org / 255.0
    image_org = np.expand_dims(image_org, axis=0)

    return image_org 

def get_classes(file) :
    """ 클래스의 이름을 가져온다. 
    리스트로 클래스 이름을 반환한다. """

    with open(file) as f :
        name_of_class = f.readlines()
    
    name_of_class = [ class_name.strip() for class_name in name_of_class ]

    return name_of_class

def box_draw(image, boxes, scores, classes, all_classes):
    """ image : 오리지날 이미지 
        boxes : 오브젝트의 박스데이터, ndarray
        classes : 오브젝트의 클래스 정보, ndarray
        scores : 오브젝트의 확률, ndarray 
        all_classes : 모든 클래스 이름 """

    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 255), 2,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print()

def detect_image(image, yolo, all_classes) : 
    """ image : 오리지날 이미지
        yolo : 욜로 모델 
        all_classes : 전체 클래스 이름 

        변환된 이미지 리턴! """

    pimage = process_image(image)

    image_boxes, image_classes, image_scores = yolo.predict(pimage, image.shape)

    if image_boxes is not None : 
        box_draw(image, image_boxes, image_scores, image_classes, all_classes)
    
    st.image(image)

def load_image(image_file) :
    img = Image.open(image_file) 
    return img 

def save_uploaded_file(directory, img) :

    if not os.path.exists(directory) :
        os.makedirs(directory)

    filename = 'company '+datetime.now().isoformat().replace(':','-').replace('.','-')

    img.save(directory+'/'+filename+'.jpg')

    return st.success('Saved file : {} in {}'.format( filename+'.jpg', directory ))

def run_yolo() :
    st.title('YOLO Detection')
    side_radio = st.sidebar.radio('Detection Select',['Image Detection', 'Video Detection'])

    if side_radio =='Image Detection' :
        st.markdown('## <Image Detection>')
        st.markdown('#### ※ AWS EC2 프리티어를 사용하기 때문에 YOLO모델을 실행시키기 어려워서 로컬에서 YOLO모델을 실행 시키는 영상(전체화면 권장)')
        st.write('')
        st.video('data/videos/YOLO image.mp4')
        st.markdown('***')
        st.markdown('## <Detection Image Result>')
        st.image('data/images/yolo_ret.png')
        st.image('data/images/yolo_ret2.png')
        st.image('data/images/yolo_ret3.png')

        # image_files_list = st.file_uploader('Uploader Image', type=['png', 'jpg', 'jpeg', 'JPG'], accept_multiple_files= True)
        # img_list = []
        # if image_files_list is not None :
            
        #     for img_files in image_files_list :
        #         img = load_image(img_files)
        #         img_array = np.array(img)
        #         img_list.append(img_array)
        #         st.image(img)

        #     if st.button('Detection') :
        #         yolo = YOLO(0.6, 0.5)
        #         all_classes = get_classes('yolo/data/coco_classes.txt')

                
        #         for i in np.arange(len(img_list)) :
        #             result_image = detect_image(img_list[i], yolo, all_classes)
                
    if side_radio == 'Video Detection' :
        st.markdown('## <Video Detection>')
        st.markdown('#### ※ AWS EC2 프리티어를 사용하기 때문에 YOLO모델을 실행시키기 어려워서 로컬에서 YOLO모델을 실행 시키는 영상(전체화면 권장)')
        st.write('')
        st.video('data/videos/yolo_video.mp4')
        st.markdown('***')

        st.write('[원본영상]')
        st.video('data/videos/SSD_ori.mp4')
        st.write('[Detection 영상]')
        st.video('data/videos/YOLO.mp4')
        st.markdown('***')

        st.write('[원본영상]')
        st.video('data/videos/SSD3_ori.mp4')
        st.write('[Detection 영상]')
        st.video('data/videos/YOLO2.mp4')
