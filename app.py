import streamlit as st 
from ssd_app import run_ssd
from yolo_app import run_yolo
from segmentation_app import run_segmentation
from lane_app import run_lane

def main():

    
    side_bar =  st.sidebar.selectbox('Menu',['Home','SSD', 'YOLO', 'Semantic Segmentation','Lane Detection'])
    
    if side_bar == 'Home' :
        st.title('자율주행 자동차 Object Detection 프로젝트')
        st.write('')
        side_radio = st.sidebar.radio('Introduction',['Object Detection', 'SSD', 'YOLO', 'Semantic Segmentation','Lane Detection'])
        
        if side_radio == 'Object Detection' :
            st.markdown('## <Object Detection>')
            st.write('')
            st.write('[Object Detection 예제 영상]')
            st.video('data/videos/Object Detection.mp4')
            st.markdown('***')
            st.markdown('## <Object Detection 모델소개>')
            st.write('')
            st.image('data/images/object.PNG')
            st.write('↑ Computer Vision 분야 Major 학회에서 소개되었던 Object Detection 논문흐름')
            st.markdown('### ● 사용한 모델은 **_SSD, YOLO_** 모델을 이용해서 Object Detection 구현, 또한 자동차 주변을 식별할 수 있는 Semantic Segmentation 구현')
            

        if side_radio == 'SSD' :
            st.markdown('## <SSD : Single Shot Multibox Detector>')
            st.markdown('#### SSD와 YOLO V1는 비슷한 시기에 나왔습니다. YOLO의 문제점인 그리드 크기보다 작은 물체를 잡아내지 못하는 문제를 보완한 모델입니다.  ')
            st.markdown('***')
            st.markdown('## <SSD Model Introduction>')
            st.write('')
            st.image('data/images/SSD.PNG')
            st.write('↑ Detection하기위한 여러 feature map')
            st.markdown('#### ● 2015년에 나온 YOLO v1 모델의 단점을 보완한 SSD 모델의 feature map을 확인 할 수 있습니다. ')
            st.markdown('***')
            st.markdown('## <NMS>')
            st.write('')
            st.image('data/images/SSD2.PNG')
            st.markdown('#### ● 위와 같이 요약을 할 수 있는데 이렇게 되면 각각의 ouput feature map 에서 뽑은 anchor boxes 때문에 최종단계에서 한 객체에 대한 많은 bounding boexs가 생성이 됩니다.')
            st.markdown('## ')
            st.image('data/images/SSD3.PNG')
            st.image('data/images/SSD1.PNG')
            st.markdown('#### ● 그래서 많은 anchor boxes중 NMS를 적용하여 가장 높은 class confidence score를 갖는 anchor box를 선택하게 됩니다.')
        
        if side_radio == 'YOLO' :
            st.markdown('## <YOLO : You Only Look Once >')
            st.image('data/images/YOLO.PNG')
            st.image('data/images/YOLO3.PNG')
            st.markdown('#### ● 위에서 표시한 30개의 cell에는 각각 다른 정보가 담겨져있습니다. 하나의 bounding box에 대한 좌표정보(=x,y,w,h)와 confidence(=P(object))정보가 총 2개(=2x5)들어있고, 나머지 20개의 정보는 Pascal VOC challenge가 20개 class에 대해서 detection 작업을 하는것이기 때문에, 20개의 class에 대한 classification 확률을 담게 됩니다. ')
            st.markdown('####')
            st.markdown('***')
            st.markdown('## <YOLO Model Introduction>')
            st.image('data/images/YOLO2.PNG')
            st.write('↑ YOLO Architecture')
            st.image('data/images/YOLO1.PNG')
            st.write('↑ Comparison to Other Detectors')
            st.markdown('#### ● YOLOv3은 매우 빠르고 정확합니다. mAP는 초점 손실과 동등하지만 약 4배 더 빠릅니다. 또한 재교육이 필요 없이 모델의 크기를 변경하기만 하면 속도와 정확성을 쉽게 비교할 수 있습니다. ')

        if side_radio == 'Semantic Segmentation' :
            st.markdown('## <Semantic Segmentation>')
            st.markdown('#### Semantic Segmentation이란, 이미지를 픽섹별로 분류하는 것입니다')
            st.image('data/images/segmentation1.PNG')
            st.markdown('***')
            st.markdown('## <자율주행 자동차에서의 Semantic Segmentation>')
            st.image('data/images/segmentation3.PNG')
            st.markdown('#### ● 자율 주행 자동차 정면의 카메라를 장착해, 대상이 사람인지, 자동차인지, 횡단보도인지를 구분해서 Semantic Segmentation이 사용된다. 빠르고 정확하게 구분 하지 못하면 사고가 날수 있기 때문에 현재 Semantic Segmentation의 정확도와 속도를 높이기 위해 많은 연구가 이루어지고 있습니다.')
            st.markdown('####')
            st.markdown('***')
            st.markdown('## <Semantic Segmentation VS Instance Segmentation>')
            st.image('data/images/segmentation.PNG')
            st.markdown('#### ● Object Detection과 Semantic Segmentation 그리고 Instance Segmentation의 차이를 보여주는 예시입니다.')
        
        if side_radio == 'Lane Detection' :
            st.markdown('## <Lane Detection>')
            st.image('data/images/lane_img.jpg')
            st.write('위의 이미지로 Lane Detection')
            st.write('')
            st.markdown('***')
            st.image('data/images/lane_img2.png')
            st.write('OpenCV에서 제공하는 Canny Edge Detection을 사용하여 위의 이미지를 만들어 냅니다.')
            st.write('원본이미지에서 엣지 검출을 보다 좋게 하기위해 Gray Scale, Smoothing(GaussinBlur)을 합니다')
            st.write('')
            st.markdown('***')
            st.image('data/images/lane_img4.png')
            st.write('위의 이미지와 Canny Edge Detection의 이미지를 bitwise_and(비트와이즈)을 해서 아래의 이미지를 만듭니다.')
            st.image('data/images/lane_img3.png')
            st.write('효율적인 Lane Detection을 위해 필요한 부분만 남겨둔것 입니다.')
            st.write('')
            st.markdown('***')
            st.image('data/images/lane_img5.png')
            st.write('위의 선들을 허브변환을 이용하여 이어줍니다.')

    if side_bar == 'SSD' :
        run_ssd()

    if side_bar == 'YOLO' :
        run_yolo()

    if side_bar == 'Semantic Segmentation' :
        run_segmentation()
    
    if side_bar == 'Lane Detection' :
        run_lane()

if __name__ == '__main__' :
    main()