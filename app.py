import streamlit as st 
from ssd_app import run_ssd
from yolo_app import run_yolo
from segmentation_app import run_segmentation

def main():

    st.title('Streamlit Object Detection')
    side_bar =  st.sidebar.selectbox('Menu',['Home','SSD', 'YOLO', 'Semantic Segmentation'])

    if side_bar == 'Home' :
        st.subheader('')

    if side_bar == 'SSD' :
        run_ssd()

    if side_bar == 'YOLO' :
        run_yolo()

    if side_bar == 'Semantic Segmentation' :
        run_segmentation()

if __name__ == '__main__' :
    main()