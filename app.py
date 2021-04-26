import streamlit as st 
from ssd_app import run_ssd

def main():

    st.title('Streamlit Object Detection')
    side_bar =  st.sidebar.selectbox('Menu',['Home','SSD', 'YOLO', 'Semantic Segmentation'])

    if side_bar == 'Home' :
        st.subheader('')

    if side_bar == 'SSD' :
        run_ssd()

if __name__ == '__main__' :
    main()