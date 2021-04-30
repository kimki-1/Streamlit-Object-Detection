import streamlit as st


def run_lane() :
    st.title('Lane Detection Video Test')
    st.markdown('## <로컬에서 실행 되는 Lane Detection>')
    st.video('data/videos/lane_video.mp4')
    st.markdown('***')
    st.write('[원본영상]')
    st.video('data/videos/lane_video2.mp4')
    st.write('[Result 영상]')
    st.video('data/videos/lane_ret.mp4')