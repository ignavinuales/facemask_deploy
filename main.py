import streamlit as st
import cv2 
import numpy as np
from detect_st_v2 import detect
from detect_video import detect_video
from PIL import Image
import subprocess
import tempfile

def main():
    st.title("Face mask detection app 😷")
    st.markdown("""
    This app uses computer vision to detect whether people are using face masks or not. 

    Upload an image or video of your choosing or use your webcam, and hit the button 'Detect face mask'.
    """)

    with st.expander("Click to see an example after running the model on a single image"):
        img_example = Image.open("example.jpg")
        st.image(img_example, width=500)
    
    # RUN THE MODEL ON IMAGE
    st.subheader("Run model on image")
    uploaded_image = st.file_uploader(label="Please upload an image", )

    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        if st.button("Detect face mask on image 😷"):
            with st.spinner("In progress..."):
                img = detect(im0=opencv_image, weights='best.pt', img_size=640, iou_thres=0.5, conf_thres=0.5)
                st.image(img, channels='BGR')

    # RRUN THE MODEL ON MP4 VIDEO
    st.subheader("Run model on video")
    video_data = st.file_uploader("Upload .mp4 video", ['mp4'])

    if video_data:
        # save uploaded video to disc
        temp_file_1 = tempfile.NamedTemporaryFile(delete=False,suffix='.mp4')
        temp_file_1.write(video_data.getbuffer())
        temp_file_2 = tempfile.NamedTemporaryFile(delete=False,suffix='.mp4')
        st.write("Note: it may take a while due to server slowness. A 15-second video takes about 2 minutes to be processed.")
        # st.video(temp_file_1.name)
        if st.button("Detect face mask on video 😷"):
            with st.spinner(text="In progress..."):
                output = detect_video(video_path=temp_file_1.name, temp_file=temp_file_2.name)          
                # temp_file_3 = tempfile.NamedTemporaryFile(delete=False,suffix='.mp4')
                # subprocess.call(args=f"ffmpeg -y -i {temp_file_2.name} -c:v libx264 {temp_file_3.name}".split(" "))
            st.success("Done!")
            # st.video(temp_file_3.name)
            result_video = open(temp_file_2.name, "rb")
            st.write("Please download the video:")
            st.download_button(label="Download video file", data=result_video,file_name='mask_detection.mp4')

    # RUN THE MODEL ON WEBCAM
    st.subheader("Run model on webcam")
    st.write("Click the checkbox to open up your camera. You'll probabily have to give permission from your web browser and reload the website.")
    webcam = st.checkbox('Turn on webcam')
    if webcam:
        uploaded_file = st.camera_input("Take a photo from your webcam:")


        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            if st.button("Detect face mask on webcam 😷"):
                with st.spinner(text="In progress..."):
                    img = detect(im0=opencv_image, weights='best.pt', img_size=640, iou_thres=0.5, conf_thres=0.5)
                    st.image(img, channels='BGR')
     
if __name__ == "__main__":
    main()