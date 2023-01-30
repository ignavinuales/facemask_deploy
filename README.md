# Face mask detection using YOLOv7 (Computer Vision)
This project aimed to create an app to detect people wearing face masks. The app is deployed on streamlint and you can use and test it by accessing the following link: https://ignavinuales-facemask-deploy-main-evthqf.streamlit.app/

## Objectives
Trained a YOLOv7 model to Detect people wearing face masks, not wearing masks, and wearing them incorrectly. Then, create an MVP web app for users to test the model.

## Project Directory Structure
    .
    ├── models/         
         ├──...                 # Official YOLOv7 modules 
    ├── utils/
         ├──...                 # Official YOLOv7 utilitt modules 
    ├── example.ong             # output image after running model inference 
    ├── inference_image.py      # python module for image inference using YOLOv7
    ├── inference_video.py      # python module for video inference using YOLOv7
    ├── main.py                 # python module to run Streamlit web app
    ├── packages.txt            # necessary python packages
    ├── requirements.txt        # necessary python libraries
    ├── yolo_model.pt           # .pt file with the trained YOLOv7 model

## Methodology
1. First, some of the data were collected from open data sources. Then, I collected some other data by gathering personal photos from 2020 to 2022.
2. The unlabeled data were manually labelled using roboflow.
3. After researching open-source object detection algorithms, I picked YOLOv7. You can access the published paper here: https://arxiv.org/abs/2207.02696
4. I trained the YOLOv7 model on my custom data.
5. Measure the model performance and collect new data to improve accuracy/recall. Example: the model was not performing well on incorrectly worn masks. Therefore, I collected and labelled new data from that class and retrained the model.
6. After obtaining a good-performing model, the deployment was done on Streamlit Cloud for user testing.

If you want access to the official YOLOv7 implementation, refer to https://github.com/WongKinYiu/yolov7 