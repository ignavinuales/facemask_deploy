# Face mask detection using YOLOv7 (Computer Vision)
This project aimed to retrain a YOLOv7 model to detect people wearing face masks. Moreover, I created and deployed a web app on Streamlit for users to test the model. It can be accessed by the following link: https://ignavinuales-facemask-deploy-main-evthqf.streamlit.app/

## Objectives Then, create an MVP web app for users to test the model.
The objective of this project is to use the official pre-trained YOLOv7 model (built on PyTorch) and retrain it on a custom dataset to detect people wearing face masks. The class labels are:

- Wearing mask
- Not wearing mask
- Worn incorrectly

If you want access to the official YOLOv7 implementation for a deeper understanding of its architechure, please refer to https://github.com/WongKinYiu/yolov7 

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
5. I measure the model performance and collect new data to improve accuracy/recall. Example: the model was not performing well on incorrectly worn masks. Therefore, I collected and labelled new data from that class and retrained the model.
6. After obtaining a well-performing model, the deployment was done on Streamlit Cloud for user testing.