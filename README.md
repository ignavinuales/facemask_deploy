# Face mask detection using Yolov7 (Computer Vision)
This project aimed to create an app to detect people wearing face masks and if they are, whether they are worn correctly or not. The app is deployed on streamlint and you can use and test it by accessing the following link: https://ignavinuales-facemask-deploy-main-evthqf.streamlit.app/

## Project Directory Structure
    .
    ├── models/         
         ├──...                 # Official Yolov7 modules 
    ├── utils/
         ├──...                 # Official Yolov7 utilitt modules 
    ├── example.ong             # output image after running model inference 
    ├── inference_image.py      # python module for image inference using Yolov7
    ├── inference_video.py      # python module for video inference using Yolov7
    ├── main.py                 # python module to run Streamlit web app
    ├── packages.txt            # necessary python packages
    ├── requirements.txt        # necessary python libraries
    ├── yolo_model.pt           # .pt file with the trained Yolov7 model

## Methodology
1. First, some of the data were collected from open data sources. Then, I collected some other data by gathering personal photos from 2020 to 2022.
2. The unlabeled data were manually labelled using roboflow.
3. After researching open-source object detection algorithms, I picked Yolov7. You can access the published paper here.
4. I trained the Yolov7 model on my custom data.
5. Measure the model performance and collect new data to improve accuracy/recall. Example: the model was not performing well on incorrectly worn masks. Therefore, I collected and labelled new data from that class and retrained the model.
6. After obtaining a well-performing model, the deployment was done on Streamlit Cloud services for user testing.