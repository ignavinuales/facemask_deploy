import cv2
import torch
import numpy as np
from numpy import random
from models.experimental import attempt_load
from typing import Union
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box


def run_inference_video(video_path: str, output_file: str, weights: str, img_size: int, conf_thres: float,
                         iou_thres: float, device: Union[str, int] = 'cpu') -> None:
  """
  Run video inference on the Yolov7 computer vision model.

  Args:
    video_path (str): directory path where the video to be fed into the model is located.
    output_file (str) directory path where the output video is saved after running the model. 
    weights (str): path to the model .pt file.
    img_size (int): number of pixels of the input_image.
    conf_thres (float): model's confidence threshold.
    iou_thres (float): IoU threshold.
    device (str): device where the model is run (GPU-CUDA or CPU). It defaults to 'cpu'. For GPU choose 0, 1, 2, or 3. 
  """

  # Initializing video object
  video = cv2.VideoCapture(video_path)

  #Video information
  fps = video.get(cv2.CAP_PROP_FPS)
  w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

  # output = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'),fps , (w,h))
  output = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'),fps , (w,h))
  torch.cuda.empty_cache()

  # Initializing model and setting it for inference
  with torch.no_grad():
    set_logging()
    half = device != 'cpu'
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    img_size = check_img_size(img_size, s=stride)  # check img_size
    if half:
      model.half()

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    if device != 'cpu':
      model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))

    for j in range(nframes):

        ret, img0 = video.read()
        
        if ret:
          img = letterbox(img0, img_size, stride=stride)[0]
          img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
          img = np.ascontiguousarray(img)
          img = torch.from_numpy(img).to(device)
          img = img.half() if half else img.float()  # uint8 to fp16/32
          img /= 255.0  # 0 - 255 to 0.0 - 1.0
          if img.ndimension() == 3:
            img = img.unsqueeze(0)

          # Inference
          with torch.no_grad(): 
            pred = model(img, augment= False)[0]

          pred = non_max_suppression(pred, conf_thres, iou_thres)

          for _, det in enumerate(pred):
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
              det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

              for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
      
              for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=2)
          
          
          output.write(img0)

        else:
          break

  output.release()
  video.release()
  print("Done!")