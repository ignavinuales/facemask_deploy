from models.experimental import attempt_load
import numpy as np
import random
import time
import torch
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device


def run_inference_image(input_image: np.ndarray, weights: str, img_size: int, conf_thres: float, iou_thres: float, device: str = 'cpu') -> np.ndarray:
    """
    Run image inference on the Yolov7 computer vision model.

    Args:
        input_image (numpy.ndarray): image to be fed into the model.
        weights (str): path to the model .pt file.
        img_size (int): number of pixels of the input_image.
        conf_thres (float): model's confidence threshold.
        iou_thres (float): IoU threshold.
        device (str): device where the model is run (GPU-CUDA or CPU). It defaults to 'cpu'.

    Returns:
        input_image (numpy.ndarray): output of the model. Hence, input_image with bounding boxes and classes detected.

    """

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once

    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    # Read image
    img = letterbox(input_image, img_size, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Warmup
    if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]

    # Inference
    with torch.no_grad(): 
        pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    # Process detections
    for _, det in enumerate(pred):  # detections per image

        # gn = torch.tensor(input_image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to input_image size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], input_image.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, input_image, label=label,
                             color=colors[int(cls)], line_thickness=2)

    print(f'Done. ({time.time() - t0:.3f}s)')

    return input_image
