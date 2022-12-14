from fastapi import FastAPI, File, UploadFile

from pydantic  import BaseModel
import cv2
import time
import requests
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple
import numpy

app = FastAPI()

class DetectMask(BaseModel):
    image: float

@app.get("/")
async def get_detect(item:DetectMask):
    return [item, item]

# @app.post("/files/")
# async def create_file(file: bytes = File()):
#     return {"file_size": len(file)}

# @app.post("/uploadfile/")
# async def create_upload_file(image: UploadFile, model:UploadFile):
#     cuda = True
#     w = model
#     img = cv2.imread(image)

#     providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
#     session = ort.InferenceSession(w, providers=providers)


#     def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
#         # Resize and pad image while meeting stride-multiple constraints
#         shape = im.shape[:2]  # current shape [height, width]
#         if isinstance(new_shape, int):
#             new_shape = (new_shape, new_shape)

#         # Scale ratio (new / old)
#         r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#         if not scaleup:  # only scale down, do not scale up (for better val mAP)
#             r = min(r, 1.0)

#         # Compute padding
#         new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#         dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

#         if auto:  # minimum rectangle
#             dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

#         dw /= 2  # divide padding into 2 sides
#         dh /= 2

#         if shape[::-1] != new_unpad:  # resize
#             im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
#         top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#         left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#         im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        
#         return im, r, (dw, dh)

#     names = ['without_mask', 'mask_wear_incorrect', 'with_mask']
#     colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     image = img.copy()
#     image, ratio, dwdh = letterbox(image, auto=False)
#     image = image.transpose((2, 0, 1))
#     image = np.expand_dims(image, 0)
#     image = np.ascontiguousarray(image)

#     im = image.astype(np.float32)
#     im /= 255
#     im.shape

#     outname = [i.name for i in session.get_outputs()]
#     outname

#     inname = [i.name for i in session.get_inputs()]
#     inname

#     inp = {inname[0]:im}

#     outputs = session.run(outname, inp)[0]
#     return outputs
