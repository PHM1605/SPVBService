
import cv2
import time
import requests
import random
import numpy as np
import time
import os
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from core.helpers.azure_blob_helper import download_model
from collections import OrderedDict,namedtuple
import cv2
import re
from core.settings import get_settings

import itertools, copy
from PIL import ImageFont, ImageDraw, Image

settings = get_settings()

model_name = settings.MODEL_NAME
class_name = settings.CLASS_NAME
model_path = os.path.join("./core/weights", model_name)
class_path = os.path.join("./core/ml_utils/ml_classes", class_name)

if not os.path.isfile(model_path):
    print('Check Download Model: ', model_path)
    if not os.path.isdir("core/weights"):
        print('Make Model Dir: core/weights ')
        os.makedirs("core/weights")
    download_model(model_name, model_path)
    
if not os.path.isfile(class_path):
    if not os.path.isdir("core/ml_utils/ml_classes"):
        os.makedirs("core/ml_utils/ml_classes")
    download_model(class_name, class_path)



names = open(class_path).read().strip().split("\n")

colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}
template = re.compile(r'^posm_vsc_(\d+)f$')
cuda = False




providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4
session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)
        

def predict(data: dict, image_path: str):
    img = cv2.imread(image_path)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, C = image.shape
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    im = image.astype(np.float32)
    im /= 255
    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]
    inp = {inname[0]:im}
    outputs = session.run(outname, inp)[0]
    results = data.copy()
    results['details'] = {}
    results['details']['detections'] = []
    results['details']['result'] = {}
    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
        batch_id = int(batch_id)
        score = round(float(score),3)
        if score < 0.25: continue
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        box = [max(0,box[0]),max(0,box[1]),min(W,box[2]),min(H,box[3]), score, cls_id]
        name = names[cls_id]
        color = colors[name]
        results['details']['detections'].append(box)
        
    print(results)
    return results