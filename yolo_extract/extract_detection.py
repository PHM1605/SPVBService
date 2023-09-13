import cv2, os, glob, json, torch
import numpy as np
from yolo_extract.config import config 
from yolo_extract.models.experimental import attempt_load
from yolo_extract.utils.datasets import letterbox
from yolo_extract.utils.general import non_max_suppression, scale_coords

def extract(request):
    # img_dir: '../samples/images/'
    # img_size: 640
    # extensions: ["*.jpg", "*.png", "*.jpeg"]
    # model: 'rack0821.pt'
    img_list = []
    for extension in request["extensions"]:
        img_list.extend(glob.glob(request["img_dir"] + extension))
    model = attempt_load(request["model"], map_location='cpu')
    response = []
    for img_path in img_list:
        one_img_config = config
        one_img_config["classes"] = model.names
        tmp = img_path.split('/')[1:]
        one_img_config["image_url"] = os.path.join(*tmp)
        img0 = cv2.imread(img_path) # BGR
        img = letterbox(img0, request["img_size"], stride = int(model.stride.max()))[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to('cpu')
        img = img.float()  
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        conf_thres = 0.25
        iou_thres = 0.45
        with torch.no_grad():
            pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=None) [0]
        # Rescale boxes from img_size to img0_size
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
        pred = pred.cpu().detach().numpy()
        pred = [pr for pr in pred]
        for pr in pred:
            pr_cvt = [int(pr[0]), int(pr[1]), int(pr[2]), int(pr[3]), float(pr[4]), int(pr[-1])]
            one_img_config["details"]["detections"].append(pr_cvt)
        response.append(one_img_config)
    return response
