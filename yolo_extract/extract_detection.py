import copy, cv2, os, glob, json, torch
import numpy as np
from .config import config 
from ml_utils.analysis import get_boxes_and_indices, remove_overlap_boxes
from yolo_extract.models.experimental import attempt_load
from yolo_extract.utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

def extract(request):
    img_list = []
    for extension in request["extensions"]:
        img_list.extend(glob.glob(os.path.join(request["img_dir"], extension)))
    model = attempt_load(request["model"], map_location='cpu')
    response = []
    for img_path in img_list:
        one_img_config = copy.deepcopy(config)
        one_img_config["image_path"] = img_path
        one_img_config["classes"] = model.names
        img0 = cv2.imread(img_path) # BGR
        one_img_config["image_shape"] = img0.shape
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
        print(f"Done for {os.path.basename(img_path)}")

    with open("samples/images_result.json", "w") as f:
        json.dump(response, f, ensure_ascii=True)
    return response
