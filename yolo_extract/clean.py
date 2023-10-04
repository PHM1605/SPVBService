import os
from ml_utils.analysis import calculate_iou
from ml_utils.models import BoundingBox

def max_iou_box_and_boxes(box, ibox1, boxes):
    max_iou = 0
    idx_max_box = -1
    for ibox2, other_box in enumerate(boxes):
        box1 = BoundingBox(*box)
        box2 = BoundingBox(*other_box)
        if ibox1 == ibox2: continue
        iou = calculate_iou(box1, box2)
        if iou > max_iou:
            max_iou = iou
            idx_max_box = ibox2
    return max_iou, idx_max_box

def clean_output_yolo(input_format):  
    ret = []  
    for detection in input_format:
        file_name = os.path.basename(detection["image_path"]).split('.')[0] + ".txt"
        boxes = detection["details"]["detections"]
        detections = []
        for ibox, box in enumerate(boxes):
            max_iou, idx_max_box = max_iou_box_and_boxes(box, ibox, boxes)
            # choose box with bigger probability
            if max_iou < 0.9 or (max_iou >= 0.9 and box[-2] > boxes[idx_max_box][-2]):
                cl = box[-1]
                w = (box[2] - box[0]) / detection["image_shape"][1]
                h = (box[3] - box[1]) / detection["image_shape"][0]
                cen_x = (box[0] + box[2]) / 2 / detection["image_shape"][1]
                cen_y = (box[1] + box[3]) / 2 / detection["image_shape"][0]
                detections.append([cl, round(cen_x, 3), round(cen_y, 3), round(w, 3), round(h, 3)])
        ret.append({"file_name": file_name, "detections": detections})
    return ret

    

