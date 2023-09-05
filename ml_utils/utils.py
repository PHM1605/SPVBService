import os, cv2, copy
import numpy as np
from PIL import ImageFont, ImageDraw, Image

def draw_result(img, boxes, color, put_percent, put_label=False):
    ret = copy.deepcopy(img)
    for box in boxes:
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        x1, y1, x2, y2, prob, cl = box
        ret = cv2.rectangle(ret, (x1, y1), (x2, y2), color, thickness)
        if put_percent:
            ret = cv2.putText(ret, str(round(prob, 2)), (x1, y1), font, 0.6, color, thickness)
        if put_label:
            ret = cv2.putText(ret, str(cl), (x1, y1), font, 0.6, color, thickness)
    return ret

def extract_to_image(img, response):
    detections = response["details"]["detections"]
    result = response["details"]["result"]
    # draw all as gray
    img = draw_result(img, detections, color=(192,192,192), put_percent=True)
    # draw bottles as green
    img = draw_result(img, get_boxes_exclude_labels(result, ["SPACE", "NON_SPVB"]), color=(0,255,0), put_percent=True)
    # draw space as red
    img = draw_result(img, get_boxes_of_labels(result, ["SPACE"]), color=(0,0,255), put_percent=True)
    # draw nonspvb as purple
    img = draw_result(img, get_boxes_of_labels(result, ["NON_SPVB"]), color=(255,0,255), put_percent=True)
    img = put_text(img, response["message"], loc=[10,10])
    return img

def get_boxes_of_labels(dict_labels, labels):
    ret = []
    for floor in dict_labels:
        for item in dict_labels[floor]:
            if item in labels:
                ret += dict_labels[floor][item]
    return ret

def get_boxes_exclude_labels(dict_labels, labels):
    ret = []
    for floor in dict_labels:
        for item in dict_labels[floor]:
            if item not in labels:
                ret += dict_labels[floor][item]
    return ret

# get result dictionary of one image
def get_result_dict(img_path, total_result_dict):
    for one_img_result in total_result_dict:
        if img_path == one_img_result["image_url"]:
            return one_img_result

# Vietnamese display on image
def put_text(img, text, loc):
    # set font size; for image-height of 640 fontsize of 16 is ok 
    h = img.shape[0]
    font_size = int(16/640*h)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(os.path.join("ml_utils", "fonts", "SVN-Arial 2.ttf"), font_size)
    bbox = draw.textbbox(loc, text, font=font)
    draw.rectangle(bbox, fill=(0,255,255))
    draw.text(loc, text, font=font, fill=(0,0,255))
    return np.array(img_pil)

# count number of items from a list of items
def count_item(item, list_items):
    count = 0
    for it in list_items:
        if item == it:
            count += 1
    return count

def search_bounding_boxes(boxes, label):
    ret = []
    for box in boxes:
        if box.label == label:
            ret.append([box.x1, box.y1, box.x2, box.y2])
    return ret
