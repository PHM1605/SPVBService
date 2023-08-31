import os, cv2, copy, math, torch
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from indices import get_indices
from models import BoundingBox

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
            ret = cv2.putText(ret, str(label), (x1, y1), font, 0.6, color, thickness)
    return ret

def extract_to_image(img, result_dict):
    detections = result_dict["details"]["detections"]
    result = result_dict["details"]["result"]
    # draw all as gray
    img = draw_result(img, detections, color=(192,192,192), put_percent=True)
    # draw bottles as green
    img = draw_result(img, get_boxes_exclude_labels(result, ["SPACE", "NON_SPVB"]), color=(0,255,0), put_percent=True)
    # draw space as red
    img = draw_result(img, get_boxes_of_labels(result, ["SPACE"]), color=(0,0,255), put_percent=True)
    # draw nonspvb as purple
    img = draw_result(img, get_boxes_of_labels(result, ["NON_SPVB"]), color=(255,0,255), put_percent=True)
    img = put_text(img, result_dict["message"], loc=[10,10])
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
    font = ImageFont.truetype(os.path.join("fonts", "SVN-Arial 2.ttf"), font_size)
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

def extract_to_image_and_json(result_dict):
    if result_dict["evaluation_result"] == True:
        outp_img_path = f"{self.img_key}_output_ok"
        output_file = outp_img_path + ".json"
        cv2.imwrite(
            os.path.join(self.result_folder, outp_img_path + ".jpg"), img
        )
    elif export_data["evaluation_result"] == False:
        outp_img_path = f"{self.img_key}_output_notok"
        output_file = outp_img_path + ".json"
        cv2.imwrite(
            os.path.join(self.result_folder, outp_img_path + ".jpg"), img
        )
    json_str = json.dumps(export_data, ensure_ascii=False)
    with open(
        os.path.join(self.result_folder, output_file), "w", encoding="utf-8"
    ) as f:
        f.write(json_str)



