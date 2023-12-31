import os, cv2, copy, requests, shutil
import numpy as np
import pandas as pd
from .indices import get_thresholds
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
    result = []
    for floor in response["details"]["result"]:
        for item in response["details"]["result"][floor]:
            result += response["details"]["result"][floor][item]
    # draw all as gray
    img = draw_result(img, detections, color=(192,192,192), put_percent=True)
    # draw posm and split lines as blue
    low_thr, up_thr = get_thresholds(response["posm_type"], "fridge")
    img = draw_result(img, get_boxes_of_labels(detections, list(range(low_thr, up_thr+1))), color=(255,0,0), put_percent=False, put_label=False)
    
    low_thr, up_thr = get_thresholds(response["posm_type"], "shelf")
    img = draw_result(img, get_boxes_of_labels(detections, list(range(low_thr, up_thr+1))), color=(255,0,0), put_percent=False, put_label=False)
    # draw bottles as green
    low_thr, up_thr = get_thresholds(response["posm_type"], "bottle")
    img = draw_result(img, get_boxes_of_labels(result, list(range(low_thr, up_thr+1))), color=(0,255,0), put_percent=True)
    # draw nonspvb as purple
    img = draw_result(img, get_boxes_of_labels(result, [up_thr]), color=(255,0,255), put_percent=True)
    # draw space as red
    img = draw_result(img, get_boxes_of_labels(result, [-1]), color=(0,0,255), put_percent=True)
    img = put_text(img, response["message"], loc=[10,10])
    return img

def get_boxes_of_labels(boxes, labels):
    return [box for box in boxes if box[-1] in labels]

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

def convert_to_list_of_reasons(reasons):
    ret = []
    for reason in reasons:
        if len(reasons[reason]) > 0:
            ret = ret + reasons[reason] if isinstance(reasons[reason], list) else ret + [reasons[reason]]
    return ret

def export_to_xlsx(response):
    file_name = "samples/results/audit_result.xlsx"
    image_names = [os.path.basename(r["image_path"]) for r in response]
    image_out_names = [os.path.basename(r["result_image_path"]) for r in response]
    results = [r["evaluation_result"] for r in response]
    reasons = [convert_to_list_of_reasons(r["reasons"]) for r in response]
    df = pd.DataFrame.from_dict({"image_name": image_names, "image_out_name": image_out_names, "result": results, "reason": reasons})
    df.to_excel(file_name, sheet_name="Sheet1", index=False)
    return file_name

def download_img_from_url(url):
    #data = requests.get(url).content
    file_name = url.split("/")[-1]
    images_folder = os.path.join("samples", "images")
    img_path = os.path.join(images_folder, file_name)
    #with open(img_path, "wb") as f:
    #    f.write(data)
    return img_path