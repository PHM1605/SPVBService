import os, json, datetime, cv2, copy, glob
import numpy as np
import pandas as pd

from utils import (
    extract_to_image,
    get_result_dict
)
from models import BoundingBox, read_template, extract_template
from analysis import (ls
    analyze_for_one_floor,
    analyze_for_combo,
    analyze_for_normal,
    analyze_for_rack,
    check_image_skewness,
    get_boxes_and_indices,
    handle_too_few_case
)
from indices import get_indices

# filter out the list of bounding boxes x1y1x2y2 of a specific label
class SPVBApp:
    def __init__(self, data_folder, class_file, dict_file=None, audit_file=None):
        self.data_folder = data_folder
        self.img_folder = os.path.join(self.data_folder, "images")
        self.img_list = []
        for extension in ["*.jpg", "*.png", "*.jpeg"]:
            self.img_list.extend(glob.glob(os.path.join(self.img_folder, extension)))
        self.result_folder = os.path.join(self.data_folder, "results")
        self.audit_file = audit_file
        self.class_file = class_file
        
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
            
        """ Read .json file to get the yolo result of all photos """
        self.dict_file = dict_file
        with open(os.path.join(self.data_folder, dict_file), "r", encoding="utf-8-sig") as f:
            self.yolo_dict = json.load(f)            
        # which class belongs to which index
        with open(self.class_file, "r") as f:
            classes = f.readlines()
        self.classes = [cl.replace("\n", "") for cl in classes]
        if self.audit_file is not None:     
            self.audit_file = os.path.join(self.data_folder, audit_file)
            # Auto find column index
            index_col_name = 'newimagename'
            tmp_template = pd.read_excel(self.audit_file, sheet_name="Sheet1", header=0, index_col=None)
            cols = list(tmp_template.columns)
            self.index_col = cols.index(index_col_name)
            self.template = read_template(self.audit_file, index_col=self.index_col)

    def analyze_all_images(self, yolo_dict):
        output_json = []
        for img_name in self.img_list:
            result_dict_one_image = get_result_dict(img_name, yolo_dict)
            if result_dict_one_image is not None:
                result_dict = self.analyze_one_image(img_name, result_dict_one_image)
                output_json.append(result_dict)
                
        # Export all to json
        json_str = json.dumps(output_json, ensure_ascii=False)
        with open(
            os.path.join(self.result_folder, 'output_result.json'), "w", encoding="utf-8"
        ) as f:
            f.write(json_str)
        
    def analyze_one_image(self, img_name, result_dict):
        # Read image and get image name as key
        img0 = cv2.imread(img_name)
        img = copy.deepcopy(img0)
        img_key = os.path.basename(img_name).split(".")[0]
        
        # if image horizontal or invalid
        result_dict["reasons"]["OTHER"] = "PHOTOINVALID: Không tìm thấy sản phẩm của SPVB" if len(result_dict["details"]["detections"])==0 else ""
        if result_dict["reasons"]["OTHER"]=="": boxes, index_dict = get_boxes_and_indices(result_dict)
        if result_dict["reasons"]["OTHER"]=="": boxes, result_dict = handle_too_few_case(boxes, index_dict, result_dict)
        # check skewness
        result_dict["reasons"]["OTHER"] = "PHOTOINVALID: Hình bị xiên, vui lòng chụp chính diện" \
            if check_image_skewness(boxes, index_dict["shelf"], mode="overlap") else ""
        
        if result_dict["reasons"]["OTHER"]== "":
            if result_dict["posm_type"] == "vsc":
                if result_dict["is_one_floor"]:
                    result_dict = analyze_for_one_floor(boxes, index_dict, result_dict)
                elif result_dict["is_combo"]:
                    result_dict = analyze_for_combo(boxes, index_dict, result_dict)
                else:
                    result_dict = analyze_for_normal(boxes, index_dict, result_dict)
            else: # rack
                result_dict = analyze_for_rack(boxes, index_dict, result_dict)
            
        if result_dict["evaluation_result"] == 0:
            if check_image_skewness( boxes, index_dict["shelf"], mode="size" ):
                result_dict["reasons"]["OTHER"] = "PHOTOINVALID: Hình bị xiên, vui lòng chụp chính diện"
                
            # we remove the keyword
            short_ok_status = result_dict["reasons"]["OTHER"].split(":")[-1]  
            result_dict["message"] += f"{short_ok_status}\n"
            
        # Extract to image and json
        img = extract_to_image(img, result_dict)
        if result_dict["evaluation_result"] == 1:
            out_img_path = os.path.join(self.result_folder, f"{img_key}_output_ok.jpg")
            
        elif result_dict["evaluation_result"] == 0:
            out_img_path = os.path.join(self.result_folder, f"{img_key}_output_notok.jpg")
        cv2.imwrite(out_img_path, img)
        result_dict["result_image_url"] = out_img_path
        
        print(f"Done for {img_name}")
        return result_dict

if __name__ == "__main__":
    app = SPVBApp(
        data_folder="samples",
        dict_file='result_dict_2.json',
        audit_file=None,
        class_file="rackclass.txt"
    )
    app.analyze_all_images(app.yolo_dict)
    if app.audit_file is not None:
        extract_template(app.template, app.result_folder, app.audit_file)
