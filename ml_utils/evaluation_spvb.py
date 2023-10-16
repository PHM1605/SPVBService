import os, cv2, copy
from .utils import extract_to_image
import time
import traceback
from .analysis import (
    analyze_for_one_floor,
    analyze_for_combo,
    analyze_for_normal,
    analyze_for_rack,
    check_image_skewness,
    get_boxes_and_indices,
    handle_too_few_case
)

def evaluate(request):
    img_name = os.path.basename(request['image_path'])
    dir_name = "samples/results"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    img0 = cv2.imread(request["image_path"])
    img = copy.deepcopy(img0)
    
    # if image horizontal or invalid
    response = request.copy()

    if len(response["details"]["detections"])==0:
        response["reasons"]["OTHER"] = "PHOTOINVALID: Không tìm thấy sản phẩm của SPVB"
        response["evaluation_result"] = 0
    else:
        response["reasons"]["OTHER"] = ""

    if response["reasons"]["OTHER"]=="": boxes, index_dict = get_boxes_and_indices(response)
    if response["reasons"]["OTHER"]=="": response = handle_too_few_case(boxes, index_dict, response)
    # check skewness
    if response["reasons"]["OTHER"] == "":
        if check_image_skewness(boxes, index_dict["shelf"], mode="overlap"):
            response["reasons"]["OTHER"] = "PHOTOINVALID: Hình bị xiên, vui lòng chụp chính diện" 
            response["evaluation_result"] = 0
        else:
            response["reasons"]["OTHER"] = "" 
            
    if response["reasons"]["OTHER"]== "":
        if response["posm_type"] == "VC":
            if response["is_one_floor"]==1:
                response = analyze_for_one_floor(boxes, index_dict, img, response)
                #print("Analyze for one floor")
            elif response["is_combo"]==1:
                response = analyze_for_combo(boxes, index_dict, response)
                #print("Analyze for combo")
            else:
                response = analyze_for_normal(boxes, index_dict, response)
                #print("Analyze for normal")
        else: # RACK
            response = analyze_for_rack(boxes, index_dict, response)
            #print("Analyze for rack")
        
    if response["evaluation_result"] == 1:
        if check_image_skewness( boxes, index_dict["shelf"], mode="size" ):
            response["reasons"]["OTHER"] = "PHOTOINVALID: Hình bị xiên, vui lòng chụp chính diện"
            response["evaluation_result"] = 0
            
    # we remove the keyword
    if response["reasons"]["OTHER"] != "":
        short_ok_status = response["reasons"]["OTHER"].split(":")[-1]  
        response["message"] += f"{short_ok_status}\n"
        
    # Extract to image and json
    img = extract_to_image(img, response)
    response.pop("message")
    response["result_image_path"] = os.path.join(dir_name, img_name)
    if response["evaluation_result"] == 1:
        response["result_image_path"] = response["result_image_path"].split('.')[0] + "_output_ok.jpg"
    else:
        response["result_image_path"] = response["result_image_path"].split('.')[0] + "_output_notok.jpg"
    cv2.imwrite(response["result_image_path"], img)
    #print("RESPONSE: ", response)
    print(f"Done for {response['result_image_path']}")
    return response