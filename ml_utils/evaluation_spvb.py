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
    img_name = os.path.basename(request["image_path"])
    img0 = cv2.imread( request['image_path'])
    img = copy.deepcopy(img0)
    # if image horizontal or invalid
    response = request.copy()
    response["reasons"]["OTHER"] = "PHOTOINVALID: Không tìm thấy sản phẩm của SPVB" if len(response["details"]["detections"])==0 else ""
    if response["reasons"]["OTHER"]=="": boxes, index_dict = get_boxes_and_indices(response)
    if response["reasons"]["OTHER"]=="": boxes, response = handle_too_few_case(boxes, index_dict, response)
    # check skewness
    response["reasons"]["OTHER"] = "PHOTOINVALID: Hình bị xiên, vui lòng chụp chính diện" \
        if check_image_skewness(boxes, index_dict["shelf"], mode="overlap") else ""
    
    if response["reasons"]["OTHER"]== "":
        if response["posm_type"] == "vsc":
            if response["is_one_floor"]:
                response = analyze_for_one_floor(boxes, index_dict, response)
            elif response["is_combo"]:
                response = analyze_for_combo(boxes, index_dict, response)
            else:
                response = analyze_for_normal(boxes, index_dict, response)
        else: # rack
            response = analyze_for_rack(boxes, index_dict, response)
        
    if response["evaluation_result"] == 0:
        if check_image_skewness( boxes, index_dict["shelf"], mode="size" ):
            response["reasons"]["OTHER"] = "PHOTOINVALID: Hình bị xiên, vui lòng chụp chính diện"

    if response["reasons"]["OTHER"] != "": 
        # we remove the keyword
        short_ok_status = response["reasons"]["OTHER"].split(":")[-1]  
        response["message"] += f"{short_ok_status}\n"
        
    # Extract to image and json
    img = extract_to_image(img, response)
    response.pop("message")
    response["result_image_path"] = os.path.join("samples/results", img_name)
    cv2.imwrite(response["result_image_path"], img)
    print(f"Done for {response['result_image_path']}")
    return response



