import os, cv2, copy
from .utils import extract_to_image
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
    img_name = request["image_path"]
    img0 = cv2.imread(img_name)
    img = copy.deepcopy(img0)
    img_key = os.path.basename(img_name).split(".")[0]
    
    # if image horizontal or invalid
    response = request
    response["reasons"]["OTHER"] = "PHOTOINVALID: Không tìm thấy sản phẩm của SPVB" if len(response["details"]["detections"])==0 else ""
    if response["reasons"]["OTHER"]=="": boxes, index_dict = get_boxes_and_indices(response)
    if response["reasons"]["OTHER"]=="": boxes, result_dict = handle_too_few_case(boxes, index_dict, response)
    # check skewness
    response["reasons"]["OTHER"] = "PHOTOINVALID: Hình bị xiên, vui lòng chụp chính diện" \
        if check_image_skewness(boxes, index_dict["shelf"], mode="overlap") else ""
    
    if response["reasons"]["OTHER"]== "":
        if response["posm_type"] == "vsc":
            if response["is_one_floor"]:
                response = analyze_for_one_floor(boxes, index_dict, response)
            elif result_dict["is_combo"]:
                response = analyze_for_combo(boxes, index_dict, response)
            else:
                response = analyze_for_normal(boxes, index_dict, response)
        else: # rack
            response = analyze_for_rack(boxes, index_dict, response)
        
    if response["evaluation_result"] == 0:
        if check_image_skewness( boxes, index_dict["shelf"], mode="size" ):
            response["reasons"]["OTHER"] = "PHOTOINVALID: Hình bị xiên, vui lòng chụp chính diện"
            
        # we remove the keyword
        short_ok_status = response["reasons"]["OTHER"].split(":")[-1]  
        response["message"] += f"{short_ok_status}\n"
        
    # Extract to image and json
    img = extract_to_image(img, response)
    if response["evaluation_result"] == 1:
        out_img_path = os.path.join("samples", f"{img_key}_output_ok.jpg")
        
    elif response["evaluation_result"] == 0:
        out_img_path = os.path.join("samples", f"{img_key}_output_notok.jpg")
    cv2.imwrite(out_img_path, img)
    response["result_image_url"] = out_img_path
    print(f"Done for {out_img_path}")
    return response
