# newest version 16.06.23: thêm thông báo số tầng phát hiện khi lỗi; chỉnh tủ combo: nếu đủ tủ sẽ cộng tầng.
import cv2, copy, itertools, scipy, math
from .models import BoundingBox
import numpy as np
from .indices import get_indices

""" Add more floor for types of fridge"""
def add_floor_for_one_floor(boxes, index_dict, img):
    # when the fridge only has 1 floor
    def get_left_right_border(boxes, index_dict, img):
        img = copy.deepcopy(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bottles = [boxes[idx] for idx in index_dict["bottle"]]
        bottles.sort(key=lambda box: box.x1)  # sort from right to left
        fridge = boxes[index_dict["fridge"][0]]
        line_left = gray[bottles[0].cen_y, fridge.x1 : (bottles[0].x1)]
        line_right = gray[bottles[0].cen_y, bottles[-1].x2 : fridge.x2]
        # lowpass filter to smooth the signal
        b, a = scipy.signal.butter(4, 0.2, "low")
        try:
            line_left = scipy.signal.filtfilt(b, a, line_left)
        except:
            pass
        try:
            line_right = scipy.signal.filtfilt(b, a, line_right)
        except:
            pass
        peaks_left, _ = scipy.signal.find_peaks(line_left, prominence=40)
        peaks_right, _ = scipy.signal.find_peaks(line_right, prominence=40)
        if len(peaks_left) > 0:
            # loc = list(peaks_left).index(max(peaks_left))
            loc = -1
            x1 = fridge.x1 + peaks_left[loc]
        else:
            x1 = fridge.x1
        if len(peaks_right) > 0:
            # loc = list(line_right[peaks_right]).index(max(line_right[peaks_right]))
            loc = 0
            x2 = bottles[-1].x2 + peaks_right[loc]
        else:
            x2 = fridge.x2
        return x1, x2
    # add lowest floor - only one
    bottle = boxes[index_dict["bottle"][0]]
    fridge = boxes[index_dict["fridge"][0]]
    x1, x2 = get_left_right_border(boxes, index_dict, img)
    added_box = BoundingBox(x1, int(fridge.y2 - bottle.h), x2, fridge.y2, 1.0, -1) # -1 means not yet in classes.txt file
    boxes.append(added_box)
    index_dict["shelf"].append(len(boxes) - 1)
    return boxes, index_dict

def add_floor_for_normal(boxes, index_dict, img, result_dict):
    # we only add when the new shelf not overlap and distance between the new and the old is big enough
    def should_add(box, added_box):
        if (
            calculate_overlap(box, added_box) < 0.1
            and (added_box.y1 - box.y2) > boxes[index_dict["bottle"][-1]].h / 2
        ):
            return True
        else:
            return False
    # Add box as bottle(s) is under the last shelf
    lowest_loc = max([boxes[idx].y2 for idx in index_dict["bottle"]])
    if lowest_loc > boxes[index_dict["shelf"][-1]].y2:
        box = boxes[index_dict["shelf"][-1]]
        added_box = BoundingBox(box.x1, int(lowest_loc - box.h / 2), box.x2, int(lowest_loc + box.h / 2), box.prob, box.label)
        #if should_add(box, added_box):
        boxes.append(added_box)
        index_dict["shelf"].append(len(boxes) - 1)
        return boxes, index_dict

    if len(index_dict["fridge"]) > 0:
        # if the gap between the lowest shelf and the fridge too big
        if (
            boxes[index_dict["fridge"][0]].y2 - boxes[index_dict["shelf"][-1]].cen_y
        ) > boxes[index_dict["bottle"][-1]].h:
            box = boxes[index_dict["shelf"][-1]]
            added_box = BoundingBox(box.x1, int(boxes[index_dict["fridge"][0]].y2 - box.h), box.x2, int(boxes[index_dict["fridge"][0]].y2), box.prob, box.label)
            if should_add(box, added_box):
                boxes.append(added_box)
                index_dict["shelf"].append(len(boxes) - 1)
        # remove shelf if shelf is outside the fridge
        index_dict["shelf"] = [
            idx
            for idx in index_dict["shelf"]
            if calculate_overlap(boxes[idx], boxes[index_dict["fridge"][0]]) >= 0.3
        ]
    return boxes, index_dict

def add_floor_for_combo(boxes, index_dict):
    # we add when the new shelf not overlap and distance between the new and the old is big enough
    def should_add(box, added_box):
        if (
            calculate_overlap(box, added_box) < 0.1
            and (added_box.y1 - box.y2) > boxes[index_dict["bottle"][-1]].h / 5
        ):
            return True
        else:
            return False

    # add box as bottle(s) is under the last shelf
    lowest_loc = max([boxes[idx].y2 for idx in index_dict["bottle"]])
    if lowest_loc > boxes[index_dict["shelf"][-1]].y2:
        box = boxes[index_dict["shelf"][-1]]
        added_box = BoundingBox(box.x1, int(lowest_loc - box.h / 2), box.x2, int(lowest_loc + box.h / 2), box.prob, box.label)
        #if should_add(box, added_box):
        boxes.append(added_box)
        index_dict["shelf"].append(len(boxes) - 1)
        return boxes, index_dict
    
    if len(index_dict["fridge"]) > 0:
        # if the gap between the lowest shelf and the fridge too big
        if (
            boxes[index_dict["fridge"][0]].y2 - boxes[index_dict["shelf"][-1]].cen_y
        ) > boxes[index_dict["bottle"][-1]].h:
            box = boxes[index_dict["shelf"][-1]]
            added_box = BoundingBox(box.x1, int(boxes[index_dict["fridge"][0]].y2 - box.h), box.x2, int(boxes[index_dict["fridge"][0]].y2), box.prob, box.label)
            if should_add(box, added_box):
                boxes.append(added_box)
                index_dict["shelf"].append(len(boxes) - 1)
        # remove shelf if shelf is outside the fridge
        index_dict["shelf"] = [
            idx
            for idx in index_dict["shelf"]
            if calculate_overlap(boxes[idx], boxes[index_dict["fridge"][0]]) >= 0.3
        ]
    return boxes, index_dict

""" Analysis part """
def analyze_for_one_floor(boxes, index_dict, img, result_dict):
    boxes, index_dict = add_floor_for_one_floor(boxes, index_dict, img)
    if len(index_dict["shelf"]) != 1:
        result_dict["issue"] = f"LACKOFFLOOR: Tủ 1 tầng nhưng phát hiện {len(index_dict['shelf'])} tầng.\nVui lòng liên hệ NVBH hoặc tổng đài hỗ trợ 18001250"
        result_dict["evaluation_result"] = 0
        return boxes, index_dict, result_dict
    
    if result_dict["issue"] == "":
        list_bottles = assign_shelves(boxes, index_dict, result_dict)
        list_missing, list_nonspvb = get_list_missing_non_spvb(list_bottles, index_dict, result_dict)
        result_dict = update_statistics(list_missing, list_nonspvb)
    return boxes, index_dict, result_dict

def analyze_for_normal(boxes, index_dict, img, result_dict):
    boxes, index_dict = add_floor_for_normal(boxes, index_dict, img, result_dict)
    if len(index_dict["shelf"]) == result_dict["number_of_floor"]:
        if not result_dict["consider_last_shelf"]:
            index_dict["shelf_excluded"] = [index_dict["shelf"][-1]]
            index_dict = remove_low_boxes(boxes, index_dict, index_dict["shelf"][-2])
        else:
            index_dict = remove_low_boxes(boxes, index_dict, index_dict["shelf"][-1])
    elif len(index_dict["shelf"]) == result_dict["number_of_floor"]-1:
        index_dict = remove_low_boxes(boxes, index_dict, index_dict["shelf"][-1])
    else:
        if "shelf_excluded" in index_dict.keys():
            result_dict["issue"] = f"LACKOFFLOOR: Tủ {result_dict['number_of_floor']} tầng nhưng phát hiện {len(index_dict['shelf']) + 1} tầng.\nVui lòng liên hệ NVBH hoặc tổng đài hỗ trợ 18001250"
        else:
            result_dict["issue"] = f"LACKOFFLOOR: Tủ {result_dict['number_of_floor']} tầng nhưng phát hiện {len(index_dict['shelf'])} tầng.\nVui lòng liên hệ NVBH hoặc tổng đài hỗ trợ 18001250"
        result_dict["evaluation_result"] = 0
    
    if result_dict["issue"] == "":
        list_bottles = assign_shelves(boxes, index_dict, result_dict)
        list_missing, list_nonspvb = get_list_missing_non_spvb(list_bottles, index_dict, result_dict)
        result_dict = update_statistics(list_missing, list_nonspvb, result_dict)
    return result_dict
    
def analyze_for_combo(boxes, index_dict, result_dict):
    boxes, index_dict = add_floor_for_combo(boxes, index_dict)
    if len(index_dict["shelf"]) >= 3:
        index_dict = remove_low_boxes(boxes, index_dict, index_dict["shelf"][-1])
    else:
        result_dict["issue"] = f"LACKOFFLOOR: Tủ {master_num_floor-1} tầng + 1 ngăn combo, phát hiện {len(index_dict['shelf'])} tầng.\nVui lòng liên hệ NVBH hoặc tổng đài hỗ trợ 18001250"
        result_dict["evaluation_result"] = 0
    list_bottles = assign_shelves(boxes, index_dict["shelf"], index_dict["bottle"], distance_threshold=1)
    if result_dict["issue"] == "":
        list_bottles = assign_shelves(boxes, index_dict, result_dict)
        list_missing, list_nonspvb = get_list_missing_non_spvb(list_bottles, index_dict, result_dict)
        result_dict = update_statistics(list_missing, list_nonspvb, result_dict)
    return result_dict

def analyze_for_rack(boxes, index_dict, result_dict):
    list_bottles = assign_shelves(boxes, index_dict, result_dict)
    list_missing, list_nonspvb = get_list_missing_nonspvb(list_bottles, boxes, index_dict, result_dict)
    result_dict = update_statistics(list_bottles, list_missing, list_nonspvb, result_dict)
    return result_dict

""" Functions for use in all type of fridge"""
def get_boxes_and_indices(result_dict):
    boxes = []
    for box in result_dict["details"]["detections"]:
        boxes.append( BoundingBox(box[0], box[1], box[2], box[3], box[4], box[-1]) )
    index_dict = get_indices(boxes, result_dict["posm_type"])
    boxes, index_dict = preprocessing_boxes(boxes, index_dict, posm_type = result_dict["posm_type"])
    return boxes, index_dict

def handle_too_few_case(boxes, index_dict, result_dict):    
    if len(index_dict["bottle"]) == 0:
        result_dict["issue"] = "PHOTOINVALID: Không tìm thấy sản phẩm của SPVB"
        return boxes, result_dict
        
    if len(index_dict["shelf"]) < 2:
        if result_dict["posm_type"] == "rack":
            result_dict["issue"] = "PHOTOINVALID: Không tìm thấy sản phẩm của SPVB"
            result_dict["evaluation_result"] = 0
            return boxes, result_dict
        else:
            if result_dict["classes"][boxes[index_dict["fridge"][0]].label] == "POSM_VSC_1F":
                result_dict["is_one_floor"] = 1
            else:
                result_dict["issue"] = "PHOTOINVALID: Không tìm thấy sản phẩm của SPVB"
                result_dict["evaluation_result"] = 0
                return boxes, result_dict
            
    # here we are sure fridge is visicooler
    if len(index_dict["fridge"]) == 0:
        result_dict["is_full_posm"] = 0
        if result_dict["is_one_floor"] or result_dict["consider_full_fridge"]:
            result_dict["issue"] = "PHOTOINVALID: Không nhận dạng đủ 4 cạnh Tủ lạnh. Vui lòng chụp lại."
            result_dict["evaluation_result"] = 0
    else:
        result_dict["is_full_posm"] = 1

    return boxes, result_dict

def preprocessing_boxes(boxes, index_dict, posm_type):
    boxes = remove_overlap_boxes(
        boxes, exclude_indices=index_dict["fridge"] + index_dict["bottle"]
    )  # remove overlap shelves
    index_dict = get_indices(boxes, posm_type)
    boxes = remove_overlap_boxes(
        boxes, exclude_indices=index_dict["fridge"] + index_dict["shelf"]
    )  # remove overlap bottles
    index_dict = get_indices(boxes, posm_type)
    boxes = remove_boxes_outside_fridge(boxes, index_dict)
    index_dict = get_indices(boxes, posm_type)
    index_dict["shelf"] = sort_upper_to_lower(boxes, index_dict["shelf"])
    index_dict["bottle"] = sort_upper_to_lower(boxes, index_dict["bottle"])
    return boxes, index_dict

# which bottles belong to which shelves
def assign_shelves(boxes, index_dict, result_dict):
    ret = {}
    distance_threshold = 1
    flag = [True for _ in range(len(index_dict["bottle"]))]
    for si, shelf_idx in enumerate(index_dict["shelf"]):
        # Main layer lies directly on a shelf; sublayer is only to check if there is Non-SPVB product
        ret[f"F{si + 1}"] = {"Main layer": [], "Sublayer": []}
        for bi, bottle_idx in enumerate(index_dict["bottle"]):
            # If bottom of bottle is on a shelf and top of bottle is above the shelf
            if boxes[shelf_idx].y2 >= boxes[bottle_idx].y2 and flag[bi]:
                if (
                    boxes[shelf_idx].y1 - boxes[bottle_idx].y2
                    < boxes[bottle_idx].h / distance_threshold
                ):
                    ret[f"F{si + 1}"]["Main layer"].append(boxes[bottle_idx])
                    flag[bi] = False
                else:
                    ret[f"F{si + 1}"]["Sublayer"].append(boxes[bottle_idx])
                    flag[bi] = False
    return ret

def get_list_missing_nonspvb(list_bottles, boxes, index_dict, result_dict):
    list_missing, list_nonspvb = {}, {}
    for shelf_idx in range(len(index_dict["shelf"])):
        list_bottles_one_shelf = list_bottles[f"F{shelf_idx + 1}"]
        list_bottles_one_shelf["Main layer"].sort(key=lambda box: box.x1)
        shelf = boxes[index_dict["shelf"][shelf_idx]]
        
        if result_dict["posm_type"] == "vscooler":
            flag = False if result_dict["consider_last_shelf"] and index_dict["shelf"][shelf_idx] == index_dict["shelf"][-1] else True
        else:
            flag = True
        missing, nonspvb = check_missing_nonspvb_oneshelf(list_bottles_one_shelf, 
                                                           shelf, 
                                                           boxes[index_dict["fridge"][0]] if len(index_dict["fridge"]) > 0 else None,
                                                           is_missing_analyze = flag, 
                                                           result_dict=result_dict)
        floor = f"F{shelf_idx + 1}"
        if len(missing) > 0: list_missing.update({floor: missing})
        if len(nonspvb) > 0: list_nonspvb.update({floor: nonspvb})
    return list_missing, list_nonspvb


def update_statistics(list_bottles, list_missing, list_nonspvb, result_dict):
    """
    Args:
        list_bottles: {'F1':{'Main layer':[], 'Sublayer':[]}, 'F2':{'Main layer':[], 'Sublayer':[]}}
        list_missing, list_nonspvb:  {'FX': [box, box], 'FY': [box]}
    """
    for floor in list_bottles:
        list_bottles_one_shelf_all = list_bottles[floor]['Main layer'] + list_bottles[floor]['Sublayer']
        list_labels = np.unique([box.label for box in list_bottles_one_shelf_all])
        result_dict["details"]["result"][floor] = {}
         # update details of list of bottles according to products
        for lab in list_labels:
            bboxes = [[bottle.x1, bottle.y1, bottle.x2, bottle.y2, bottle.prob, bottle.label] for bottle in list_bottles_one_shelf_all if bottle.label==lab]
            result_dict["details"]["result"][floor].update( {result_dict["classes"][lab]: bboxes} )
        
    # update nonspvb and missing standards
    msg1  = ""
    for floor in list_missing:
        result_dict["evaluation_result"] = 0
        result_dict["details"]["result"][floor].update({"SPACE": [[box.x1, box.y1, box.x2, box.y2, box.prob, box.label] for box in list_missing[floor]]})
        result_dict["reasons"]["SPACE"].append(f"SPACE: Trưng bày có khoảng trống ở tầng {floor}")
        if msg1 == "":
            msg1 = f"Trưng bày có khoảng trống ở tầng {floor[-1]}"
        else:
            msg1 += f",{floor[-1]}"
    msg2 = ""
    for floor in list_nonspvb:
        result_dict["evaluation_result"] = 0
        result_dict["details"]["result"][floor].update( {"NON_SPVB": [[box.x1, box.y1, box.x2, box.y2, box.prob, box.label] for box in list_nonspvb[floor]]} )
        result_dict["reasons"]["NON_SPVB"].append(f"NON_SPVB: Sản phẩm không phải của SPVB ở tầng {floor}")
        if msg2 == "":
            msg2 = f"Sản phẩm không phải SPVB ở tầng {floor[-1]}"
        else:
            msg2 += f",{floor[-1]}"
    result_dict["message"] = msg1 + "\n" + msg2 + "\n"
    return result_dict
            
# check bottles on one shelf; return list of missing and non-spvb
def check_missing_nonspvb_oneshelf(bottles, shelf, fridge, is_missing_analyze, result_dict):
    list_missing, list_nonspvb = [], []
    for bottle in bottles["Main layer"] + bottles["Sublayer"]:
        if bottle.label == result_dict["classes"].index('NON_SPVB'):  # non spvb
            list_nonspvb.append(bottle)
    if is_missing_analyze:
        if len(bottles["Main layer"]) == 0:  # missing the whole shelf
            list_missing.append(
                BoundingBox(shelf.x1, shelf.y1 - shelf.h, shelf.x2, shelf.y1, 1.0, -1)
            )
            return list_missing, list_nonspvb
        average_width = np.median([bottle.w for bottle in bottles["Main layer"]])
        average_height = np.mean([bottle.h for bottle in bottles["Main layer"]])
        average_y = np.mean([bottle.y1 for bottle in bottles["Main layer"]])
        prev_loc = shelf.x1
        for bottle in bottles["Main layer"]:
            gap = bottle.x1 - prev_loc
            empty_slots = math.floor(gap / average_width)
            if empty_slots >= 1:
                # print(f'Missing {empty_slots} slots')
                for i in range(empty_slots):
                    x_loc = prev_loc + i * average_width
                    list_missing.append(
                        BoundingBox(x_loc, average_y, x_loc + average_width, average_y + average_height, 1.0, -1,)
                    )
            prev_loc = bottle.x2
        # check the final slot of the shelf
        gap = shelf.x2 - prev_loc
        empty_slots = math.floor(gap / average_width)
        if empty_slots >= 1:
            # print(f'Missing {empty_slots} slots')
            for i in range(empty_slots):
                x_loc = prev_loc + i * average_width
                list_missing.append(
                    BoundingBox(x_loc, average_y, x_loc + average_width, average_y + average_height, 1.0, -1)
                )
        # in case of 1-floor fridge, postprocess the purity standard
        if result_dict["posm_type"] == "vscooler" and result_dict['is_one_floor']:
            # remove missing slots at the end of the shelf
            if (len(bottles["Main layer"]) >= 5):  # only remove when there is enough bottles (e.g. 8 bottles)
                list_missing = [
                    missing_slot
                    for i, missing_slot in enumerate(list_missing)
                    if missing_slot.x1 > bottles["Main layer"][0].x1
                    and missing_slot.x2 < bottles["Main layer"][-1].x2
                ]
    
            # if the gap between the shelf and the fridge is too big -> detect a space
            if (fridge.x2 - shelf.x2) > 3 * average_width:
                list_missing.append(
                    BoundingBox(shelf.x2, shelf.y2 - average_height, fridge.x2, shelf.y2, 1.0, -1)
                )
            if (shelf.x1 - fridge.x1) > 3 * average_width:
                list_missing.append(
                    BoundingBox(fridge.x1, shelf.y2 - average_height, shelf.x1, shelf.y2, 1.0, -1)
                )

    return list_missing, list_nonspvb            

def remove_low_boxes(boxes, index_dict, lowest_shelf_idx):
    index_dict["shelf"] = [
        idx
        for idx in index_dict["shelf"]
        if boxes[idx].y2 <= boxes[lowest_shelf_idx].y2
    ]
    index_dict["bottle"] = [
        
        idx
        for idx in index_dict["bottle"]
        if boxes[idx].y2 <= boxes[lowest_shelf_idx].y2
    ]
    return index_dict

# mode is "size" to check skewness via box size ration, "overlap" to check skewness via overlap
def check_image_skewness(boxes, shelf_indices, mode):
    def if_shelf_too_big():
        size_ratio = 2
        for idx in shelf_indices:
            if boxes[idx].w / boxes[idx].h < size_ratio:
                return True
        return False

    def if_shelves_overlap():
        overlap_count = 1
        overlap_threshold = 0.1
        count = 0
        for pair in itertools.combinations(shelf_indices, 2):
            if calculate_overlap(boxes[pair[0]], boxes[pair[1]]) >= overlap_threshold:
                count = count + 1
        return True if count >= overlap_count else False

    return if_shelf_too_big() if mode == "size" else if_shelves_overlap()
    
def remove_overlap_boxes(boxes, exclude_indices):
    flag = [True for _ in boxes]
    thres = 0.5  # new
    for i, box in enumerate(boxes):
        if flag[i] == False:
            continue
        for j, other_box in enumerate(boxes):
            if i in exclude_indices or j in exclude_indices or i == j:
                continue
            overlap_area = calculate_overlap(box, other_box)
            if overlap_area > thres:
                flag[i] = True
                flag[j] = False
    return [boxes[i] for i in range(len(boxes)) if flag[i]]


def remove_boxes_outside_fridge(boxes, index_dict):
    if len(index_dict["fridge"]) > 0:
        removed_indices = []
        for idx in index_dict["bottle"] + index_dict["shelf"]:
            if calculate_overlap(boxes[idx], boxes[index_dict["fridge"][0]]) < 0.2:
                removed_indices.append(idx)

        boxes = [box for i, box in enumerate(boxes) if i not in removed_indices]
    return boxes


def sort_upper_to_lower(boxes, indices):
    list_boxes = [(idx, boxes[idx]) for idx in indices]
    list_boxes.sort(key=lambda shelf: shelf[1].y1)
    indices = [box[0] for box in list_boxes]
    return indices

def calculate_overlap(box1, box2):
    a, b = box1, box2
    dx = min(a.x2, b.x2) - max(a.x1, b.x1)
    dy = min(a.y2, b.y2) - max(a.y1, b.y1)
    min_area = min(a.area, b.area)
    if dx >= 0 and dy >= 0:
        return dx * dy / min_area
    else:
        return 0

