# which index is of which type of products
def get_thresholds(posm_type, type_obj):
    if posm_type == 'VC':
        if type_obj == 'fridge':
            lower_thr, upper_thr = 84, 90
        elif type_obj == 'shelf':
            lower_thr, upper_thr = 91, 91
        elif type_obj == 'bottle':
            lower_thr, upper_thr = 0, 83
    elif posm_type == 'RACK':
        if type_obj == 'fridge':
            lower_thr, upper_thr = 86, 88
        elif type_obj == 'shelf':
            lower_thr, upper_thr = 85, 85
        elif type_obj == 'bottle':
            lower_thr, upper_thr = 0, 84
    return lower_thr, upper_thr

def get_indices(boxes, posm_type):
    assert posm_type in ['VC', 'RACK'], f"type of {posm_type} is not suitable"
    ret_dict = {}
    
    def get_indices_helper(lower_thr, upper_thr):
        ret = []
        for i, box in enumerate(boxes):
            if box.label >= lower_thr and box.label <= upper_thr:
                ret.append(i)
        return ret

    ret_dict.update({"fridge": get_indices_helper(*get_thresholds(posm_type, 'fridge'))})
    ret_dict.update({"shelf": get_indices_helper(*get_thresholds(posm_type, 'shelf'))})
    ret_dict.update({"bottle": get_indices_helper(*get_thresholds(posm_type, 'bottle'))})
    return ret_dict
