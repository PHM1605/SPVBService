import os, glob, json, copy
import openpyxl
import pandas as pd
from openpyxl.utils.cell import get_column_letter
import numpy as np


class BoundingBox:
    def __init__(self, *res):
        self.x1, self.y1, self.x2, self.y2 = (
            int(res[0]),
            int(res[1]),
            int(res[2]),
            int(res[3]),
        )
        self.cen_x = int((self.x1 + self.x2) / 2)
        self.cen_y = int((self.y1 + self.y2) / 2)
        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1
        self.prob = res[4]
        self.label = res[5]
        self.area = self.w * self.h

    def display(self):
        print(f"Bounding box is [{self.x1}, {self.y1}, {self.x2}, {self.y2}]")
        print(f"Label is {self.label}")
        # print(f'Probability is {self.prob}')

def extract_template(template, result_folder, output_file):
    preds_path = glob.glob(os.path.join(result_folder, "*.json"))
    for pred_path in preds_path:
        img_name = os.path.basename(pred_path).split(".")[0]
        img_name = img_name[: img_name.index("_output")]
        with open(pred_path, encoding="utf-8") as f:
            pred = json.load(f)
        reason_str = ""
        for reason in ["NON_SPVB", "SPACE", "OTHERS"]:
            if len(pred["reasons"][reason]) > 0:
                template.loc[img_name, "new_result"] = 0
                reason_str += pred["reasons"][reason][0]
        if reason_str == "":
            template.loc[img_name, "new_result"] = 1
            
        template.loc[img_name, "new_reason"] = reason_str if reason_str != "" else np.nan
        # template.loc[img_name, "is_full_tu"] = 1 if pred["is_full_tu"] else 0
        template.loc[img_name, "new_result_path"] = img_name + '_output_ok' if reason_str=="" else img_name + '_output_notok'
    template.to_excel('samples/results/output_result.xlsx', sheet_name="Sheet1")

def fix_keys_df(old_df):
    new_df = copy.deepcopy(old_df)
    new_df.index = [elem.split(".")[0] for elem in old_df.index]
    return new_df


def read_template(audit_file, index_col):
    template = pd.read_excel(audit_file, sheet_name="Sheet1", index_col=index_col, header=0)
    template = fix_keys_df(template)
    template['new_result'] = np.nan
    template['new_reason'] = np.nan
    template['new_result_path'] = np.nan
    return template
