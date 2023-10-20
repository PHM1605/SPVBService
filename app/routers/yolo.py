import os
from .. import oauth2
from ..schemas import YoloFileRequest, YoloFolderRequest, YoloResponse
from fastapi import APIRouter, Depends, status
from ml_utils.evaluation_spvb import evaluate
from yolo_extract import extract_detection
from yolo_extract.clean import clean_output_yolo

router = APIRouter()

@router.get("/folder_extract", status_code=status.HTTP_200_OK, response_model=YoloResponse)
def get_json(img_folder: YoloFolderRequest, current_user = Depends(oauth2.get_current_user)):
    response = extract_detection.extract_from_folder(img_folder.model_dump())

    # # ------------- to export in new format --------------------------
    # detections_only = clean_output_yolo(response)
    # for det in detections_only:
    #     with open(os.path.join("./samples/results", det["file_name"]), 'w') as f:
    #         for box in det["detections"]:
    #             box = str(box)[1:-1].replace(",", "") + "\n"
    #             f.write(box) 
    # response = detections_only
    # ----------------------------------------------------------------
    return {"list_detection": response}

@router.get("/file_extract", status_code=status.HTTP_200_OK, response_model=YoloResponse)
def get_json(xlsx_file: YoloFileRequest, current_user=Depends(oauth2.get_current_user)):
    response = extract_detection.extract_from_file(xlsx_file.model_dump())
    return {"list_detection": response}