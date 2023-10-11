import os
from .. import oauth2
from ..schemas import YoloRequest, YoloResponse
from fastapi import APIRouter, Depends, status
from ml_utils.evaluation_spvb import evaluate
from yolo_extract import extract_detection
from yolo_extract.clean import clean_output_yolo

router = APIRouter()

@router.get("/folder_extract", status_code=status.HTTP_200_OK, response_model=YoloResponse)
def get_json(img_folder: YoloRequest, current_user = Depends(oauth2.get_current_user)):
#def get_json(img_folder: YoloRequest):
    print("AAAAA")
    response = extract_detection.extract(img_folder.model_dump())

    # ------------- to export in new format --------------------------
    detections_only = clean_output_yolo(response)
    for det in detections_only:
        with open(os.path.join("./samples/results", det["file_name"]), 'w') as f:
            for box in det["detections"]:
                box = str(box)[1:-1].replace(",", "") + "\n"
                f.write(box) 
    response = detections_only
    # ----------------------------------------------------------------
    return {"list_detection": response}
