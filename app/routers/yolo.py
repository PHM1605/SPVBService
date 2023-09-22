from .. import oauth2
from ..schemas import YoloRequest, YoloResponse
from fastapi import APIRouter, Depends, status
from yolo_extract import extract_detection

router = APIRouter()

@router.get("/folder_extract", status=status.HTTP_200_OK, response_model=YoloResponse)
def get_json(img_folder: YoloRequest, current_user = Depends(oauth2.get_current_user)):
    response = extract_detection.extract(img_folder.model_dump())
    return {"list_detection": response}
