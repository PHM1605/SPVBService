from ..models import ImageFolderRequest
from fastapi import APIRouter
from yolo_extract import extract_detection

router = APIRouter()

@router.get("/folder_extract")
def get_json(img_folder: ImageFolderRequest):
    response = extract_detection.extract(img_folder.model_dump())
    return response
