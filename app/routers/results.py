import json
from ..models import DetectionRequest, ImageResult
from fastapi import APIRouter, HTTPException, status
from ml_utils import evaluation_spvb, utils

router = APIRouter()

@router.get("/eval_results")
async def get_results(request: DetectionRequest):
    request = request.model_dump()
    try:
        with open(request["json_path"]) as f:
            request = json.load(f)
    except:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File requested {request['json_path']} not found!")
    
    response = []
    for one_img_request in request:
        one_img_response = evaluation_spvb.evaluate(one_img_request)
        response.append(one_img_response)
    file_name = utils.export_to_xlsx(response)
    return f"Export to {file_name}"

@router.get("/image_results")
async def get_results(request: ImageResult):
    response = evaluation_spvb.evaluate(request.model_dump())
    return response