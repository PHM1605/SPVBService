from ..models import ImageResult
from fastapi import APIRouter
from typing import List
from ml_utils import evaluation_spvb, utils

router = APIRouter()

@router.get("/images_results")
async def get_results(request: List[ImageResult]):
    response = []
    for i, one_img_request in enumerate(request):
        one_img_response = evaluation_spvb.evaluate(one_img_request.model_dump())
        response.append(one_img_response)
    file_name = utils.export_to_xlsx(response)
    return f"Export to {file_name}"

@router.get("/image_results")
async def get_results(request: ImageResult):
    response = evaluation_spvb.evaluate(request.model_dump())
    return response