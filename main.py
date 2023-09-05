from datetime import datetime
from fastapi import FastAPI
from fastapi.params import Body
from pydantic import BaseModel
from typing import Dict, List, Union
from ml_utils import evaluation_spvb

app = FastAPI()

class Post(BaseModel):
    title: str
    body: str

class Details(BaseModel):
    detections: List[List[Union[int, float]]]
    result: Dict[str, Dict[str, List[List[Union[int, float]]]]]

class Reason(BaseModel):
    NON_SPVB: List[str]
    SPACE: List[str]
    OTHER: str

class Image(BaseModel):
    classes: List[str]
    consider_full_posm: int = 0
    consider_last_floor: int = 1
    created_date: str = str(datetime.now())
    details: Details
    evaluation_result: int 
    image_id: int 
    image_url: str
    result_image_url: str 
    is_combo: int = -1
    is_full_posm: int = -1
    is_one_floor: int = -1
    message: str
    number_of_floor: int = -1
    posm_type: str
    program_code: str 
    reasons: Reason
    tenant_id: str

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/images")
def post_images(image: Image):
    result = evaluation_spvb.evaluate(image.model_dump())
    return result