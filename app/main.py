from datetime import datetime
import psycopg2
from . import models
from .database import engine
from fastapi import FastAPI, HTTPException, Response, status
from ml_utils import evaluation_spvb
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Dict, List, Union, Optional

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

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
    image_path: str
    result_image_path: str 
    is_combo: int = -1
    is_full_posm: int = -1
    is_one_floor: int = -1
    message: Optional[str] = ""
    number_of_floor: int = -1
    posm_type: str
    program_code: str 
    reasons: Reason
    tenant_id: str

try: 
    conn = psycopg2.connect(host='localhost', database="spvb_images", user="root", password="matKH4U12$$", cursor_factory=RealDictCursor)
    cursor = conn.cursor()
    print("Database connection was successful!")
except Exception as error:
    print("Connecting to database failed")
    print("Error: ", error)

@app.post("/images", status_code=status.HTTP_201_CREATED)
def create_images(image: Image):
    return {"data": image}

@app.get("/images")
def get_images(image: Image):
    pass

@app.get("/results")
def get_results(image: Image):
    result = evaluation_spvb.evaluate(image.model_dump())
    return result

@app.get("/images/{id}")
def get_image(id:int, image: Image):
    if not image: 
       raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Image with id {id} was not found")

    return {"image_detail": image}

@app.delete("/images/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_image(id: int):
    if False:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Post with id {id} does not exist")
    return Response(status_code = status.HTTP_204_NO_CONTENT)

@app.patch("images/{id}") 
def update_image(id: int, image):
    if False:
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail=f"Image with id {id} does not exist")
    image_dict = image.model_dump()
    
    return {"message": "Updated image"}