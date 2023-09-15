import time
import mysql.connector
from . import models, schemas
from .database import engine, SessionLocal
from datetime import datetime
from fastapi import Depends, FastAPI, HTTPException, Response, status
from ml_utils import evaluation_spvb
from ml_utils.utils import export_to_xlsx
from passlib.context import CryptContext
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from sqlalchemy.orm import Session
from typing import Dict, List, Union, Optional

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class Details(BaseModel):
    detections: List[List[Union[int, float]]]
    result: Dict[str, Dict[str, List[List[Union[int, float]]]]]

class Reason(BaseModel):
    NON_SPVB: List[str]
    SPACE: List[str]
    OTHER: str


class ImageResult(BaseModel):
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

while True:
    try: 
        conn = mysql.connector.connect(host="localhost", user="root", passwd="matKH4U12$$", database="spvb_images", port=3306)
        cursor = conn.cursor(dictionary=True)
        print("Database connection was successful!")
        break
    except Exception as error:
        print("Connecting to database failed")
        print("Error: ", error)
        time.sleep(2)

@app.post("/images", status_code=status.HTTP_201_CREATED, response_model=schemas.ImageResponse)
def create_image(image: schemas.ImageCreate, db: Session = Depends(get_db)):
    new_image = models.Image(**image.model_dump())
    db.add(new_image)
    db.commit()
    db.refresh(new_image)
    return new_image

@app.get("/images", response_model = List[schemas.ImageResponse])
def get_images(db: Session = Depends(get_db)):
    images = db.query(models.Image).all()
    return images

@app.get("/images/{id}", response_model = schemas.ImageResponse)
def get_image(id: int, db: Session = Depends(get_db)):
    image = db.query(models.Image).filter(models.Image.image_id == id).first()
    if not image:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Image with id {id} was not found")
    return image

@app.put("/images/{id}", response_model = schemas.ImageResponse)
def update_image(id:int, updated_image: schemas.ImageUpdate, db:Session = Depends(get_db)):
    img_query = db.query(models.Image).filter(models.Image.image_id==id)
    image = img_query.first()
    if image == None:
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail=f"Image with id {id} does not exist")
    img_query.update(updated_image.model_dump(), synchronize_session=False)
    db.commit()
    return img_query.first()

@app.delete("/images/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_post(id: int, db: Session = Depends(get_db)):
    img_query = db.query(models.Image).filter(models.Image.image_id == id)
    
    if img_query.first() == None:
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail=f"Image with id {id} does not exist")
    img_query.delete(synchronize_session=False)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.get("/folder_results")
async def get_results(request: List[ImageResult]):
    response = []
    for i, one_img_request in enumerate(request):
        one_img_response = evaluation_spvb.evaluate(one_img_request.model_dump())
        response.append(one_img_response)
    file_name = export_to_xlsx(response)
    return f"Export to {file_name}"

@app.get("/image_results")
async def get_results(request: ImageResult):
    response = evaluation_spvb.evaluate(request.model_dump())
    return response

@app.post("/users", status_code = status.HTTP_201_CREATED, response_model=schemas.UserResponse)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    hashed_password = pwd_context.hash(user.password)
    user.password = hashed_password

    new_user = models.User(**user.model_dump())
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user