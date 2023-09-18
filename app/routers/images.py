from .. import models, schemas
from ..database import get_db
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session
from typing import List

router = APIRouter()

@router.post("/images", status_code=status.HTTP_201_CREATED, response_model=schemas.ImageResponse)
def create_image(image: schemas.ImageCreate, db: Session = Depends(get_db)):
    new_image = models.Image(**image.model_dump())
    db.add(new_image)
    db.commit()
    db.refresh(new_image)
    return new_image

@router.get("/images", response_model = List[schemas.ImageResponse])
def get_images(db: Session = Depends(get_db)):
    images = db.query(models.Image).all()
    return images

@router.get("/images/{id}", response_model = schemas.ImageResponse)
def get_image(id: int, db: Session = Depends(get_db)):
    image = db.query(models.Image).filter(models.Image.image_id == id).first()
    if not image:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Image with id {id} was not found")
    return image

@router.put("/images/{id}", response_model = schemas.ImageResponse)
def update_image(id:int, updated_image: schemas.ImageUpdate, db:Session = Depends(get_db)):
    img_query = db.query(models.Image).filter(models.Image.image_id==id)
    image = img_query.first()
    if image == None:
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail=f"Image with id {id} does not exist")
    img_query.update(updated_image.model_dump(), synchronize_session=False)
    db.commit()
    return img_query.first()

@router.delete("/images/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_post(id: int, db: Session = Depends(get_db)):
    img_query = db.query(models.Image).filter(models.Image.image_id == id)
    
    if img_query.first() == None:
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail=f"Image with id {id} does not exist")
    img_query.delete(synchronize_session=False)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)