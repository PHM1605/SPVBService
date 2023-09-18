from datetime import datetime
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String
from sqlalchemy.sql.expression import text
from sqlalchemy.sql.sqltypes import TIMESTAMP
from typing import Dict, List, Union, Optional
from .database import Base

class Image(Base):
    __tablename__ = "images"
    image_id = Column(Integer, primary_key=True, nullable=False)
    name = Column(String, nullable=False)
    is_combo = Column(Integer, server_default='-1')
    image_url = Column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('now()'))

class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, nullable=False)
    email = Column(String, nullable=False, unique=True)
    password = Column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('now()'))

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

class YoloRequest(BaseModel):
    img_dir: str # '../samples/images/'
    img_size: int # 640
    extensions: List[str] # ["*.jpg", "*.png", "*.jpeg"]
    model: str # 'rack0821.pt'

class DetectionRequest(BaseModel):
    json_path: str