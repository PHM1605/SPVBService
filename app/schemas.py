import uuid
from datetime import datetime
from pydantic import BaseModel, EmailStr
from typing import Dict, List, Optional, Union

""" Image """
class ImageCreate(BaseModel):
    name: str
    is_combo: int
    image_url: str = f"{str(uuid.uuid4())}.jpg"

class ImageUpdate(BaseModel):
    is_combo: int

class ImageResponse(BaseModel):
    name: str
    image_url: str
    created_at: datetime
    is_combo: int
    class Config:
        from_attributes = True

""" JWT """
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[int] = None

""" User """
class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    user_id: int
    email: EmailStr
    created_at: datetime
    class Config:
        from_attributes = True

""" Result """
class Details(BaseModel):
    detections: List[List[Union[int, float]]]
    result: Dict[str, Dict[str, List[List[Union[int, float]]]]]

class Reason(BaseModel):
    NON_SPVB: List[str]
    SPACE: List[str]
    OTHER: str

class ResultRequest(BaseModel):
    json_path: str

class ResultResponse(BaseModel):
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

""" YOLO """
class YoloRequest(BaseModel):
    img_dir: str # '../samples/images/'
    img_size: int # 640
    extensions: List[str] # ["*.jpg", "*.png", "*.jpeg"]
    model: str # 'rack0821.pt'

class YoloResponse(BaseModel):
    list_detection: List[Dict]