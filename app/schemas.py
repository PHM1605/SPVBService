import uuid
from datetime import datetime
from pydantic import BaseModel, EmailStr

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

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    user_id: int
    email: EmailStr
    created_at: datetime
    class Config:
        from_attributes = True