from sqlalchemy import Column, JSON, Integer
from sqlalchemy.sql.expression import text
from sqlalchemy.sql.sqltypes import TIMESTAMP
from .database import Base

class ImageTable(Base):
    __tablename__ = "minhpham_image_result"
    classes = Column(JSON)
    consider_full_posm = Column(Integer, default=0)
    consider_last_floor = Column(Integer, default=1)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('now()'))
    id_image_result = Column(Integer, )