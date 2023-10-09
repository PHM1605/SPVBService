from . import models
from .config import settings
from .database import engine
from .routers import auth, images, results, users, yolo
from fastapi import FastAPI

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(auth.router)
app.include_router(images.router)
app.include_router(results.router)
app.include_router(users.router)
app.include_router(yolo.router)