from fastapi import FastAPI
from fastapi.params import Body
from pydantic import BaseModel

app = FastAPI()

# class Image(BaseModel):
#     "tenant_id": str
#     ""



@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/images")
def create_images(payload: dict = Body(...)):
    
    return payload