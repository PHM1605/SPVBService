FROM python:3.9.16

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y python3-opencv

COPY . .

CMD uvicorn app.main:app --host 0.0.0.0 --port 8000