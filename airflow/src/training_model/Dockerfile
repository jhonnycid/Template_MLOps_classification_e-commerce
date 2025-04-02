FROM tensorflow/tensorflow:latest

WORKDIR /app

COPY ./requirements.txt ./requirements.txt

COPY ./train_model_v2.py ./train_model_v2.py

RUN pip install -r requirements.txt

RUN mkdir -p data
