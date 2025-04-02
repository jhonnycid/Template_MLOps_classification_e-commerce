FROM python:3.8-slim

WORKDIR /app

COPY ./data_processing.py ./data_processing.py

COPY ./requirements.txt ./requirements.txt

RUN mkdir -p data

RUN pip install -r requirements.txt