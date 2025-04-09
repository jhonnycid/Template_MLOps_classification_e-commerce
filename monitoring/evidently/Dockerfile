FROM python:3.8-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn pandas evidently jinja2

COPY app.py .

EXPOSE 8050

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8050"]