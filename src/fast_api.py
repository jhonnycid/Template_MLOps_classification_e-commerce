from fastapi import FastAPI, Request, HTTPException, Depends, Header, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import os
import json
import pickle
import base64
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import logging
import mlflow
import mlflow.pytorch
from torchvision import transforms
from dotenv import load_dotenv

from models.train_model import LSTMModel, VGG16Model, TextPreprocessor

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialisation de l'app FastAPI
app = FastAPI(
    title="Rakuten Prediction API",
    description="API de prédiction avec authentification par clé API",
    version="1.0.0",
    openapi_tags=[
        {"name": "Prediction", "description": "Endpoints de prédiction de produits"},
    ]
)

# Ajouter schéma d'authentification pour Swagger
API_KEY = os.environ.get("API_KEY", "secret123")
API_KEY_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": API_KEY_NAME
        }
    }
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method.setdefault("security", [{"APIKeyHeader": []}])
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Autoriser CORS si besoin (optionnel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Clé API invalide")

# Variables globales
text_preprocessor = None
lstm_model = None
vgg16_model = None
best_weights = None
category_mapper = None
image_transform = None

class PredictInput(BaseModel):
    description: str
    image: str

def preprocess_text(text):
    import re
    from bs4 import BeautifulSoup
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text.lower().strip()

def preprocess_image(image_data):
    try:
        if isinstance(image_data, str) and os.path.exists(image_data):
            img = Image.open(image_data).convert('RGB')
        else:
            img_data = base64.b64decode(image_data + "==")
            img = Image.open(BytesIO(img_data)).convert('RGB')
        return image_transform(img)
    except (base64.binascii.Error, UnidentifiedImageError, Exception) as e:
        logger.error(f"Erreur image: {e}")
        return torch.zeros((3, 224, 224))

@app.on_event("startup")
def startup_event():
    global text_preprocessor, lstm_model, vgg16_model, best_weights, category_mapper, image_transform
    logger.info("Chargement des modèles...")

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with open("models/text_preprocessor.pkl", "rb") as f:
        data = pickle.load(f)
    text_preprocessor = TextPreprocessor(max_words=data.get("max_words", 10000))
    text_preprocessor.word_to_idx = data.get("word_to_idx", {})
    text_preprocessor.idx_to_word = data.get("idx_to_word", {})
    text_preprocessor.word_counts = data.get("word_counts", {})
    text_preprocessor.n_words = data.get("n_words", 2)

    with open("models/model_info.json") as f:
        model_info = json.load(f)

    lstm_model = LSTMModel(
        vocab_size=text_preprocessor.n_words,
        embedding_dim=model_info["embedding_dim"],
        hidden_dim=model_info["lstm_units"],
        output_dim=model_info["num_classes"]
    )
    lstm_model.load_state_dict(torch.load("models/best_lstm_model.pth", map_location="cpu"))
    lstm_model.eval()

    vgg16_model = VGG16Model(output_dim=model_info["num_classes"], pretrained=False)
    vgg16_model.load_state_dict(torch.load("models/best_vgg16_model.pth", map_location="cpu"))
    vgg16_model.eval()

    with open("models/best_weights.json") as f:
        best_weights = json.load(f)

    with open("models/mapper.json") as f:
        category_mapper = json.load(f)

@app.get("/health")
def health():
    return {"status": "ok", "backend": "PyTorch"}

@app.post("/predict", tags=["Prediction"])
def predict(
    input_data: PredictInput = Body(...),
    api_key: str = Depends(verify_api_key)
):
    try:
        description = input_data.description
        image_data = input_data.image

        if not description or not image_data:
            raise HTTPException(status_code=400, detail="Champs requis: 'description', 'image'")

        processed_text = preprocess_text(description)
        sequence = text_preprocessor.texts_to_sequences([processed_text], max_len=10)
        text_tensor = torch.tensor(sequence, dtype=torch.long)

        image_tensor = preprocess_image(image_data).unsqueeze(0)

        with torch.no_grad():
            lstm_logits = lstm_model(text_tensor)
            lstm_proba = torch.softmax(lstm_logits, dim=1).numpy()

            vgg_logits = vgg16_model(image_tensor)
            vgg_proba = torch.softmax(vgg_logits, dim=1).numpy()

        combined = best_weights[0] * lstm_proba + best_weights[1] * vgg_proba
        pred_idx = int(np.argmax(combined, axis=1)[0])
        category = category_mapper.get(str(pred_idx), "inconnu")

        if os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
            with mlflow.start_run(run_name="prediction_api"):
                mlflow.log_param("description_length", len(description))
                mlflow.log_metric("prediction_confidence", float(combined[0][pred_idx]))

        return {
            "category": category,
            "category_id": pred_idx,
            "confidence": float(combined[0][pred_idx]),
            "weights": {
                "lstm": best_weights[0],
                "vgg": best_weights[1]
            }
        }

    except Exception as e:
        logger.error(f"Erreur: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.fast_api:app", host="0.0.0.0", port=8000, reload=False)
