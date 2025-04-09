from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
import numpy as np
import json
import jwt
from datetime import datetime, timedelta
import os
from PIL import Image
import io
import time
import requests
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import threading
import traceback

# Métriques Prometheus
PREDICTION_COUNTER = Counter('prediction_total', 'Total number of predictions made', ['status'])
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Time spent processing prediction')

# Configuration Evidently
EVIDENTLY_ENABLED = True  # Activer/désactiver l'envoi de données à Evidently
EVIDENTLY_URL = "http://evidently:8050"  # URL du service Evidently dans le réseau Docker
EVIDENTLY_BATCH_SIZE = 100  # Nombre de prédictions à accumuler avant l'envoi
EVIDENTLY_SEND_INTERVAL = 60  # Intervalle en secondes pour l'envoi des données (s'il y en a)

# Variables pour la collecte des données
prediction_data = []
prediction_lock = threading.Lock()  # Verrou pour l'accès concurrent aux données

# Fonction pour collecter les données de prédiction
def collect_prediction_data(description, image_path, prediction, confidence, processing_time):
    if not EVIDENTLY_ENABLED:
        return
    
    # Créer un enregistrement pour cette prédiction
    record = {
        "description": description,
        "image_path": image_path if image_path else "unknown",
        "prediction": int(prediction),
        "confidence": float(confidence),
        "processing_time": float(processing_time),
        "timestamp": time.time()
    }
    
    # Ajouter l'enregistrement à la liste des prédictions
    with prediction_lock:
        prediction_data.append(record)
        
        # Vérifier si nous avons atteint la taille du lot
        if len(prediction_data) >= EVIDENTLY_BATCH_SIZE:
            # Envoyer les données de manière asynchrone
            threading.Thread(target=send_data_to_evidently).start()

# Fonction pour envoyer les données à Evidently
def send_data_to_evidently():
    global prediction_data
    
    # Récupérer les données et vider la liste
    with prediction_lock:
        if not prediction_data:
            return
        
        data_to_send = prediction_data.copy()
        prediction_data = []
    
    try:
        print(f"Envoi de {len(data_to_send)} enregistrements à Evidently...")
        
        # Envoyer les données pour la détection de dérive
        response = requests.post(
            f"{EVIDENTLY_URL}/drift/detect",
            json={"data": data_to_send},
            timeout=10  # Timeout de 10 secondes
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Réponse d'Evidently: Dérive détectée = {result['drift_detected']}, Score = {result['drift_score']}")
        else:
            print(f"Erreur lors de l'envoi des données à Evidently: {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"Exception lors de l'envoi des données à Evidently: {str(e)}")

# Fonction pour envoyer périodiquement les données à Evidently
def periodic_send():
    while True:
        time.sleep(EVIDENTLY_SEND_INTERVAL)
        
        with prediction_lock:
            if prediction_data:
                # Envoyer les données de manière asynchrone
                threading.Thread(target=send_data_to_evidently).start()

# Démarrer la tâche périodique en arrière-plan
if EVIDENTLY_ENABLED:
    threading.Thread(target=periodic_send, daemon=True).start()

# Modèles Pydantic pour la validation des données
class PredictionRequest(BaseModel):
    description: str
    image_base64: Optional[str] = None

class PredictionResponse(BaseModel):
    prediction: int
    category: str
    confidence: float

class Token(BaseModel):
    access_token: str
    token_type: str

# Configuration de l'authentification
SECRET_KEY = "votre_clé_secrète_à_changer_en_production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Fonction pour créer un token JWT
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Fonction pour vérifier le token
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Rakuten Classification API",
    description="API pour la classification de produits e-commerce",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route pour l'authentification
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # En production, vérifiez les informations d'identification dans une base de données
    if form_data.username != "user" or form_data.password != "password":
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Route pour les prédictions
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    description: str = Form(...),
    image: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    start_time = time.time()
    try:
        # Charger le tokenizer, les modèles et les configurations
        models_dir = "/app/models"
        print("Contenu de models_dir:", os.listdir(models_dir))

        try:
            # Charger les poids
            with open(f"{models_dir}/best_weights.json", "r") as json_file:
                best_weights = json.load(json_file)
            
            # Charger le tokenizer et les modèles individuels
            with open(f"{models_dir}/tokenizer_config.json", "r", encoding="utf-8") as json_file:
                tokenizer_config = json_file.read()
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)
            
            lstm = tf.keras.models.load_model(f"{models_dir}/best_lstm_model.h5")
            vgg16 = tf.keras.models.load_model(f"{models_dir}/best_vgg16_model.h5")
            
            # Charger le mapper et l'inverser (indices -> codes)
            with open(f"{models_dir}/mapper.json", "r") as json_file:
                original_mapper = json.load(json_file)
                # Inverser le mapper pour obtenir un mappage index -> code produit
                mapper = {str(v): k for k, v in original_mapper.items()}
                print(f"Mappage inverser: {mapper}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")
        
        # Prétraitement du texte
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        sequences = tokenizer.texts_to_sequences([description])
        padded_sequences = pad_sequences(sequences, maxlen=10, padding="post", truncating="post")
        
        # Prétraitement de l'image
        from tensorflow.keras.applications.vgg16 import preprocess_input
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        img = img.resize((224, 224))
        img_array = np.array(img).astype('float32')
        if img_array.shape[-1] != 3:  # Gestion des images non RGB
            img_array = np.stack((img_array,)*3, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Prédictions des modèles individuels
        print("Prédiction avec le modèle LSTM...")
        lstm_proba = lstm.predict(padded_sequences)
        print("Prédiction avec le modèle VGG16...")
        vgg16_proba = vgg16.predict(img_array)
        
        # Application manuelle de la pondération (remplace le modèle concatenate)
        print(f"Application des poids: LSTM={best_weights[0]}, VGG16={best_weights[1]}")
        concatenate_proba = best_weights[0] * lstm_proba + best_weights[1] * vgg16_proba
        prediction_index = np.argmax(concatenate_proba[0])
        print(f"Index de prédiction: {prediction_index}")
        confidence = float(concatenate_proba[0][prediction_index])
        print(f"Confiance: {confidence}")
        
        # Conversion de l'index en catégorie
        try:
            category = mapper[str(prediction_index)]
            print(f"Catégorie trouvée: {category}")
        except KeyError:
            print(f"Attention: L'index {prediction_index} n'existe pas dans le mapper. Utilisation de la catégorie par défaut.")
            # Utiliser une catégorie par défaut ou la première catégorie disponible
            if mapper:
                category = list(mapper.values())[0]  # Première catégorie disponible
            else:
                category = "Catégorie inconnue"
        
        # Incrémenter le compteur de prédictions réussies
        PREDICTION_COUNTER.labels(status="success").inc()
        
        # Calculer le temps de traitement
        processing_time = time.time() - start_time
        
        # Collecter les données pour Evidently
        threading.Thread(
            target=collect_prediction_data,
            args=(description, image.filename, prediction_index, confidence, processing_time)
        ).start()
        
        response = {
            "prediction": int(prediction_index),
            "category": category,
            "confidence": confidence
        }
        
        # Enregistrer la latence
        PREDICTION_LATENCY.observe(processing_time)
        
        return response
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        print("🔴 Une exception est survenue :")
        print(tb_str)

        # Incrémenter le compteur de prédictions échouées
        PREDICTION_COUNTER.labels(status="error").inc()
        
        # Enregistrer la latence même en cas d'erreur
        PREDICTION_LATENCY.observe(time.time() - start_time)
        
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

# Route pour les métriques Prometheus
@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Route de surveillance (healthcheck)
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Route pour initialiser Evidently avec des données de référence
@app.post("/init-evidently")
async def init_evidently(current_user: str = Depends(get_current_user)):
    """
    Initialise le service Evidently avec les données de prédiction actuelles comme référence.
    """
    try:
        with prediction_lock:
            if not prediction_data or len(prediction_data) < 10:
                raise HTTPException(
                    status_code=400, 
                    detail="Pas assez de données pour initialiser Evidently. Au moins 10 prédictions sont nécessaires."
                )
            
            data_to_send = prediction_data.copy()
        
        # Envoyer les données comme référence à Evidently
        response = requests.post(
            f"{EVIDENTLY_URL}/drift/reference",
            json={"data": data_to_send},
            timeout=10
        )
        
        if response.status_code == 200:
            return {"status": "success", "message": "Evidently initialisé avec succès"}
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Erreur lors de l'initialisation d'Evidently: {response.status_code} - {response.text}"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'initialisation d'Evidently: {str(e)}")

# Documentation API
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Rakuten Classification API",
        "documentation": "/docs",
        "healthcheck": "/health",
        "metrics": "/metrics",
        "evidently_status": "/init-evidently" if EVIDENTLY_ENABLED else "disabled"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", log_level="debug", port=8000)