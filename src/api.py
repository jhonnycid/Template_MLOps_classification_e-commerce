from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import os
import json
import pickle
import base64
from io import BytesIO
from PIL import Image
import logging
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv
from torchvision import transforms
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

app = Flask(__name__)

# Variables globales pour stocker les modèles et configurations
text_preprocessor = None
lstm_model = None
vgg16_model = None
best_weights = None
category_mapper = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_transform = None

# Fonction de prétraitement du texte
def preprocess_text(text):
    # Fonction simplifiée pour l'API
    import re
    from bs4 import BeautifulSoup
    
    # Supprimer les balises HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Supprimer les caractères non alphabétiques
    text = re.sub(r"[^a-zA-Z]", " ", text)
    
    # Mettre en minuscule et enlever les espaces en trop
    text = text.lower().strip()
    
    return text


# Prétraitement de l'image
def preprocess_image(image_data):
    try:
        # Si c'est un chemin de fichier
        if isinstance(image_data, str) and os.path.exists(image_data):
            img = Image.open(image_data).convert('RGB')
        # Si c'est une image en base64
        else:
            img_data = base64.b64decode(image_data)
            img = Image.open(BytesIO(img_data)).convert('RGB')
        
        # Appliquer les transformations
        img_tensor = image_transform(img)
        return img_tensor
    
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement de l'image: {e}")
        return torch.zeros((3, 224, 224))


@app.before_first_request
def load_models():
    global text_preprocessor, lstm_model, vgg16_model, best_weights, category_mapper, image_transform
    
    logger.info("Chargement des modèles...")
    
    # Définir les transformations d'image
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 1. Charger le préprocesseur de texte
    with open("models/text_preprocessor.pkl", "rb") as f:
        text_preprocessor_data = pickle.load(f)
    
    text_preprocessor = TextPreprocessor(
        max_words=text_preprocessor_data.get("max_words", 10000)
    )
    text_preprocessor.word_to_idx = text_preprocessor_data.get("word_to_idx", {})
    text_preprocessor.idx_to_word = text_preprocessor_data.get("idx_to_word", {})
    text_preprocessor.word_counts = text_preprocessor_data.get("word_counts", {})
    text_preprocessor.n_words = text_preprocessor_data.get("n_words", 2)
    
    # 2. Charger les informations du modèle
    with open("models/model_info.json", "r") as f:
        model_info = json.load(f)
    
    # 3. Créer et charger le modèle LSTM
    lstm_model = LSTMModel(
        vocab_size=text_preprocessor.n_words,
        embedding_dim=model_info["embedding_dim"],
        hidden_dim=model_info["lstm_units"],
        output_dim=model_info["num_classes"]
    )
    lstm_model.load_state_dict(torch.load("models/best_lstm_model.pth", map_location=device))
    lstm_model.to(device)
    lstm_model.eval()
    
    # 4. Créer et charger le modèle VGG16
    vgg16_model = VGG16Model(output_dim=model_info["num_classes"], pretrained=False)
    vgg16_model.load_state_dict(torch.load("models/best_vgg16_model.pth", map_location=device))
    vgg16_model.to(device)
    vgg16_model.eval()
    
    # 5. Charger les poids optimaux
    with open("models/best_weights.json", "r") as f:
        best_weights = json.load(f)
    
    # 6. Charger le mapper de catégories
    with open("models/mapper.json", "r") as f:
        category_mapper = json.load(f)
    
    logger.info("Modèles chargés avec succès")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données de la requête
        data = request.json
        
        if not data or 'description' not in data or 'image' not in data:
            return jsonify({"error": "La requête doit contenir 'description' et 'image'"}), 400
        
        description = data['description']
        image_data = data['image']
        
        # Prétraiter le texte
        processed_text = preprocess_text(description)
        
        # Tokeniser le texte
        sequence = text_preprocessor.texts_to_sequences([processed_text], max_len=10)
        text_tensor = torch.tensor(sequence, dtype=torch.long).to(device)
        
        # Prétraiter l'image
        image_tensor = preprocess_image(image_data).unsqueeze(0).to(device)  # Ajouter dimension batch
        
        # Prédire avec les modèles
        with torch.no_grad():
            # Prédictions LSTM
            lstm_logits = lstm_model(text_tensor)
            lstm_proba = torch.softmax(lstm_logits, dim=1).cpu().numpy()
            
            # Prédictions VGG16
            vgg16_logits = vgg16_model(image_tensor)
            vgg16_proba = torch.softmax(vgg16_logits, dim=1).cpu().numpy()
        
        # Combiner les prédictions
        combined_proba = best_weights[0] * lstm_proba + best_weights[1] * vgg16_proba
        prediction_index = np.argmax(combined_proba, axis=1)[0]
        
        # Obtenir la catégorie prédite
        predicted_category = category_mapper.get(str(prediction_index), "Catégorie inconnue")
        
        # Log des prédictions avec MLflow
        if os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
            with mlflow.start_run(run_name="prediction_api"):
                mlflow.log_param("description_length", len(description))
                mlflow.log_metric("prediction_confidence", float(combined_proba[0][prediction_index]))
                mlflow.log_dict(
                    {
                        "lstm_weight": best_weights[0],
                        "vgg16_weight": best_weights[1],
                        "predicted_category": predicted_category,
                    },
                    "prediction_details.json"
                )
        
        # Construire la réponse
        response = {
            "predicted_category": predicted_category,
            "category_id": int(prediction_index),
            "confidence": float(combined_proba[0][prediction_index]),
            "model_weights": {
                "lstm": best_weights[0],
                "vgg16": best_weights[1]
            }
        }
        
        logger.info(f"Prédiction réussie: {response['predicted_category']} avec {response['confidence']:.2f} de confiance")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "backend": "PyTorch"})


if __name__ == '__main__':
    # Vérifier si le dossier de logs existe
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    # Définir l'URI de tracking MLflow s'il est configuré
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("rakuten-prediction-api")
    
    # Démarrer l'API
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
