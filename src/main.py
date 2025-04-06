import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json
import yaml
import os
import pickle
import mlflow
import mlflow.pytorch
from features.build_features import DataImporter, TextPreprocessor, ImagePreprocessor
from models.train_model import TextLSTMModel, ImageVGG16Model, Concatenate, TextPreprocessor as TextTokenizer
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Charger les paramètres depuis params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Configuration de MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
# mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("rakuten-classification")

# Définir le device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Démarrer un run MLflow
with mlflow.start_run() as run:
    # Loguer les paramètres
    mlflow.log_params(params)

    # Importer les données
    data_importer = DataImporter()
    df = data_importer.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = data_importer.split_train_test(df)

    # Prétraitement des textes et des images
    text_preprocessor = TextPreprocessor()
    image_preprocessor = ImagePreprocessor()
    
    text_preprocessor.preprocess_text_in_df(X_train, columns=["description"])
    text_preprocessor.preprocess_text_in_df(X_val, columns=["description"])
    text_preprocessor.preprocess_text_in_df(X_test, columns=["description"])
    
    image_preprocessor.preprocess_images_in_df(X_train)
    image_preprocessor.preprocess_images_in_df(X_val)
    image_preprocessor.preprocess_images_in_df(X_test)

    # Entraînement du modèle LSTM
    print("Entraînement du modèle LSTM...")
    text_lstm_model = TextLSTMModel(
        max_words=params["max_words"],
        max_sequence_length=params["max_sequence_length"],
        embedding_dim=params["embedding_dim"],
        hidden_dim=params["lstm_units"]
    )
    
    lstm_history = text_lstm_model.preprocess_and_fit(
        X_train, y_train, X_val, y_val,
        epochs=params["epochs"],
        batch_size=params["batch_size"]
    )
    
    print("Entraînement du modèle LSTM terminé")

    # Entraînement du modèle VGG16
    print("Entraînement du modèle VGG16...")
    image_vgg16_model = ImageVGG16Model()
    
    vgg_history = image_vgg16_model.preprocess_and_fit(
        X_train, y_train, X_val, y_val,
        epochs=params["epochs"],
        batch_size=params["batch_size"]
    )
    
    print("Entraînement du modèle VGG16 terminé")

    # Charger les modèles entraînés
    lstm_model = text_lstm_model.model
    vgg16_model = image_vgg16_model.model
    
    # Charger le préprocesseur de texte
    text_tokenizer = TextTokenizer.load("models/text_preprocessor.pkl")

    # Optimisation des poids de l'ensemble
    print("Optimisation des poids de l'ensemble...")
    model_concatenate = Concatenate(text_tokenizer, lstm_model, vgg16_model)
    lstm_proba, vgg16_proba, new_y_train = model_concatenate.predict(X_train, y_train)
    best_weights = model_concatenate.optimize(lstm_proba, vgg16_proba, new_y_train)
    print("Optimisation terminée")

    # Sauvegarder les poids optimaux
    with open("models/best_weights.pkl", "wb") as file:
        pickle.dump(best_weights, file)
    
    # Sauvegarder au format JSON pour la lisibilité
    with open("models/best_weights.json", "w") as f:
        json.dump(best_weights, f)

    # Évaluer le modèle final sur l'ensemble de test
    print("Évaluation sur l'ensemble de test...")
    
    # Prétraiter les séquences de texte
    test_sequences = text_tokenizer.texts_to_sequences(X_test["description"], max_len=params["max_sequence_length"])
    test_text_dataset = torch.tensor(test_sequences, dtype=torch.long)
    test_text_loader = DataLoader(test_text_dataset, batch_size=params["batch_size"])
    
    # Transformer les images
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Prédire avec LSTM
    lstm_model.eval()
    lstm_proba_test = []
    
    with torch.no_grad():
        for batch in tqdm(test_text_loader, desc="LSTM predictions"):
            batch = batch.to(device)
            logits = lstm_model(batch)
            probs = torch.softmax(logits, dim=1)
            lstm_proba_test.append(probs.cpu().numpy())
    
    lstm_proba_test = np.vstack(lstm_proba_test)
    
    # Prédire avec VGG16
    vgg16_model.eval()
    vgg16_proba_test = []
    
    # Créer un dataset pour les images de test
    from models.train_model import ImageDataset
    test_image_dataset = ImageDataset(X_test["image_path"].values, transform=test_transform)
    test_image_loader = DataLoader(test_image_dataset, batch_size=params["batch_size"])
    
    with torch.no_grad():
        for images in tqdm(test_image_loader, desc="VGG16 predictions"):
            images = images.to(device)
            logits = vgg16_model(images)
            probs = torch.softmax(logits, dim=1)
            vgg16_proba_test.append(probs.cpu().numpy())
    
    vgg16_proba_test = np.vstack(vgg16_proba_test)
    
    # Combiner les prédictions
    combined_proba = best_weights[0] * lstm_proba_test + best_weights[1] * vgg16_proba_test
    final_predictions = np.argmax(combined_proba, axis=1)
    
    # Calculer l'accuracy
    accuracy = np.mean(final_predictions == y_test.values)
    
    # Log de l'accuracy finale
    mlflow.log_metric("test_accuracy", accuracy)
    
    # Sauvegarder les métriques dans un fichier JSON
    metrics = {
        "test_accuracy": float(accuracy),
        "lstm_weight": float(best_weights[0]),
        "vgg16_weight": float(best_weights[1])
    }
    
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f)
    
    # Sauvegarder l'architecture des modèles et leur mapping
    model_info = {
        "num_classes": int(params["num_classes"]),
        "max_sequence_length": int(params["max_sequence_length"]),
        "max_words": int(params["max_words"]),
        "embedding_dim": int(params["embedding_dim"]),
        "lstm_units": int(params["lstm_units"]),
        "image_size": int(params["image_size"])
    }
    
    with open("models/model_info.json", "w") as f:
        json.dump(model_info, f)
    
    # Sauvegarder le mapping des classes (déjà fait dans DataImporter.load_data)
    # Mais nous allons le convertir au format JSON
    with open("models/mapper.pkl", "rb") as f:
        mapper = pickle.load(f)
    
    mapper_json = {str(v): str(k) for k, v in mapper.items()}
    with open("models/mapper.json", "w") as f:
        json.dump(mapper_json, f)
    
    mlflow.log_artifact("models/best_weights.json")
    mlflow.log_artifact("models/metrics.json")

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Best weights: LSTM={best_weights[0]:.4f}, VGG16={best_weights[1]:.4f}")
    print("Modèles et métriques sauvegardés")
