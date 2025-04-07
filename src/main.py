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
mlflow.set_experiment("rakuten-classification")

# Définir le device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Vérifier si les modèles existent déjà
lstm_exists = os.path.exists("models/best_lstm_model.pth")
vgg_exists = os.path.exists("models/best_vgg16_model.pth")

with mlflow.start_run(run_name="Rakuten-Full-Training") as run:
    mlflow.log_params(params)

    # Importer les données
    data_importer = DataImporter()
    df = data_importer.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = data_importer.split_train_test(df)

    # Prétraitement des textes et images
    text_preprocessor = TextPreprocessor()
    image_preprocessor = ImagePreprocessor()

    for X in [X_train, X_val, X_test]:
        text_preprocessor.preprocess_text_in_df(X, columns=["description"])
        image_preprocessor.preprocess_images_in_df(X)

    # Entraînement du modèle LSTM
    if not lstm_exists:
        print("🧠 Entraînement du modèle LSTM...")
        text_lstm_model = TextLSTMModel(
            max_words=params["max_words"],
            max_sequence_length=params["max_sequence_length"],
            embedding_dim=params["embedding_dim"],
            hidden_dim=params["lstm_units"]
        )
        text_lstm_model.preprocess_and_fit(X_train, y_train, X_val, y_val, epochs=params["epochs"], batch_size=params["batch_size"])
        mlflow.pytorch.log_model(text_lstm_model.model, artifact_path="lstm_model", registered_model_name="rakuten-lstm")

        print("✅ LSTM entraîné.")
    else:
        print("✅ Modèle LSTM déjà présent.")

    # Entraînement du modèle VGG16
    if not vgg_exists:
        print("🧠 Entraînement du modèle VGG16...")
        image_vgg16_model = ImageVGG16Model()
        image_vgg16_model.preprocess_and_fit(X_train, y_train, X_val, y_val, epochs=params["epochs"], batch_size=params["batch_size"])
        mlflow.pytorch.log_model(image_vgg16_model.model, artifact_path="vgg_model", registered_model_name="rakuten-vgg16")

        print("✅ VGG16 entraîné.")
    else:
        print("✅ Modèle VGG16 déjà présent.")

    # Chargement des modèles
    from models.train_model import LSTMModel, VGG16Model
    model_info = {
        "num_classes": int(params["num_classes"]),
        "embedding_dim": int(params["embedding_dim"]),
        "lstm_units": int(params["lstm_units"])
    }

    text_tokenizer = TextTokenizer.load("models/text_preprocessor.pkl")

    lstm_model = LSTMModel(
        vocab_size=text_tokenizer.n_words,
        embedding_dim=params["embedding_dim"],
        hidden_dim=params["lstm_units"],
        output_dim=params["num_classes"]
    )
    lstm_model.load_state_dict(torch.load("models/best_lstm_model.pth"))
    lstm_model.eval()

    vgg16_model = VGG16Model(output_dim=params["num_classes"], pretrained=False)
    vgg16_model.load_state_dict(torch.load("models/best_vgg16_model.pth"))
    vgg16_model.eval()

    # Optimisation des poids
    model_concatenate = Concatenate(text_tokenizer, lstm_model, vgg16_model)
    lstm_proba, vgg16_proba, new_y_train = model_concatenate.predict(X_train, y_train)
    best_weights = model_concatenate.optimize(lstm_proba, vgg16_proba, new_y_train)

    with open("models/best_weights.pkl", "wb") as file:
        pickle.dump(best_weights, file)
    with open("models/best_weights.json", "w") as f:
        json.dump(best_weights, f)

    # Évaluation finale
    print("🧪 Évaluation sur le test set...")
    test_sequences = text_tokenizer.texts_to_sequences(X_test["description"], max_len=params["max_sequence_length"])
    test_text_loader = DataLoader(torch.tensor(test_sequences, dtype=torch.long), batch_size=params["batch_size"])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    from models.train_model import ImageDataset
    test_image_loader = DataLoader(ImageDataset(X_test["image_path"].values, transform=transform), batch_size=params["batch_size"])

    lstm_model.eval()
    vgg16_model.eval()

    lstm_proba_test = []
    with torch.no_grad():
        for batch in test_text_loader:
            batch = batch.to(device)
            logits = lstm_model(batch)
            probs = torch.softmax(logits, dim=1)
            lstm_proba_test.append(probs.cpu().numpy())
    lstm_proba_test = np.vstack(lstm_proba_test)

    vgg16_proba_test = []
    with torch.no_grad():
        for batch in test_image_loader:
            batch = batch.to(device)
            logits = vgg16_model(batch)
            probs = torch.softmax(logits, dim=1)
            vgg16_proba_test.append(probs.cpu().numpy())
    vgg16_proba_test = np.vstack(vgg16_proba_test)

    combined_proba = best_weights[0] * lstm_proba_test + best_weights[1] * vgg16_proba_test
    final_predictions = np.argmax(combined_proba, axis=1)
    accuracy = np.mean(final_predictions == y_test.values)

    mlflow.log_metric("test_accuracy", accuracy)
    # 🔐 Log des artefacts utiles dans MLflow
    mlflow.log_artifact("models/best_lstm_model.pth")
    mlflow.log_artifact("models/best_vgg16_model.pth")
    mlflow.log_artifact("models/text_preprocessor.pkl")
    mlflow.log_artifact("models/best_weights.json")
    mlflow.log_artifact("models/metrics.json")

    # (Optionnel) si tu veux aussi archiver le mapping classe/label :
    if os.path.exists("models/mapper.json"):
        mlflow.log_artifact("models/mapper.json")
        
    print(f"🎯 Test Accuracy: {accuracy:.4f}")
