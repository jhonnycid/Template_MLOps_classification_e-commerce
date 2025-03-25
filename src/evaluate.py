import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import pickle
import os
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision import transforms
from PIL import Image
from models.train_model import LSTMModel, VGG16Model, TextPreprocessor
from tqdm import tqdm

# Configuration de MLflow
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("rakuten-classification-evaluation")

# Définir le device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Transformations d'image
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = image_transform(img)
        return img_tensor
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return torch.zeros((3, 224, 224))

class TextDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = None
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.ToTensor()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            if self.labels is not None:
                return image, self.labels[idx]
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image in case of error
            dummy_img = torch.zeros((3, 224, 224))
            if self.labels is not None:
                return dummy_img, self.labels[idx]
            return dummy_img

def main():
    # Charger les données de test
    data_path = "data/preprocessed/X_train_update.csv"
    target_path = "data/preprocessed/Y_train_CVw08PX.csv"
    
    # Charger les données
    X_test = pd.read_csv(data_path)
    X_test["description"] = X_test["designation"] + " " + X_test["description"].fillna("")
    X_test = X_test.drop(["Unnamed: 0", "designation"], axis=1)
    
    y_test = pd.read_csv(target_path)
    y_test = y_test.drop(["Unnamed: 0"], axis=1)
    
    # Charger le mapping des classes
    try:
        with open("models/mapper.pkl", "rb") as file:
            modalite_mapping = pickle.load(file)
        
        # Appliquer le mapping aux étiquettes
        y_test["prdtypecode"] = y_test["prdtypecode"].replace(modalite_mapping)
        
        # Convertir le mapper pour la sortie JSON
        mapper_json = {str(v): str(k) for k, v in modalite_mapping.items()}
        
    except Exception as e:
        print(f"Erreur lors du chargement du mapper: {e}")
        # Essayer de charger le mapper JSON à la place
        with open("models/mapper.json", "r") as f:
            mapper_json = json.load(f)
        # Inverser le mapping pour l'utiliser
        modalite_mapping = {int(v): int(k) for k, v in mapper_json.items()}
        y_test["prdtypecode"] = y_test["prdtypecode"].replace(modalite_mapping)
    
    # Fusionner X et y
    df = pd.concat([X_test, y_test], axis=1)
    
    # Prendre un sous-ensemble pour l'évaluation
    df_test = df.groupby("prdtypecode").sample(n=20, random_state=42)
    
    X_test = df_test.drop(["prdtypecode"], axis=1)
    y_test = df_test["prdtypecode"].values
    
    with mlflow.start_run():
        # Log des paramètres
        mlflow.log_param("test_samples", len(X_test))
        
        # Ajouter les chemins d'image
        X_test["image_path"] = "data/preprocessed/image_train/image_" + X_test["imageid"].astype(str) + "_product_" + X_test["productid"].astype(str) + ".jpg"
        
        # Prétraiter le texte pour NLTK
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from bs4 import BeautifulSoup
        import re
        import math
        
        # Télécharger les ressources NLTK nécessaires
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        # Fonction de prétraitement du texte
        def preprocess_text(text):
            if isinstance(text, float) and math.isnan(text):
                return ""
            # Supprimer les balises HTML
            text = BeautifulSoup(text, "html.parser").get_text()
            # Supprimer les caractères non alphabétiques
            text = re.sub(r"[^a-zA-Z]", " ", text)
            # Tokenization
            words = word_tokenize(text.lower())
            # Suppression des stopwords et lemmatisation
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words("french"))
            filtered_words = [
                lemmatizer.lemmatize(word)
                for word in words
                if word not in stop_words
            ]
            return " ".join(filtered_words[:10])
        
        X_test["description"] = X_test["description"].apply(preprocess_text)
        
        # Charger le préprocesseur de texte
        with open("models/text_preprocessor.pkl", "rb") as f:
            text_preprocessor_data = pickle.load(f)
        
        text_preprocessor = TextPreprocessor(
            max_words=text_preprocessor_data.get("max_words", 10000)
        )
        text_preprocessor.word_to_idx = text_preprocessor_data.get("word_to_idx", {})
        text_preprocessor.idx_to_word = text_preprocessor_data.get("idx_to_word", {})
        text_preprocessor.word_counts = text_preprocessor_data.get("word_counts", {})
        text_preprocessor.n_words = text_preprocessor_data.get("n_words", 2)
        
        # Charger les informations du modèle
        with open("models/model_info.json", "r") as f:
            model_info = json.load(f)
        
        # Tokeniser le texte
        sequences = text_preprocessor.texts_to_sequences(X_test["description"], max_len=10)
        
        # Créer dataset pour le texte
        text_dataset = TextDataset(sequences)
        text_loader = DataLoader(text_dataset, batch_size=32)
        
        # Créer dataset pour les images
        image_dataset = ImageDataset(X_test["image_path"].values, transform=image_transform)
        image_loader = DataLoader(image_dataset, batch_size=32)
        
        # Charger les modèles
        lstm_model = LSTMModel(
            vocab_size=text_preprocessor.n_words,
            embedding_dim=model_info["embedding_dim"],
            hidden_dim=model_info["lstm_units"],
            output_dim=model_info["num_classes"]
        )
        lstm_model.load_state_dict(torch.load("models/best_lstm_model.pth", map_location=device))
        lstm_model.to(device)
        lstm_model.eval()
        
        vgg16_model = VGG16Model(output_dim=model_info["num_classes"])
        vgg16_model.load_state_dict(torch.load("models/best_vgg16_model.pth", map_location=device))
        vgg16_model.to(device)
        vgg16_model.eval()
        
        # Charger les poids optimaux
        with open("models/best_weights.json", "r") as f:
            best_weights = json.load(f)
        
        # Prédire avec le modèle LSTM
        lstm_proba = []
        with torch.no_grad():
            for texts in tqdm(text_loader, desc="LSTM predictions"):
                texts = texts.to(device)
                logits = lstm_model(texts)
                probs = torch.softmax(logits, dim=1)
                lstm_proba.append(probs.cpu().numpy())
        
        lstm_proba = np.vstack(lstm_proba)
        
        # Prédire avec le modèle VGG16
        vgg16_proba = []
        with torch.no_grad():
            for images in tqdm(image_loader, desc="VGG16 predictions"):
                images = images.to(device)
                logits = vgg16_model(images)
                probs = torch.softmax(logits, dim=1)
                vgg16_proba.append(probs.cpu().numpy())
        
        vgg16_proba = np.vstack(vgg16_proba)
        
        # Combiner les prédictions
        combined_proba = best_weights[0] * lstm_proba + best_weights[1] * vgg16_proba
        predictions = np.argmax(combined_proba, axis=1)
        
        # Évaluer les résultats
        accuracy = accuracy_score(y_test, predictions)
        
        # Log des métriques
        mlflow.log_metric("accuracy", accuracy)
        
        # Créer le rapport de classification
        class_report = classification_report(y_test, predictions, output_dict=True)
        mlflow.log_dict(class_report, "classification_report.json")
        
        # Créer la matrice de confusion
        conf_matrix = confusion_matrix(y_test, predictions).tolist()
        mlflow.log_dict({"confusion_matrix": conf_matrix}, "confusion_matrix.json")
        
        # Sauvegarder les résultats d'évaluation
        eval_results = {
            "accuracy": float(accuracy),
            "classification_report": class_report,
            "confusion_matrix": conf_matrix,
            "lstm_weight": float(best_weights[0]),
            "vgg16_weight": float(best_weights[1])
        }
        
        with open("models/evaluation.json", "w") as f:
            json.dump(eval_results, f, indent=4)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Poids des modèles: LSTM={best_weights[0]:.2f}, VGG16={best_weights[1]:.2f}")
        print("Évaluation terminée. Résultats enregistrés dans models/evaluation.json")

if __name__ == "__main__":
    main()
