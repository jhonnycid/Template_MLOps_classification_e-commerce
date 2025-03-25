import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import pickle
import argparse
import os
from torchvision import transforms
from PIL import Image
from models.train_model import LSTMModel, VGG16Model, TextPreprocessor
from features.build_features import ImagePreprocessor


class Predict:
    def __init__(
        self,
        text_preprocessor,
        lstm_model,
        vgg16_model,
        best_weights,
        mapper,
        filepath,
        imagepath
    ):
        self.text_preprocessor = text_preprocessor
        self.lstm_model = lstm_model
        self.vgg16_model = vgg16_model
        self.best_weights = best_weights
        self.mapper = mapper
        self.filepath = filepath
        self.imagepath = imagepath
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Charger les modèles sur le device approprié
        self.lstm_model.to(self.device)
        self.vgg16_model.to(self.device)
        
        # Définir les transformations d'image
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.image_transform(img)
            return img_tensor
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return torch.zeros((3, 224, 224))

    def predict(self):
        # Charger les données
        X = pd.read_csv(self.filepath)[:10]  # Limiter à 10 pour test
        
        # Prétraiter les textes
        from features.build_features import TextPreprocessor as FeatureTextPreprocessor
        text_preprocessor_feature = FeatureTextPreprocessor()
        image_preprocessor = ImagePreprocessor(self.imagepath)
        
        text_preprocessor_feature.preprocess_text_in_df(X, columns=["description"])
        image_preprocessor.preprocess_images_in_df(X)
        
        # Tokeniser les textes
        sequences = self.text_preprocessor.texts_to_sequences(X["description"], max_len=10)
        text_tensors = torch.tensor(sequences, dtype=torch.long).to(self.device)
        
        # Prétraiter les images
        image_tensors = []
        for img_path in X["image_path"]:
            img_tensor = self.preprocess_image(img_path)
            image_tensors.append(img_tensor)
        
        image_tensors = torch.stack(image_tensors).to(self.device)
        
        # Prédire avec les modèles en mode évaluation
        self.lstm_model.eval()
        self.vgg16_model.eval()
        
        with torch.no_grad():
            # Prédictions LSTM
            lstm_logits = self.lstm_model(text_tensors)
            lstm_proba = torch.softmax(lstm_logits, dim=1).cpu().numpy()
            
            # Prédictions VGG16
            vgg16_logits = self.vgg16_model(image_tensors)
            vgg16_proba = torch.softmax(vgg16_logits, dim=1).cpu().numpy()
        
        # Combiner les prédictions
        concatenate_proba = self.best_weights[0] * lstm_proba + self.best_weights[1] * vgg16_proba
        final_predictions = np.argmax(concatenate_proba, axis=1)
        
        # Mapper les prédictions aux catégories
        return {
            i: self.mapper.get(str(final_predictions[i]), "Unknown")
            for i in range(len(final_predictions))
        }


def main():
    parser = argparse.ArgumentParser(description="Input data")
    
    parser.add_argument("--dataset_path", default="data/preprocessed/X_train_update.csv", type=str, help="File path for the input CSV file.")
    parser.add_argument("--images_path", default="data/preprocessed/image_train", type=str, help="Base path for the images.")
    args = parser.parse_args()
    
    # Charger les configurations et modèles
    
    # 1. Charger le préprocesseur de texte
    with open("models/text_preprocessor.pkl", "rb") as f:
        text_preprocessor = pickle.load(f)
    
    # 2. Charger les informations du modèle
    with open("models/model_info.json", "r") as f:
        model_info = json.load(f)
    
    # 3. Créer et charger le modèle LSTM
    lstm_model = LSTMModel(
        vocab_size=text_preprocessor["n_words"],
        embedding_dim=model_info["embedding_dim"],
        hidden_dim=model_info["lstm_units"],
        output_dim=model_info["num_classes"]
    )
    lstm_model.load_state_dict(torch.load("models/best_lstm_model.pth", map_location=torch.device('cpu')))
    
    # 4. Créer et charger le modèle VGG16
    vgg16_model = VGG16Model(output_dim=model_info["num_classes"], pretrained=False)
    vgg16_model.load_state_dict(torch.load("models/best_vgg16_model.pth", map_location=torch.device('cpu')))
    
    # 5. Charger les poids optimaux
    with open("models/best_weights.json", "r") as f:
        best_weights = json.load(f)
    
    # 6. Charger le mapper de catégories
    with open("models/mapper.json", "r") as f:
        mapper = json.load(f)
    
    # Créer l'instance Predict et exécuter la prédiction
    predictor = Predict(
        text_preprocessor=TextPreprocessor(**text_preprocessor),
        lstm_model=lstm_model,
        vgg16_model=vgg16_model,
        best_weights=best_weights,
        mapper=mapper,
        filepath=args.dataset_path,
        imagepath=args.images_path
    )
    
    predictions = predictor.predict()
    
    # Sauvegarder les prédictions
    with open("data/preprocessed/predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Prédictions enregistrées dans data/preprocessed/predictions.json")


if __name__ == "__main__":
    main()
