import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os
import json
import pickle
import mlflow
import mlflow.pytorch
from tqdm import tqdm

# Définir les transformations d'image pour VGG16
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class TextPreprocessor:
    def __init__(self, max_words=10000):
        self.max_words = max_words
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.word_counts = {}
        self.n_words = 2  # Compte <PAD> et <UNK>

    def fit_on_texts(self, texts):
        for text in texts:
            for word in text.split():
                self.word_counts[word] = self.word_counts.get(word, 0) + 1
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.max_words - 2]:
            self.word_to_idx[word] = self.n_words
            self.idx_to_word[self.n_words] = word
            self.n_words += 1

    def texts_to_sequences(self, texts, max_len=10):
        sequences = []
        for text in texts:
            seq = [self.word_to_idx.get(word, 1) for word in text.split()[:max_len]]
            seq += [0] * (max_len - len(seq)) if len(seq) < max_len else []
            sequences.append(seq[:max_len])
        return sequences

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        instance = cls()
        instance.__dict__.update(data)
        return instance

class TextDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (self.sequences[idx], self.labels[idx]) if self.labels is not None else self.sequences[idx]

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            img = self.transform(img)
        except Exception:
            img = torch.zeros((3, 224, 224))
        return (img, self.labels[idx]) if self.labels is not None else img

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(self.dropout(hidden[-1]))

class TextLSTMModel:
    def __init__(self, max_words=10000, max_sequence_length=10, embedding_dim=128, hidden_dim=128):
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.preprocessor = TextPreprocessor(max_words=max_words)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def preprocess_and_fit(self, X_train, y_train, X_val, y_val, epochs=1, batch_size=32):
        self.preprocessor.fit_on_texts(X_train["description"])
        os.makedirs("models", exist_ok=True)
        self.preprocessor.save("models/text_preprocessor.pkl")

        train_seq = self.preprocessor.texts_to_sequences(X_train["description"], self.max_sequence_length)
        val_seq = self.preprocessor.texts_to_sequences(X_val["description"], self.max_sequence_length)

        train_loader = DataLoader(TextDataset(train_seq, y_train.values), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TextDataset(val_seq, y_val.values), batch_size=batch_size)

        num_classes = len(np.unique(y_train))
        self.model = LSTMModel(
            vocab_size=self.preprocessor.n_words,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=num_classes
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())

        with mlflow.start_run(run_name="LSTM-Training", nested=True):
            mlflow.log_params({
                "model": "LSTM",
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "epochs": epochs,
                "batch_size": batch_size
            })

            mlflow.pytorch.log_model(self.model, "initial_model")

            best_val_loss = float('inf')
            for epoch in range(epochs):
                self.model.train()
                train_loss, train_acc = 0, 0
                for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train"):
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    preds = self.model(x)
                    loss = criterion(preds, y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    train_acc += (preds.argmax(1) == y).float().mean().item()

                train_loss /= len(train_loader)
                train_acc /= len(train_loader)

                self.model.eval()
                val_loss, val_acc = 0, 0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        preds = self.model(x)
                        loss = criterion(preds, y)
                        val_loss += loss.item()
                        val_acc += (preds.argmax(1) == y).float().mean().item()

                val_loss /= len(val_loader)
                val_acc /= len(val_loader)

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                }, step=epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), "models/best_lstm_model.pth")

        self.model.load_state_dict(torch.load("models/best_lstm_model.pth"))
        return None

class VGG16Model(nn.Module):
    def __init__(self, output_dim, pretrained=True):
        super().__init__()
        self.vgg16 = models.vgg16(pretrained=pretrained)
        in_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(in_features, output_dim)

    def forward(self, x):
        return self.vgg16(x)

class ImageVGG16Model:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def preprocess_and_fit(self, X_train, y_train, X_val, y_val, epochs=1, batch_size=32):
        train_loader = DataLoader(ImageDataset(X_train["image_path"].values, y_train.values, transform=image_transforms), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ImageDataset(X_val["image_path"].values, y_val.values, transform=image_transforms), batch_size=batch_size)

        num_classes = len(np.unique(y_train))
        self.model = VGG16Model(output_dim=num_classes).to(self.device)
        for param in self.model.vgg16.features.parameters():
            param.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()))

        with mlflow.start_run(run_name="VGG16-Training", nested=True):
            mlflow.log_params({
                "model": "VGG16",
                "epochs": epochs,
                "batch_size": batch_size
            })
            mlflow.pytorch.log_model(self.model, "initial_model")

            best_val_loss = float('inf')
            for epoch in range(epochs):
                self.model.train()
                train_loss, train_acc = 0, 0
                for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train"):
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    preds = self.model(x)
                    loss = criterion(preds, y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    train_acc += (preds.argmax(1) == y).float().mean().item()

                train_loss /= len(train_loader)
                train_acc /= len(train_loader)

                self.model.eval()
                val_loss, val_acc = 0, 0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        preds = self.model(x)
                        loss = criterion(preds, y)
                        val_loss += loss.item()
                        val_acc += (preds.argmax(1) == y).float().mean().item()

                val_loss /= len(val_loader)
                val_acc /= len(val_loader)

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                }, step=epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), "models/best_vgg16_model.pth")

        self.model.load_state_dict(torch.load("models/best_vgg16_model.pth"))
        return None

class Concatenate:
    def __init__(self, preprocessor, lstm_model, vgg16_model):
        self.preprocessor = preprocessor
        self.lstm_model = lstm_model
        self.vgg16_model = vgg16_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def preprocess_image(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = image_transforms(img)
            return img_tensor
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return torch.zeros((3, 224, 224))
    
    def predict(self, X_data, y_data, new_samples_per_class=50):
        num_classes = len(np.unique(y_data))
        
        # Échantillonner les données
        new_X_data = pd.DataFrame(columns=X_data.columns)
        new_y_data = []
        
        for class_label in range(num_classes):
            # Indices des échantillons de cette classe
            indices = np.where(y_data == class_label)[0]
            
            # Vérifier qu'il y a suffisamment d'échantillons
            n_samples = min(new_samples_per_class, len(indices))
            if n_samples < new_samples_per_class:
                print(f"Warning: Only {n_samples} samples available for class {class_label}")
            
            # Sous-échantillonnage aléatoire
            sampled_indices = resample(
                indices, n_samples=n_samples, replace=False, random_state=42
            )
            
            # Ajouter les échantillons
            new_X_data = pd.concat([new_X_data, X_data.iloc[sampled_indices]])
            new_y_data.extend([class_label] * n_samples)
        
        # Réinitialiser les index
        new_X_data = new_X_data.reset_index(drop=True)
        new_y_data = np.array(new_y_data)
        
        # Prétraiter les textes
        sequences = self.preprocessor.texts_to_sequences(
            new_X_data["description"], max_len=10
        )
        
        # Prétraiter les images
        image_tensors = []
        for img_path in new_X_data["image_path"]:
            image_tensors.append(self.preprocess_image(img_path))
        
        # Prédictions avec LSTM
        text_dataset = TextDataset(sequences)
        text_loader = DataLoader(text_dataset, batch_size=32)
        
        self.lstm_model.eval()
        lstm_proba = []
        
        with torch.no_grad():
            for texts in text_loader:
                texts = texts.to(self.device)
                logits = self.lstm_model(texts)
                probs = torch.softmax(logits, dim=1)
                lstm_proba.append(probs.cpu().numpy())
        
        lstm_proba = np.vstack(lstm_proba)
        
        # Prédictions avec VGG16
        image_dataset = ImageDataset(new_X_data["image_path"].values, transform=image_transforms)
        image_loader = DataLoader(image_dataset, batch_size=32)
        
        self.vgg16_model.eval()
        vgg16_proba = []
        
        with torch.no_grad():
            for images in image_loader:
                images = images.to(self.device)
                logits = self.vgg16_model(images)
                probs = torch.softmax(logits, dim=1)
                vgg16_proba.append(probs.cpu().numpy())
        
        vgg16_proba = np.vstack(vgg16_proba)
        
        mlflow.log_metric("lstm_train_samples", len(sequences))
        mlflow.log_metric("vgg16_train_samples", len(image_tensors))
        
        return lstm_proba, vgg16_proba, new_y_data
    
    def optimize(self, lstm_proba, vgg16_proba, y_train):
        # Recherche des poids optimaux
        best_weights = None
        best_accuracy = 0.0
        
        # Log des résultats d'optimisation
        weight_results = []
        
        for lstm_weight in np.linspace(0, 1, 101):
            vgg16_weight = 1.0 - lstm_weight
            
            combined_predictions = (lstm_weight * lstm_proba) + (vgg16_weight * vgg16_proba)
            final_predictions = np.argmax(combined_predictions, axis=1)
            accuracy = accuracy_score(y_train, final_predictions)
            
            # Enregistrer ce résultat
            weight_results.append({
                "lstm_weight": float(lstm_weight),
                "vgg16_weight": float(vgg16_weight),
                "accuracy": float(accuracy)
            })
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = (float(lstm_weight), float(vgg16_weight))
        
        # Log des résultats de la recherche de poids
        mlflow.log_dict(weight_results, "weight_optimization_results.json")
        
        # Log des métriques finales
        mlflow.log_metric("best_lstm_weight", best_weights[0])
        mlflow.log_metric("best_vgg16_weight", best_weights[1])
        mlflow.log_metric("best_ensemble_accuracy", best_accuracy)
        
        return best_weights