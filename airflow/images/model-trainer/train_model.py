"""
Script pour entraîner les modèles LSTM et VGG16.
Ce script peut être utilisé pour entraîner soit le modèle LSTM, soit le modèle VGG16,
en fonction des arguments passés en ligne de commande.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.keras
import json
from datetime import datetime
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split

# Configuration des chemins
DATA_DIR = "/app/data"
MODELS_DIR = "/app/models"
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

def parse_args():
    """Parse les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(description="Script d'entraînement des modèles LSTM ou VGG16.")
    parser.add_argument("--model", type=str, choices=["lstm", "vgg16", "both"], default="both",
                      help="Le modèle à entraîner: 'lstm', 'vgg16' ou 'both' (défaut)")
    parser.add_argument("--epochs", type=int, default=10, help="Nombre d'époques d'entraînement (défaut: 10)")
    parser.add_argument("--batch_size", type=int, default=32, help="Taille du batch (défaut: 32)")
    parser.add_argument("--data_file", type=str, default="X_train_update.csv", 
                      help="Nom du fichier CSV de données (défaut: X_train_update.csv)")
    return parser.parse_args()

def load_data(data_file="X_train_update.csv"):
    """Charge les données d'entraînement."""
    try:
        file_path = os.path.join(PREPROCESSED_DIR, data_file)
        df = pd.read_csv(file_path)
        
        # Extraire les caractéristiques et les étiquettes
        X = df.drop("prdtypecode", axis=1, errors='ignore')
        y = df["prdtypecode"] if "prdtypecode" in df.columns else None
        
        if y is None:
            raise ValueError("La colonne 'prdtypecode' n'est pas présente dans le DataFrame")
        
        # Diviser en ensembles d'entraînement et de validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Données chargées avec succès: {len(X_train)} exemples d'entraînement, {len(X_val)} exemples de validation")
        
        return X_train, X_val, y_train, y_val
    except Exception as e:
        print(f"Erreur lors du chargement des données: {str(e)}")
        sys.exit(1)

class TextLSTMModel:
    def __init__(self, max_words=10000, max_sequence_length=10, embedding_dim=128, batch_size=32, epochs=10):
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        # Configurer MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("text_lstm_model")
        
        with mlflow.start_run():
            # Enregistrer les hyperparamètres
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("embedding_dim", self.embedding_dim)
            mlflow.log_param("max_words", self.max_words)
            mlflow.log_param("max_sequence_length", self.max_sequence_length)
            
            # Prétraitement du texte
            self.tokenizer.fit_on_texts(X_train["description"])
            
            # Sauvegarder la configuration du tokenizer
            tokenizer_config = self.tokenizer.to_json()
            tokenizer_path = os.path.join(MODELS_DIR, "tokenizer_config.json")
            with open(tokenizer_path, "w", encoding="utf-8") as json_file:
                json_file.write(tokenizer_config)
                
            mlflow.log_artifact(tokenizer_path)
            
            # Transformer les textes en séquences
            train_sequences = self.tokenizer.texts_to_sequences(X_train["description"])
            train_padded_sequences = pad_sequences(
                train_sequences,
                maxlen=self.max_sequence_length,
                padding="post",
                truncating="post",
            )
            
            val_sequences = self.tokenizer.texts_to_sequences(X_val["description"])
            val_padded_sequences = pad_sequences(
                val_sequences,
                maxlen=self.max_sequence_length,
                padding="post",
                truncating="post",
            )
            
            # Construire le modèle
            text_input = Input(shape=(self.max_sequence_length,))
            embedding_layer = Embedding(input_dim=self.max_words, output_dim=self.embedding_dim)(
                text_input
            )
            lstm_layer = LSTM(128)(embedding_layer)
            output = Dense(27, activation="softmax")(lstm_layer)
            
            self.model = Model(inputs=[text_input], outputs=output)
            
            self.model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )
            
            # Définir les callbacks
            class MLflowCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    mlflow.log_metrics(logs, step=epoch)
            
            model_path = os.path.join(MODELS_DIR, "best_lstm_model.h5")
            
            lstm_callbacks = [
                ModelCheckpoint(
                    filepath=model_path, save_best_only=True
                ),
                EarlyStopping(
                    patience=3, restore_best_weights=True
                ),
                TensorBoard(log_dir=os.path.join(MODELS_DIR, "logs")),
                MLflowCallback(),
            ]
            
            # Entraîner le modèle
            history = self.model.fit(
                [train_padded_sequences],
                tf.keras.utils.to_categorical(y_train, num_classes=27),
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(
                    [val_padded_sequences],
                    tf.keras.utils.to_categorical(y_val, num_classes=27),
                ),
                callbacks=lstm_callbacks,
            )
            
            # Évaluer le modèle final
            val_loss, val_accuracy = self.model.evaluate(
                [val_padded_sequences],
                tf.keras.utils.to_categorical(y_val, num_classes=27)
            )
            
            # Enregistrer les métriques finales
            mlflow.log_metric("final_val_loss", val_loss)
            mlflow.log_metric("final_val_accuracy", val_accuracy)
            
            # Enregistrer le modèle
            mlflow.keras.log_model(self.model, "lstm_model")
            mlflow.log_artifact(model_path)
            
            print(f"Modèle LSTM entraîné avec succès. Précision de validation: {val_accuracy:.4f}")
            return val_accuracy

class ImageVGG16Model:
    def __init__(self, batch_size=32, epochs=1):
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        # Configurer MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("image_vgg16_model")
        
        with mlflow.start_run():
            # Enregistrer les hyperparamètres
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("epochs", self.epochs)
            
            # Paramètres
            num_classes = 27
            
            # Préparer les données
            df_train = pd.concat([X_train, y_train.astype(str)], axis=1)
            df_val = pd.concat([X_val, y_val.astype(str)], axis=1)
            
            # Créer les générateurs d'images
            train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
            train_generator = train_datagen.flow_from_dataframe(
                dataframe=df_train,
                x_col="image_path",
                y_col="prdtypecode",
                target_size=(224, 224),
                batch_size=self.batch_size,
                class_mode="categorical",
                shuffle=True,
            )
            
            val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
            val_generator = val_datagen.flow_from_dataframe(
                dataframe=df_val,
                x_col="image_path",
                y_col="prdtypecode",
                target_size=(224, 224),
                batch_size=self.batch_size,
                class_mode="categorical",
                shuffle=False,
            )
            
            # Construire le modèle
            image_input = Input(shape=(224, 224, 3))
            
            vgg16_base = VGG16(
                include_top=False, weights="imagenet", input_tensor=image_input
            )
            
            x = vgg16_base.output
            x = Flatten()(x)
            x = Dense(256, activation="relu")(x)
            output = Dense(num_classes, activation="softmax")(x)
            
            self.model = Model(inputs=vgg16_base.input, outputs=output)
            
            # Geler les couches VGG16
            for layer in vgg16_base.layers:
                layer.trainable = False
            
            self.model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )
            
            # Définir les callbacks
            class MLflowCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    mlflow.log_metrics(logs, step=epoch)
            
            model_path = os.path.join(MODELS_DIR, "best_vgg16_model.h5")
            
            vgg_callbacks = [
                ModelCheckpoint(
                    filepath=model_path, save_best_only=True
                ),
                EarlyStopping(
                    patience=3, restore_best_weights=True
                ),
                TensorBoard(log_dir=os.path.join(MODELS_DIR, "logs")),
                MLflowCallback(),
            ]
            
            # Entraîner le modèle
            history = self.model.fit(
                train_generator,
                epochs=self.epochs,
                validation_data=val_generator,
                callbacks=vgg_callbacks,
            )
            
            # Évaluer le modèle final
            val_generator.reset()
            val_steps = len(val_generator)
            val_metrics = self.model.evaluate(val_generator, steps=val_steps)
            
            # Enregistrer les métriques finales
            mlflow.log_metric("final_val_loss", val_metrics[0])
            mlflow.log_metric("final_val_accuracy", val_metrics[1])
            
            # Enregistrer le modèle
            mlflow.keras.log_model(self.model, "vgg16_model")
            mlflow.log_artifact(model_path)
            
            print(f"Modèle VGG16 entraîné avec succès. Précision de validation: {val_metrics[1]:.4f}")
            return val_metrics[1]

def main():
    args = parse_args()
    X_train, X_val, y_train, y_val = load_data(args.data_file)
    
    # Créer un fichier mapper.json si nécessaire
    mapper_path = os.path.join(MODELS_DIR, "mapper.json")
    if not os.path.exists(mapper_path):
        classes = sorted(y_train.unique())
        mapper = {str(i): str(cls) for i, cls in enumerate(classes)}
        with open(mapper_path, "w") as f:
            json.dump(mapper, f)
        print(f"Fichier mapper.json créé: {mapper_path}")
    
    if args.model == "lstm" or args.model == "both":
        print("Entraînement du modèle LSTM...")
        lstm_model = TextLSTMModel(epochs=args.epochs, batch_size=args.batch_size)
        lstm_accuracy = lstm_model.train(X_train, y_train, X_val, y_val)
    
    if args.model == "vgg16" or args.model == "both":
        print("Entraînement du modèle VGG16...")
        vgg16_model = ImageVGG16Model(epochs=args.epochs, batch_size=args.batch_size)
        vgg16_accuracy = vgg16_model.train(X_train, y_train, X_val, y_val)
    
    print("Entraînement terminé!")

if __name__ == "__main__":
    main()
