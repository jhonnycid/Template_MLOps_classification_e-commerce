"""
Script pour optimiser les poids du modèle combiné (LSTM + VGG16).
Ce script charge les modèles LSTM et VGG16 préalablement entraînés et trouve
les poids optimaux pour combiner leurs prédictions.
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import mlflow
import mlflow.keras
import argparse

# Configuration des chemins
DATA_DIR = "/app/data"
MODELS_DIR = "/app/models"
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

def parse_args():
    """Parse les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(description="Script d'optimisation du modèle combiné.")
    parser.add_argument("--data_file", type=str, default="X_train_update.csv", 
                      help="Nom du fichier CSV de données (défaut: X_train_update.csv)")
    parser.add_argument("--samples_per_class", type=int, default=50, 
                      help="Nombre d'échantillons par classe à utiliser pour l'optimisation (défaut: 50)")
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
        
        print(f"Données chargées avec succès: {len(X)} exemples")
        
        return X, y
    except Exception as e:
        print(f"Erreur lors du chargement des données: {str(e)}")
        sys.exit(1)

def preprocess_image(image_path, target_size=(224, 224, 3)):
    """Prétraite une image pour l'inférence VGG16."""
    try:
        img = load_img(image_path, target_size=target_size[:2])
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Erreur lors du prétraitement de l'image {image_path}: {str(e)}")
        # Retourner une image noire en cas d'erreur
        return np.zeros(target_size)

def sample_data(X, y, samples_per_class=50):
    """Sous-échantillonne les données pour l'optimisation."""
    num_classes = len(np.unique(y))
    
    new_X = pd.DataFrame(columns=X.columns)
    new_y = pd.Series(name='prdtypecode')
    
    for class_label in range(num_classes):
        indices = np.where(y == class_label)[0]
        
        if len(indices) <= samples_per_class:
            # Prendre tous les échantillons si moins que demandé
            sampled_indices = indices
        else:
            # Sous-échantillonnage aléatoire
            np.random.seed(42)
            sampled_indices = np.random.choice(indices, samples_per_class, replace=False)
        
        # Ajouter les échantillons sous-échantillonnés
        new_X = pd.concat([new_X, X.iloc[sampled_indices]])
        new_y = pd.concat([new_y, y.iloc[sampled_indices]])
    
    return new_X.reset_index(drop=True), new_y.reset_index(drop=True)

def main():
    args = parse_args()
    
    # Configurer MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("concatenate_model_optimization")
    
    with mlflow.start_run():
        try:
            # 1. Charger les données
            X, y = load_data(args.data_file)
            
            # 2. Sous-échantillonner les données pour l'optimisation
            X_sample, y_sample = sample_data(X, y, args.samples_per_class)
            
            mlflow.log_param("num_samples", len(X_sample))
            mlflow.log_param("samples_per_class", args.samples_per_class)
            
            print(f"Données sous-échantillonnées: {len(X_sample)} exemples")
            
            # 3. Charger les modèles et le tokenizer
            tokenizer_path = os.path.join(MODELS_DIR, "tokenizer_config.json")
            with open(tokenizer_path, "r", encoding="utf-8") as json_file:
                tokenizer_config = json_file.read()
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)
            
            lstm_model_path = os.path.join(MODELS_DIR, "best_lstm_model.h5")
            vgg16_model_path = os.path.join(MODELS_DIR, "best_vgg16_model.h5")
            
            lstm_model = keras.models.load_model(lstm_model_path)
            vgg16_model = keras.models.load_model(vgg16_model_path)
            
            # 4. Préparer les données pour les modèles
            # Pour LSTM
            sequences = tokenizer.texts_to_sequences(X_sample["description"])
            padded_sequences = pad_sequences(
                sequences, maxlen=10, padding="post", truncating="post"
            )
            
            # Pour VGG16
            target_size = (224, 224, 3)
            images = X_sample["image_path"].apply(
                lambda x: preprocess_image(x, target_size)
            )
            images_tensor = tf.convert_to_tensor(images.tolist(), dtype=tf.float32)
            
            # 5. Obtenir les prédictions des modèles individuels
            print("Obtention des prédictions LSTM...")
            lstm_proba = lstm_model.predict(padded_sequences)
            
            print("Obtention des prédictions VGG16...")
            vgg16_proba = vgg16_model.predict(images_tensor)
            
            # 6. Optimiser les poids
            print("Optimisation des poids...")
            best_weights = None
            best_accuracy = 0.0
            
            results = []
            
            for lstm_weight in np.linspace(0, 1, 101):  # Essayer différents poids pour LSTM
                vgg16_weight = 1.0 - lstm_weight  # Le poids total doit être égal à 1
                
                combined_predictions = (lstm_weight * lstm_proba) + (vgg16_weight * vgg16_proba)
                final_predictions = np.argmax(combined_predictions, axis=1)
                accuracy = accuracy_score(y_sample, final_predictions)
                
                # Enregistrer les résultats
                mlflow.log_metric(f"accuracy_{lstm_weight:.2f}", accuracy)
                results.append((lstm_weight, vgg16_weight, accuracy))
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = (lstm_weight, vgg16_weight)
            
            # 7. Enregistrer les meilleurs poids
            print(f"Meilleurs poids trouvés: LSTM={best_weights[0]:.4f}, VGG16={best_weights[1]:.4f}")
            print(f"Meilleure précision: {best_accuracy:.4f}")
            
            # Enregistrer les métriques et paramètres
            mlflow.log_metric("best_accuracy", best_accuracy)
            mlflow.log_param("best_lstm_weight", best_weights[0])
            mlflow.log_param("best_vgg16_weight", best_weights[1])
            
            # Enregistrer un fichier résumant les résultats
            results_df = pd.DataFrame(results, columns=["lstm_weight", "vgg16_weight", "accuracy"])
            results_path = os.path.join(MODELS_DIR, "optimization_results.csv")
            results_df.to_csv(results_path, index=False)
            mlflow.log_artifact(results_path)
            
            # Sauvegarder les poids dans un fichier JSON
            weights_path = os.path.join(MODELS_DIR, "best_weights.json")
            with open(weights_path, "w") as f:
                json.dump(best_weights, f)
            mlflow.log_artifact(weights_path)
            
            # 8. Créer et enregistrer le modèle combiné
            num_classes = 27
            
            proba_lstm = keras.layers.Input(shape=(num_classes,))
            proba_vgg16 = keras.layers.Input(shape=(num_classes,))
            
            weighted_proba = keras.layers.Lambda(
                lambda x: best_weights[0] * x[0] + best_weights[1] * x[1]
            )([proba_lstm, proba_vgg16])
            
            concatenate_model = keras.models.Model(
                inputs=[proba_lstm, proba_vgg16], outputs=weighted_proba
            )
            
            # Enregistrer le modèle
            concat_model_path = os.path.join(MODELS_DIR, "concatenate.h5")
            concatenate_model.save(concat_model_path)
            mlflow.log_artifact(concat_model_path)
            
            print("Optimisation terminée avec succès!")
        
        except Exception as e:
            print(f"Erreur lors de l'optimisation du modèle: {str(e)}")
            mlflow.log_param("error", str(e))
            sys.exit(1)

if __name__ == "__main__":
    main()
