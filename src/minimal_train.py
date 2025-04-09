"""
Script minimal d'entraînement pour tester la chaîne MLOps
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import json
import mlflow
import mlflow.keras

print("Démarrage de l'entraînement minimal...")

# Créer les répertoires nécessaires
os.makedirs("models", exist_ok=True)

# Charger un petit échantillon de données
try:
    data_path = "/app/data/external/X_train_update.csv"
    target_path = "/app/data/external/Y_train_CVw08PX.csv"
    
    print(f"Chargement des données depuis {data_path}")
    data = pd.read_csv(data_path)
    data["description"] = data["designation"] + " " + data["description"].fillna("")
    data = data.drop(["Unnamed: 0", "designation"], axis=1)
    
    print(f"Chargement des cibles depuis {target_path}")
    target = pd.read_csv(target_path)
    target = target.drop(["Unnamed: 0"], axis=1)
    
    # Créer un mapping pour les classes
    modalite_mapping = {
        int(modalite): i for i, modalite in enumerate(target["prdtypecode"].unique())
    }
    target["prdtypecode"] = target["prdtypecode"].replace(modalite_mapping)
    
    # Sauvegarder le mapping sous forme de chaînes pour JSON
    json_mapping = {str(k): v for k, v in modalite_mapping.items()}
    with open("models/mapper.json", "w") as f:
        json.dump(json_mapping, f)
    
    # Fusionner les données
    df = pd.concat([data, target], axis=1)
    
    # Prendre un petit échantillon
    df_sample = df.sample(500, random_state=42)
    
    # Séparer en train/val
    train_data = df_sample.sample(frac=0.8, random_state=42)
    val_data = df_sample.drop(train_data.index)
    
    X_train = train_data.drop("prdtypecode", axis=1)
    y_train = train_data["prdtypecode"]
    X_val = val_data.drop("prdtypecode", axis=1)
    y_val = val_data["prdtypecode"]
    
    # Ajouter les chemins d'images
    image_base_path = "/app/data/external/images/image_train"  # Chemin corrigé
    X_train["image_path"] = image_base_path + "/image_" + X_train["imageid"].astype(str) + "_product_" + X_train["productid"].astype(str) + ".jpg"
    X_val["image_path"] = image_base_path + "/image_" + X_val["imageid"].astype(str) + "_product_" + X_val["productid"].astype(str) + ".jpg"

    # Vérifier si les images existent
    X_train = X_train[X_train["image_path"].apply(os.path.exists)]
    X_val = X_val[X_val["image_path"].apply(os.path.exists)]
    
    # Mettre à jour y_train et y_val
    y_train = y_train.loc[X_train.index]
    y_val = y_val.loc[X_val.index]
    
    print(f"Données préparées: {len(X_train)} échantillons d'entraînement, {len(X_val)} échantillons de validation")
    
    # Créer un modèle très simple
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    
    # Configurer MLflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"))
    mlflow.set_experiment("minimal_model")
    
    with mlflow.start_run(run_name="minimal_run"):
        # Enregistrer les paramètres
        mlflow.log_param("sample_size", len(X_train))
        
        # Créer un modèle très simple
        model = Sequential([
            Dense(64, activation='relu', input_shape=(1,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(len(modalite_mapping), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Entraîner très brièvement
        dummy_input = np.random.random((len(X_train), 1))
        dummy_val = np.random.random((len(X_val), 1))
        
        history = model.fit(
            dummy_input,
            y_train,
            epochs=1,
            batch_size=32,
            validation_data=(dummy_val, y_val)
        )
        
        # Enregistrer les métriques
        for k, v in history.history.items():
            mlflow.log_metric(k, v[0])
            
        # Sauvegarder le modèle
        model.save("models/minimal_model.h5")
        mlflow.keras.log_model(model, "minimal_model")
        
        print("Modèle minimal entraîné et sauvegardé avec succès!")
        
except Exception as e:
    print(f"Erreur lors de l'entraînement minimal: {str(e)}")
    sys.exit(1)
