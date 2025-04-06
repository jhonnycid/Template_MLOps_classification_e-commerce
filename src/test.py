import pandas as pd
from pathlib import Path

# Charger les jeux de données
train_df = pd.read_csv('/Users/danhang/Documents/PROJET TRANSITION PRO/Formation IA/DataScientest/Projet Rakuten/archive/images/X_train_update.csv')
test_df = pd.read_csv('/Users/danhang/Documents/PROJET TRANSITION PRO/Formation IA/DataScientest/Projet Rakuten/archive/images/X_test_update.csv')

# Supprimer les colonnes inutiles (ex : cible, ID...)
columns_to_keep = ["designation","description"]  # Tu peux aussi ajouter 'image_path' ou autres si utiles
train_df = train_df[columns_to_keep].copy()
test_df = test_df[columns_to_keep].copy()

# Nettoyage optionnel (ex : valeurs manquantes)
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# Création des fichiers pour Evidently
Path("data").mkdir(parents=True, exist_ok=True)
train_df.to_csv("data/reference.csv", index=False)
test_df.to_csv("data/current.csv", index=False)

print("✅ Fichiers 'reference.csv' et 'current.csv' créés dans /data")
