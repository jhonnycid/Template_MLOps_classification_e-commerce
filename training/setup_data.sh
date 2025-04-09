#!/bin/bash

# Créer les répertoires nécessaires s'ils n'existent pas
mkdir -p /app/data/preprocessed/images/images/

# Créer des liens symboliques pour les fichiers CSV
ln -sf /app/data/external/X_train_update.csv /app/data/preprocessed/X_train_update.csv
ln -sf /app/data/external/X_test_update.csv /app/data/preprocessed/X_test_update.csv
ln -sf /app/data/external/Y_train_CVw08PX.csv /app/data/preprocessed/Y_train_CVw08PX.csv

# Créer des liens symboliques pour les répertoires d'images
ln -sf /app/data/external/images/image_train /app/data/preprocessed/images/images/image_train
ln -sf /app/data/external/images/image_test /app/data/preprocessed/images/images/image_test

echo "Configuration des données terminée."