#!/bin/bash

# Script pour lancer l'entraînement avec des paramètres personnalisés

# Paramètres par défaut
LSTM_EPOCHS=${LSTM_EPOCHS:-3}
VGG_EPOCHS=${VGG_EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-32}
SAMPLES_PER_CLASS=${SAMPLES_PER_CLASS:-50}

# Configuration des données
echo "Configuration des liens vers les données..."
bash /app/training/setup_data.sh

# Configuration des variables d'environnement pour les modèles
export LSTM_EPOCHS=$LSTM_EPOCHS
export VGG_EPOCHS=$VGG_EPOCHS
export BATCH_SIZE=$BATCH_SIZE
export SAMPLES_PER_CLASS=$SAMPLES_PER_CLASS

echo "Paramètres d'entraînement:"
echo "  - LSTM Epochs: $LSTM_EPOCHS"
echo "  - VGG Epochs: $VGG_EPOCHS"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Samples per class: $SAMPLES_PER_CLASS"

# Lancement de l'entraînement
echo "Démarrage de l'entraînement..."
python /app/src/main_with_params.py