#!/bin/bash

# Script pour lancer un entraînement minimal

# Configuration des données
echo "Configuration des liens vers les données..."
bash /app/training/setup_data.sh

# Paramètres réduits pour l'entraînement
export LSTM_EPOCHS=1
export VGG_EPOCHS=1
export BATCH_SIZE=32
export SAMPLES_PER_CLASS=10

echo "Paramètres d'entraînement minimaux:"
echo "  - LSTM Epochs: $LSTM_EPOCHS"
echo "  - VGG Epochs: $VGG_EPOCHS"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Samples per class: $SAMPLES_PER_CLASS"

# Lancement de l'entraînement avec sous-ensemble très petit
echo "Démarrage de l'entraînement minimal..."
python /app/src/main_with_params.py