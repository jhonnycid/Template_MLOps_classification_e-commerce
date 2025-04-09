#!/bin/bash

# Configuration des données
echo "Configuration des liens vers les données..."
bash /app/training/setup_data.sh

# Lancement de l'entraînement
echo "Démarrage de l'entraînement..."
python /app/src/main.py