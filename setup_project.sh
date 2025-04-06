#!/bin/bash

# Script de configuration du projet MLOps

# Couleurs pour les messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction pour afficher un message de succès
success_message() {
    echo -e "${GREEN}✔ $1${NC}"
}

# Fonction pour afficher un message d'avertissement
warning_message() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Vérifier si Python est installé
echo "Vérification des prérequis..."
if ! command -v python3 &> /dev/null; then
    echo "Python 3 n'est pas installé. Veuillez l'installer."
    exit 1
fi

# Créer un environnement virtuel
echo "Création de l'environnement virtuel..."
python3 -m venv venv
source venv/bin/activate

# Mise à jour de pip
pip install --upgrade pip

# Installation des dépendances
echo "Installation des dépendances..."
pip install -r requirements.txt

# Vérification de la compatibilité MPS
echo "Vérification de la compatibilité MPS..."
python3 -c "
import torch
if torch.backends.mps.is_available():
    print('MPS est disponible sur ce système.')
else:
    print('MPS non disponible.')
"

# Configuration initiale de MLflow
echo "Configuration de MLflow..."
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns &
MLFLOW_PID=$!

# Création des répertoires nécessaires
mkdir -p data/raw data/preprocessed models logs

success_message "Configuration du projet terminée !"
warning_message "N'oubliez pas de modifier params.yaml pour configurer use_mps"

# Optional: Tuer le serveur MLflow à la fin du script
trap "kill $MLFLOW_PID" EXIT
