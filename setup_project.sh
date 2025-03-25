#!/bin/bash
# Script d'initialisation complète du projet

# Afficher un message de bienvenue
echo "=== Initialisation du projet de classification e-commerce Rakuten ==="
echo "Ce script va configurer l'environnement MLOps avec DVC, MLflow et Docker."
echo

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "Python 3 n'est pas installé. Veuillez l'installer avant de continuer."
    exit 1
fi

# Vérifier si pip est installé
if ! command -v pip &> /dev/null; then
    echo "pip n'est pas installé. Veuillez l'installer avant de continuer."
    exit 1
fi

# Vérifier si Docker est installé
if ! command -v docker &> /dev/null; then
    echo "Docker n'est pas installé. L'installation continuera sans Docker."
    HAS_DOCKER=false
else
    HAS_DOCKER=true
fi

# Vérifier si Docker Compose est installé
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose n'est pas installé. L'installation continuera sans Docker Compose."
    HAS_DOCKER_COMPOSE=false
else
    HAS_DOCKER_COMPOSE=true
fi

# Créer et activer l'environnement conda
echo "=== Configuration de l'environnement conda ==="
if ! command -v conda &> /dev/null; then
    echo "conda n'est pas installé. Veuillez l'installer avant de continuer."
    echo "Vous pouvez télécharger Miniconda depuis: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Création de l'environnement si nécessaire
if ! conda env list | grep -q "Rakuten-project"; then
    echo "Création de l'environnement conda 'Rakuten-project'..."
    conda create -y -n "Rakuten-project" python=3.9
fi

# Activer l'environnement (utilisation de source pour bash)
echo "Activation de l'environnement conda 'Rakuten-project'..."
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "Rakuten-project"

# Installation des dépendances
echo "=== Installation des dépendances ==="
conda install -y pip
pip install -r requirements.txt
pip install dvc mlflow flask

# Configuration de DVC
echo "=== Configuration de DVC ==="
python setup_dvc.py

# Configuration de MLflow
echo "=== Configuration de MLflow ==="
python setup_mlflow.py

# Configuration Docker si disponible
if [ "$HAS_DOCKER" = true ] && [ "$HAS_DOCKER_COMPOSE" = true ]; then
    echo "=== Configuration de Docker ==="
    echo "Construction des images Docker..."
    docker-compose build
    
    echo "Vérification des images Docker..."
    docker images | grep rakuten
    
    echo "Les containers Docker sont prêts à être démarrés avec 'docker-compose up'"
fi

# Fin de l'installation
echo
echo "=== Installation terminée avec succès ==="
echo "Pour démarrer le développement:"
echo "1. Activez l'environnement: conda activate Rakuten-project"
echo "2. Importez les données: python src/data/import_raw_data.py"
echo "3. Préparez les données: python src/data/make_dataset.py data/raw data/preprocessed"
echo
echo "Pour entraîner le modèle:"
echo "  python src/main.py"
echo
echo "Pour démarrer MLflow:"
echo "  ./start_mlflow.sh"
echo
echo "Pour démarrer l'API de prédiction:"
echo "  python src/api.py"
echo
echo "Pour utiliser Docker:"
echo "  docker-compose up"
echo
echo "Bon développement!"
