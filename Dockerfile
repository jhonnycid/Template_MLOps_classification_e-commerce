FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    git \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers requirements et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers du projet
COPY . .

# Créer les répertoires nécessaires
RUN mkdir -p data/raw data/preprocessed models logs

# Exposer le port pour l'API
EXPOSE 5000

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models
ENV DATA_PATH=/app/data
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Commande par défaut pour démarrer l'API
CMD ["python", "src/api.py"]
