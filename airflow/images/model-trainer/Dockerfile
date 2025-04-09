FROM python:3.8-slim

WORKDIR /app

# Installer les dépendances pour TensorFlow et autres bibliothèques
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier les scripts d'entraînement
COPY train_model.py .
COPY optimize_model.py .

# Créer les répertoires pour les données et les modèles
RUN mkdir -p /app/data /app/models

CMD ["python", "train_model.py"]