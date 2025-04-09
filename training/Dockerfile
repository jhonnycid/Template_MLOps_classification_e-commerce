FROM python:3.8-slim

WORKDIR /app

# Installer les dépendances pour TensorFlow et autres bibliothèques
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    python3-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-cpp-103 \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de requirements
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Télécharger les ressources NLTK nécessaires
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Créer les répertoires pour les données et les modèles
RUN mkdir -p /app/data /app/models /app/logs

# Le code sera monté comme volume
CMD ["python", "src/main.py"]