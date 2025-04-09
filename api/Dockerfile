FROM python:3.8-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    pkg-config \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libhdf5-cpp-103 \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Créer un répertoire pour les modèles
RUN mkdir -p /app/models

# Exposer le port sur lequel l'API sera accessible
EXPOSE 8000

# Lancer l'API avec Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]