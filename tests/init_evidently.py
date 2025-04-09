import requests
import json
import numpy as np
import random
import os
import time

# URL du service Evidently
evidently_url = "http://localhost:8050"

# Chemins d'images fictifs pour les exemples
image_paths = [f"/app/data/external/image_{i}.jpg" for i in range(100)]

# Descriptions fictives pour les exemples
descriptions = [
    "Chaussures de sport Nike taille 42",
    "T-shirt à manches courtes noir taille L",
    "Pantalon jean bleu",
    "Robe d'été fleurie",
    "Sac à main en cuir brun",
    "Montre analogique argentée",
    "Veste légère imperméable",
    "Casquette baseball noire",
    "Enceinte Bluetooth portable",
    "Écouteurs sans fil"
]

# Générer des données de référence
def generate_data(num_samples=100):
    data = []
    for _ in range(num_samples):
        sample = {
            "description": random.choice(descriptions),
            "image_path": random.choice(image_paths),
            "prediction": random.randint(0, 26),
            "confidence": round(random.uniform(0.5, 0.99), 2),
            "processing_time": round(random.uniform(0.1, 2.0), 3),
            "timestamp": time.time()
        }
        data.append(sample)
    return data

# Fonction pour initialiser les données de référence
def initialize_reference_data():
    print("Initialisation des données de référence...")
    reference_data = generate_data(num_samples=200)
    
    try:
        response = requests.post(
            f"{evidently_url}/drift/reference",
            json={"data": reference_data}
        )
        
        if response.status_code == 200:
            print("✅ Données de référence envoyées avec succès")
            print(f"Réponse: {response.json()}")
        else:
            print(f"❌ Erreur lors de l'envoi des données de référence: {response.status_code}")
            print(f"Détails: {response.text}")
    except Exception as e:
        print(f"❌ Exception lors de l'envoi des données de référence: {str(e)}")

# Fonction pour détecter une dérive avec de nouvelles données
def detect_drift():
    print("\nDétection de dérive avec de nouvelles données...")
    # Générer des données légèrement différentes pour simuler une dérive
    current_data = generate_data(num_samples=100)
    
    # Modifier légèrement les distributions pour simuler une dérive
    for sample in current_data:
        # Augmenter la probabilité d'avoir certaines prédictions
        if random.random() < 0.3:
            sample["prediction"] = random.randint(20, 26)
        # Modifier légèrement la confiance
        sample["confidence"] = round(random.uniform(0.3, 0.95), 2)
    
    try:
        response = requests.post(
            f"{evidently_url}/drift/detect",
            json={"data": current_data}
        )
        
        if response.status_code == 200:
            print("✅ Données de test envoyées avec succès")
            result = response.json()
            print(f"Dérive détectée: {'Oui' if result['drift_detected'] else 'Non'}")
            print(f"Score de dérive: {result['drift_score']:.2f}")
        else:
            print(f"❌ Erreur lors de l'envoi des données de test: {response.status_code}")
            print(f"Détails: {response.text}")
    except Exception as e:
        print(f"❌ Exception lors de l'envoi des données de test: {str(e)}")

# Vérifier si le service Evidently est accessible
def check_service():
    try:
        response = requests.get(f"{evidently_url}/health")
        if response.status_code == 200:
            print("✅ Service Evidently accessible")
            return True
        else:
            print(f"❌ Service Evidently inaccessible: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Service Evidently inaccessible: {str(e)}")
        return False

# Exécuter les opérations
if __name__ == "__main__":
    print("=== Initialisation du service Evidently ===")
    
    # Attendre que le service soit disponible
    max_attempts = 10
    for attempt in range(max_attempts):
        if check_service():
            break
        print(f"Tentative {attempt+1}/{max_attempts} - Attente de 5 secondes...")
        time.sleep(5)
    
    # Initialiser les données de référence et détecter une dérive
    initialize_reference_data()
    detect_drift()
    
    print("\n=== Terminé ===")
    print(f"Vous pouvez maintenant accéder au dashboard Evidently à l'adresse: {evidently_url}")