#!/usr/bin/env python
"""
Script pour initialiser MLflow dans le projet.
Ce script crée les dossiers nécessaires pour MLflow et démarre un serveur MLflow.
"""

import os
import subprocess
import sys
import yaml

def run_command(command):
    """Exécute une commande shell et retourne le résultat."""
    print(f"Exécution de: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    
    if stdout:
        print(stdout.decode())
    if stderr:
        print(f"ERREUR: {stderr.decode()}", file=sys.stderr)
    
    return process.returncode

def create_directory(path):
    """Crée un répertoire s'il n'existe pas."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Création du répertoire: {path}")
    else:
        print(f"Le répertoire existe déjà: {path}")

def main():
    """Fonction principale pour l'initialisation de MLflow."""
    # Vérifier si mlflow est installé
    if run_command("mlflow --version") != 0:
        print("MLflow n'est pas installé. Installation en cours...")
        run_command("pip install mlflow")
    
    # Créer les répertoires pour MLflow
    create_directory("mlflow")
    create_directory("mlflow/artifacts")
    
    # Lire la configuration MLflow
    config_path = "mlflow/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Créer les répertoires pour les artifacts de chaque expérience
        for exp_key, exp_config in config.get("experiments", {}).items():
            artifact_location = exp_config.get("artifact_location")
            if artifact_location:
                # Enlever le préfixe /mlflow/ car il est relatif à notre structure
                clean_path = artifact_location.replace("/mlflow/", "")
                create_directory(os.path.join("mlflow", clean_path))
    
    # Définir les variables d'environnement
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
    
    # Créer un script de démarrage pour MLflow
    with open("start_mlflow.sh", "w") as f:
        f.write("""#!/bin/bash
# Démarre un serveur MLflow
export MLFLOW_TRACKING_URI=http://localhost:5001
mlflow server \\
    --backend-store-uri sqlite:///mlflow/mlflow.db \\
    --default-artifact-root ./mlflow/artifacts \\
    --host 0.0.0.0 \\
    --port 5001
""")
    
    # Rendre le script exécutable
    run_command("chmod +x start_mlflow.sh")
    
    print("\nMLflow a été configuré avec succès.")
    print("Pour démarrer le serveur MLflow, exécutez:")
    print("  ./start_mlflow.sh")
    print("Ou avec Docker:")
    print("  docker-compose up mlflow")

if __name__ == "__main__":
    main()
