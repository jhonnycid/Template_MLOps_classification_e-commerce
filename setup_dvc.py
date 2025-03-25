#!/usr/bin/env python
"""
Script pour initialiser DVC dans le projet.
Ce script crée les fichiers et dossiers nécessaires pour le versionnement 
des données et des modèles avec DVC.
"""

import os
import subprocess
import sys

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
    """Fonction principale pour l'initialisation de DVC."""
    # Vérifier si dvc est installé
    if run_command("dvc --version") != 0:
        print("DVC n'est pas installé. Installation en cours...")
        run_command("pip install dvc")
    
    # Vérifier si git est initialisé
    if not os.path.exists(".git"):
        print("Git n'est pas initialisé. Initialisation en cours...")
        run_command("git init")
    
    # Initialiser DVC
    if not os.path.exists(".dvc"):
        print("Initialisation de DVC...")
        run_command("dvc init")
    
    # Créer les répertoires nécessaires
    directories = [
        "data/raw",
        "data/raw/image_train",
        "data/raw/image_test",
        "data/preprocessed",
        "data/preprocessed/image_train",
        "data/preprocessed/image_test",
        "models",
        "logs"
    ]
    
    for directory in directories:
        create_directory(directory)
    
    # Ajouter les données et les modèles à DVC
    if not os.path.exists("data.dvc"):
        print("Ajout des données à DVC...")
        run_command("dvc add data")
    
    if not os.path.exists("models.dvc"):
        print("Ajout des modèles à DVC...")
        run_command("dvc add models")
    
    # Ajouter les fichiers .dvc à git
    print("Ajout des fichiers .dvc à git...")
    run_command("git add *.dvc .dvc .gitignore")
    
    print("\nDVC a été initialisé avec succès.")
    print("Ajoutez maintenant une remote pour stocker vos données:")
    print("  dvc remote add -d myremote /chemin/vers/stockage")
    print("  git add .dvc/config")
    print("  git commit -m 'Configure DVC remote storage'")

if __name__ == "__main__":
    main()
