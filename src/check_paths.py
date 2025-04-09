"""
Script pour vérifier les chemins d'accès aux fichiers de données
"""

import os
import pandas as pd
import glob

# Chemins à vérifier
data_path = "/app/data/external/X_train_update.csv"
target_path = "/app/data/external/Y_train_CVw08PX.csv"
image_base_path = "/app/data/external/images/image_train"  # Chemin corrigé

print("=== VÉRIFICATION DES CHEMINS ===")

# Vérifier les fichiers CSV
print(f"\n1. Vérification des fichiers CSV")
print(f"Fichier de données: {data_path} - {'Existe' if os.path.exists(data_path) else 'existe PAS'}")
print(f"Fichier de cibles: {target_path} - {'Existe' if os.path.exists(target_path) else 'existe PAS'}")

# Vérifier les dossiers d'images
print(f"\n2. Vérification du dossier d'images")
print(f"Dossier d'images: {image_base_path} - {'Existe' if os.path.exists(image_base_path) else 'existe PAS'}")

# Si le dossier d'images existe, compter les images
if os.path.exists(image_base_path):
    image_files = glob.glob(os.path.join(image_base_path, "*.jpg"))
    print(f"Nombre d'images JPEG trouvées: {len(image_files)}")
    
    # Montrer quelques exemples de noms de fichiers
    if image_files:
        print("Exemples de noms de fichiers:")
        for file in image_files[:5]:
            print(f"  - {os.path.basename(file)}")
else:
    print("Le dossier d'images n'existe pas, impossible de compter les fichiers.")

# Charger les données et vérifier les chemins d'images
if os.path.exists(data_path):
    print(f"\n3. Vérification des données")
    data = pd.read_csv(data_path)
    print(f"Nombre d'enregistrements dans X_train_update.csv: {len(data)}")
    
    # Générer quelques exemples de chemins d'images
    print("Exemples de chemins d'images attendus:")
    for i, row in data.iloc[:5].iterrows():
        img_path = f"{image_base_path}/image_{row['imageid']}_product_{row['productid']}.jpg"
        print(f"  - {img_path} - {'Existe' if os.path.exists(img_path) else 'existe PAS'}")

# Vérifier la structure du répertoire
print(f"\n4. Structure du répertoire /app/data/external")
os.system("ls -la /app/data/external")

print(f"\n5. Structure du répertoire /app/data/external/images (si existant)")
if os.path.exists("/app/data/external/images"):
    os.system("ls -la /app/data/external/images")
else:
    print("Le répertoire /app/data/external/images n'existe pas")

print("\n=== FIN DE LA VÉRIFICATION ===")
