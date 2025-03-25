# Rakuten E-commerce Classification avec PyTorch

Ce projet est une implémentation MLOps complète d'un système de classification d'articles e-commerce Rakuten utilisant PyTorch. Il combine un modèle LSTM pour l'analyse de texte et un modèle VGG16 pour l'analyse d'images.

## Architecture du modèle

Le système utilise une approche hybride qui combine deux types de modèles :

1. **Modèle LSTM pour le texte** :
   - Prend en entrée la description textuelle des produits
   - Utilise un embedding et une couche LSTM
   - Prétraite le texte (nettoyage, tokenisation, etc.)

2. **Modèle VGG16 pour les images** :
   - Utilise le modèle pré-entraîné VGG16 adapté à notre cas d'usage
   - Traite les images de produits redimensionnées à 224x224 pixels
   - Utilise le transfer learning en figeant les couches de base

3. **Ensemble pondéré** :
   - Combine les prédictions des deux modèles avec des poids optimisés
   - Les poids sont déterminés pour maximiser la précision sur un ensemble de validation

## Différences avec la version TensorFlow

Cette implémentation utilise PyTorch au lieu de TensorFlow, avec quelques différences notables :

- Utilisation de `torch.nn.Module` pour les définitions de modèles
- Boucles d'entraînement explicites avec PyTorch
- Utilisation de `DataLoader` pour le chargement efficace des données
- Implémentation des `Dataset` personnalisés pour le texte et les images
- Sauvegarde des modèles avec `torch.save` au lieu du format h5

## Structure du projet

```
├── .dvc/               # Configuration DVC
├── .github/workflows/  # Workflows GitHub Actions
├── mlflow/             # Configuration et artefacts MLflow
├── models/             # Modèles entraînés (gérés par DVC)
│   ├── best_lstm_model.pth  # Modèle LSTM PyTorch
│   ├── best_vgg16_model.pth # Modèle VGG16 PyTorch
│   └── ...
├── src/
│   ├── api.py          # API de prédiction
│   ├── main.py         # Script principal d'entraînement
│   ├── evaluate.py     # Script d'évaluation
│   ├── models/
│   │   └── train_model.py  # Définitions des modèles PyTorch
│   └── ...
```

## Prérequis

- Python 3.9+
- PyTorch 1.9.0+
- CUDA (recommandé pour l'accélération GPU)

## Installation

```bash
# Cloner le dépôt
git clone <repo-url>
cd <repo-directory>

# Créer et activer l'environnement conda
conda create -n "Rakuten-PyTorch" python=3.9
conda activate Rakuten-PyTorch

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### Entraînement du modèle

```bash
python src/main.py
```

### Évaluation du modèle

```bash
python src/evaluate.py
```

### Prédictions avec le modèle

```bash
python src/predict.py --dataset_path "data/preprocessed/X_test_update.csv" --images_path "data/preprocessed/image_test"
```

### API de prédiction

```bash
python src/api.py
```

## API REST

L'API expose deux endpoints :

- **POST /predict** : Prédit la catégorie d'un produit à partir de sa description et de son image.
  ```json
  {
    "description": "Description du produit",
    "image": "base64_encoded_image_data_or_path"
  }
  ```

- **GET /health** : Vérifie l'état de l'API.

## Tracking MLOps

Ce projet utilise :

- **MLflow** pour le suivi des expériences et des métriques
- **DVC** pour le versionnement des données et des modèles
- **Docker** pour la containerisation
- **GitHub Actions** pour CI/CD

## Performances

Sur un jeu de test représentatif, le modèle hybride atteint généralement :
- Précision : ~90%
- Les poids optimaux sont généralement autour de 60% pour le modèle texte et 40% pour le modèle image

## Contribution

Les contributions sont les bienvenues ! Veuillez consulter les guidelines de contribution dans le fichier CONTRIBUTING.md.

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
