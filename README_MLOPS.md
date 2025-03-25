# Infrastructure MLOps pour Classification E-commerce Rakuten

Ce projet met en place une infrastructure MLOps complète pour un modèle de classification d'articles e-commerce. Il utilise une architecture hybride combinant un modèle LSTM pour l'analyse de texte et un modèle VGG16 pour l'analyse d'images.

## Architecture MLOps

L'infrastructure MLOps de ce projet comprend les composants suivants :

### 1. Versioning et suivi des données et modèles (DVC)

- **[DVC](https://dvc.org/)** (Data Version Control) est utilisé pour le versionnement des données et des modèles.
- Les données brutes et prétraitées sont suivies avec DVC.
- Les modèles entraînés sont également versionnés avec DVC.
- Le pipeline d'entraînement est défini dans `dvc.yaml`.

### 2. Suivi des expériences (MLflow)

- **[MLflow](https://mlflow.org/)** est utilisé pour le suivi des expériences.
- Les paramètres, métriques et artefacts sont enregistrés pour chaque expérience.
- Le serveur MLflow est accessible via `http://localhost:5001`.
- Trois expériences principales sont configurées :
  - `rakuten-classification` : pour l'entraînement des modèles
  - `rakuten-classification-evaluation` : pour l'évaluation des modèles
  - `rakuten-prediction-api` : pour le suivi des prédictions en production

### 3. Pipeline CI/CD (GitHub Actions)

- **GitHub Actions** est configuré pour l'intégration et le déploiement continus.
- Le workflow est défini dans `.github/workflows/ml-pipeline.yml`.
- Tests automatiques et linting à chaque pull request.
- Construction et publication automatique d'images Docker lors des push sur la branche principale.

### 4. Containerisation (Docker)

- **Docker** est utilisé pour containeriser l'application.
- Le `Dockerfile` définit l'environnement d'exécution.
- `docker-compose.yml` configure les services :
  - `model-api` : API de prédiction
  - `mlflow` : Serveur MLflow pour le suivi des expériences

## Pipeline de données et d'entraînement

Le pipeline complet est défini dans `dvc.yaml` et comprend les étapes suivantes :

1. **import_data** : Importation des données brutes
2. **preprocess** : Prétraitement des données textuelles et d'images
3. **train** : Entraînement des modèles LSTM, VGG16 et de leur combinaison
4. **evaluate** : Évaluation du modèle sur un jeu de test

## API de prédiction

Une API Flask est fournie pour servir les prédictions :

- Endpoint `/predict` : Prend une description textuelle et une image, retourne la catégorie prédite
- Endpoint `/health` : Pour les vérifications de santé du service

## Instructions d'utilisation

### Installation

1. **Configuration initiale** :
   ```bash
   ./setup_project.sh
   ```

2. **Activation de l'environnement** :
   ```bash
   conda activate Rakuten-project
   ```

### Utilisation du pipeline DVC

1. **Exécution du pipeline complet** :
   ```bash
   dvc repro
   ```

2. **Exécution d'une étape spécifique** :
   ```bash
   dvc repro <étape>
   ```

3. **Visualisation du pipeline** :
   ```bash
   dvc dag
   ```

### Suivi des expériences avec MLflow

1. **Démarrage du serveur MLflow** :
   ```bash
   ./start_mlflow.sh
   ```

2. **Accès à l'interface MLflow** :
   Ouvrez un navigateur et accédez à `http://localhost:5001`

### Utilisation de Docker

1. **Construction et démarrage des services** :
   ```bash
   docker-compose up --build
   ```

2. **Accès à l'API** :
   L'API est accessible à `http://localhost:5000`

3. **Accès à MLflow** :
   Le serveur MLflow est accessible à `http://localhost:5001`

## Bonnes pratiques MLOps

Ce projet suit les bonnes pratiques MLOps suivantes :

1. **Reproductibilité** : Versionnement du code, des données et des modèles
2. **Automatisation** : Pipeline CI/CD et tests automatisés
3. **Monitoring** : Suivi des expériences et des performances
4. **Déployabilité** : Containerisation pour un déploiement facile
5. **Évolutivité** : Architecture modulaire et extensible

## Architecture technique

```
├── .dvc/               # Configuration DVC
├── .github/workflows/  # Workflows GitHub Actions
├── data/               # Données (gérées par DVC)
│   ├── raw/            # Données brutes
│   └── preprocessed/   # Données prétraitées
├── mlflow/             # Configuration et artefacts MLflow
├── models/             # Modèles entraînés (gérés par DVC)
├── notebooks/          # Notebooks Jupyter pour l'exploration
├── src/                # Code source
│   ├── data/           # Scripts de gestion des données
│   ├── features/       # Scripts de prétraitement
│   ├── models/         # Scripts d'entraînement
│   ├── api.py          # API de prédiction
│   ├── main.py         # Script principal d'entraînement
│   └── evaluate.py     # Script d'évaluation
├── Dockerfile          # Configuration Docker
├── docker-compose.yml  # Configuration Docker Compose
├── dvc.yaml            # Définition du pipeline DVC
├── params.yaml         # Paramètres du modèle
└── requirements.txt    # Dépendances Python
```

## Contribution

Pour contribuer à ce projet :

1. Forker le dépôt
2. Créer une branche pour votre fonctionnalité
3. Pousser vos modifications
4. Créer une Pull Request

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
