#!/bin/bash

### Adding processed files to repository
dvc add /opt/airflow/data/processed
git add /opt/airflow/data/processed.dvc
git commit -m "Ajout des données avec DVC via workflow Airflow"
dvc push
git push

###adding models
dvc add /opt/airflow/data/models
git add /opt/airflow/data/models.dvc /opt/airflow/data/.gitignore
git commit -m "Ajout des modèles avec DVC via workflow Airflow"
dvc push
git push
