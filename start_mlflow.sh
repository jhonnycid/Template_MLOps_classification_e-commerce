#!/bin/bash
# Démarre un serveur MLflow
export MLFLOW_TRACKING_URI=http://localhost:5001
mlflow server \
    --backend-store-uri sqlite:///mlflow/mlflow.db \
    --default-artifact-root ./mlflow/artifacts \
    --host 0.0.0.0 \
    --port 5001
