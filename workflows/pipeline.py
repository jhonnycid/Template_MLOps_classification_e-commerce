from flytekit import task, workflow
import os
import subprocess


@task
def train_model():
    # Suppose que le modèle est déjà entraîné, on ne relance rien ici
    print("🔄 Skip entraînement — modèle déjà disponible.")
    return "models/best_lstm_model.pth", "models/best_vgg16_model.pth"


@task
def predict_task(lstm_model_path: str, vgg_model_path: str) -> str:
    subprocess.run([
        "python", "src/predict.py",
        "--dataset_path", "data/preprocessed/X_test_update.csv",
        "--images_path", "data/preprocessed/image_test"
    ], check=True)
    return "✅ Prédiction terminée"


@workflow
# def prediction_pipeline() -> str:
#     lstm_model_path, vgg_model_path = train_model()
#     return predict_task(lstm_model_path, vgg_model_path)
def prediction_pipeline() -> str:
    return predict_task(
        lstm_model_path="models/best_lstm_model.pth",
        vgg_model_path="models/best_vgg16_model.pth"
    )