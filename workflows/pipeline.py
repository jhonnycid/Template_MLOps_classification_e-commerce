from flytekit import task, workflow
from flytekit.core.data_persistence import FileAccessProvider
import subprocess
import os
from pathlib import Path

# -------------- CONFIGURATION --------------

DATA_DIR = "data"
MODELS_DIR = "models"

# -------------- DVC TASKS --------------

@task
def dvc_stage_import_data(run_dvc: bool = False) -> str:
    if run_dvc:
        subprocess.run(["dvc", "repro", "import_data"], check=True)
    return f"{DATA_DIR}/raw"


@task
def dvc_stage_preprocess(run_dvc: bool = False) -> str:
    if run_dvc:
        subprocess.run(["dvc", "repro", "preprocess"], check=True)
    return f"{DATA_DIR}/preprocessed"


@task
def dvc_stage_train(run_dvc: bool = False) -> tuple[str, str]:
    if run_dvc:
        subprocess.run(["dvc", "repro", "train"], check=True)
    # Sinon on vérifie que les modèles sont là
    lstm_path = f"{MODELS_DIR}/best_lstm_model.pth"
    vgg_path = f"{MODELS_DIR}/best_vgg16_model.pth"
    if not os.path.exists(lstm_path) or not os.path.exists(vgg_path):
        raise FileNotFoundError("Les modèles ne sont pas présents. Lancer avec run_dvc=True.")
    return lstm_path, vgg_path


# -------------- PREDICTION TASK --------------

@task
def run_prediction(lstm_model_path: str, vgg_model_path: str) -> str:
    print("🔎 Lancement des prédictions avec les modèles existants")
    subprocess.run([
        "python", "src/predict.py",
        "--dataset_path", "data/preprocessed/X_test_update.csv",
        "--images_path", "data/preprocessed/image_test"
    ], check=True)
    return "✅ Prédiction terminée"


# -------------- MAIN PIPELINE --------------

@workflow
def rakuten_pipeline(run_dvc_import: bool = False, run_dvc_preprocess: bool = False, run_dvc_train: bool = False) -> str:
    raw_path = dvc_stage_import_data(run_dvc=run_dvc_import)
    preprocessed_path = dvc_stage_preprocess(run_dvc=run_dvc_preprocess)
    lstm_model, vgg_model = dvc_stage_train(run_dvc=run_dvc_train)
    result = run_prediction(lstm_model_path=lstm_model, vgg_model_path=vgg_model)
    return result
