from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from datetime import timedelta
from docker.types import Mount
import mlflow
import requests
import json
import os
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Chemins vers les dossiers de données et modèles
PROJECT_DIR = os.environ.get("PROJECT_DIR", "/path/to/project")
DATA_DIR = f"{PROJECT_DIR}/data"
MODELS_DIR = f"{PROJECT_DIR}/models"
MLFLOW_TRACKING_URI = "http://mlflow:5000"

with DAG(
    'rakuten_model_retraining',
    default_args=default_args,
    description='Réentraînement périodique du modèle Rakuten',
    schedule_interval=timedelta(days=7),  # Hebdomadaire
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'retraining'],
) as dag:
    
    # Tâche 1: Vérifier la disponibilité de nouvelles données
    def check_new_data():
        """Vérifie si de nouvelles données sont disponibles pour le réentraînement."""
        # Dans un scénario réel, vous pourriez vérifier une table de base de données, un bucket S3, etc.
        # Ici, nous simulons un scénario où de nouvelles données sont toujours disponibles
        return {"new_data_available": True, "data_path": f"{DATA_DIR}/preprocessed"}

    check_data_task = PythonOperator(
        task_id='check_new_data',
        python_callable=check_new_data,
    )
    
    # Tâche 2: Vérifier la dérive des données
    def check_data_drift(**kwargs):
        """Vérifie s'il y a une dérive dans les données par rapport à la référence."""
        ti = kwargs['ti']
        data_info = ti.xcom_pull(task_ids='check_new_data')
        
        if not data_info["new_data_available"]:
            return {"drift_detected": False, "drift_score": 0}
        
        try:
            # Dans un scénario réel, vous appelleriez un service Evidently ici
            # Pour ce POC, nous simulons une réponse positive
            drift_result = {
                "drift_detected": True, 
                "drift_score": 0.25,  # Score de dérive simulé
                "features_with_drift": ["feature1", "feature2"]
            }
            
            # Enregistrer le résultat de dérive dans MLflow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("data_drift_monitoring")
            
            with mlflow.start_run():
                mlflow.log_metric("drift_score", drift_result["drift_score"])
                mlflow.log_param("drift_detected", drift_result["drift_detected"])
                mlflow.log_dict(drift_result, "drift_details.json")
            
            return drift_result
        except Exception as e:
            print(f"Erreur lors de la vérification de la dérive des données: {str(e)}")
            # En cas d'erreur, nous supposons qu'il n'y a pas de dérive
            return {"drift_detected": False, "drift_score": 0, "error": str(e)}
        
    drift_check_task = PythonOperator(
        task_id='check_data_drift',
        python_callable=check_data_drift,
        provide_context=True,
    )
    
    # Tâche 3: Décider si un réentraînement est nécessaire
    def decide_retraining(**kwargs):
        """Décide si un réentraînement est nécessaire en fonction de la dérive des données."""
        ti = kwargs['ti']
        drift_info = ti.xcom_pull(task_ids='check_data_drift')
        
        # Seuil de dérive pour déclencher un réentraînement
        DRIFT_THRESHOLD = 0.1
        
        if drift_info.get("drift_detected", False) and drift_info.get("drift_score", 0) > DRIFT_THRESHOLD:
            print(f"Dérive détectée avec un score de {drift_info['drift_score']}. Réentraînement nécessaire.")
            return True
        else:
            print("Pas de dérive significative détectée. Réentraînement non nécessaire.")
            return False
        
    decision_task = PythonOperator(
        task_id='decide_retraining',
        python_callable=decide_retraining,
        provide_context=True,
    )
    
    # Tâche 4: Réentraîner le modèle LSTM
    retrain_lstm_task = DockerOperator(
        task_id='retrain_lstm_model',
        image='rakuten-mlops/model-trainer:latest',
        auto_remove=True,
        command='python /app/train_model.py --model lstm',
        mount_tmp_dir=False,
        mounts=[Mount(source=DATA_DIR, target="/app/data", type="bind"),
                Mount(source=MODELS_DIR, target="/app/models", type="bind")],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        trigger_rule='all_success',  # Exécuter uniquement si decide_retraining = True
    )
    
    # Tâche 5: Réentraîner le modèle VGG16
    retrain_vgg16_task = DockerOperator(
        task_id='retrain_vgg16_model',
        image='rakuten-mlops/model-trainer:latest',
        auto_remove=True,
        command='python /app/train_model.py --model vgg16',
        mount_tmp_dir=False,
        mounts=[Mount(source=DATA_DIR, target="/app/data", type="bind"),
                Mount(source=MODELS_DIR, target="/app/models", type="bind")],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        trigger_rule='all_success',  # Exécuter uniquement si decide_retraining = True
    )
    
    # Tâche 6: Optimiser le modèle combiné
    optimize_model_task = DockerOperator(
        task_id='optimize_combined_model',
        image='rakuten-mlops/model-trainer:latest',
        auto_remove=True,
        command='python /app/optimize_model.py',
        mount_tmp_dir=False,
        mounts=[Mount(source=DATA_DIR, target="/app/data", type="bind"),
                Mount(source=MODELS_DIR, target="/app/models", type="bind")],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
    )
    
    # Tâche 7: Évaluer et valider le modèle
    def evaluate_model(**kwargs):
        """Évalue le nouveau modèle et décide s'il doit être déployé en production."""
        try:
            # Se connecter à MLflow pour récupérer les métriques du nouveau modèle
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = mlflow.tracking.MlflowClient()
            
            # Récupérer les dernières exécutions pour les deux modèles
            lstm_runs = client.search_runs(
                experiment_ids=["1"],  # ID de l'expérience LSTM
                order_by=["start_time DESC"],
                max_results=1
            )
            
            vgg16_runs = client.search_runs(
                experiment_ids=["2"],  # ID de l'expérience VGG16
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if not lstm_runs or not vgg16_runs:
                print("Impossible de trouver les dernières exécutions MLflow.")
                return False
            
            # Récupérer les métriques de performance des nouveaux modèles
            new_lstm_accuracy = lstm_runs[0].data.metrics.get("final_val_accuracy", 0)
            new_vgg16_accuracy = vgg16_runs[0].data.metrics.get("final_val_accuracy", 0)
            
            # Récupérer le meilleur modèle en production
            try:
                production_models = client.get_latest_versions("rakuten-classifier", stages=["Production"])
                if not production_models:
                    # Pas de modèle en production, promouvoir les nouveaux
                    print("Aucun modèle en production. Les nouveaux modèles seront promus.")
                    return True
                
                # Récupérer l'accuracy des modèles en production
                production_run_id = production_models[0].run_id
                production_run = client.get_run(production_run_id)
                production_accuracy = production_run.data.metrics.get("accuracy", 0)
                
                # Calculer la précision moyenne des nouveaux modèles
                new_average_accuracy = (new_lstm_accuracy + new_vgg16_accuracy) / 2
                
                # Comparer les performances
                improvement_threshold = 0.01  # 1% d'amélioration minimum
                if new_average_accuracy > production_accuracy + improvement_threshold:
                    print(f"Les nouveaux modèles sont meilleurs: {new_average_accuracy} vs {production_accuracy}")
                    return True
                else:
                    print(f"Les nouveaux modèles ne sont pas significativement meilleurs: {new_average_accuracy} vs {production_accuracy}")
                    return False
            except Exception as e:
                print(f"Erreur lors de la comparaison avec le modèle en production: {str(e)}")
                # En cas d'erreur, nous promouvons les nouveaux modèles par défaut
                return True
            
        except Exception as e:
            print(f"Erreur lors de l'évaluation du modèle: {str(e)}")
            return False
        
    evaluation_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        provide_context=True,
    )
    
    # Tâche 8: Déployer le modèle
    def deploy_model(**kwargs):
        """Déploie le nouveau modèle en production."""
        ti = kwargs['ti']
        deploy_decision = ti.xcom_pull(task_ids='evaluate_model')
        
        if not deploy_decision:
            print("Déploiement ignoré - le nouveau modèle n'est pas meilleur que le modèle actuel.")
            return
        
        try:
            # 1. Enregistrer le modèle dans le registre MLflow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = mlflow.tracking.MlflowClient()
            
            # Récupérer les dernières exécutions
            lstm_runs = client.search_runs(
                experiment_ids=["1"],  # ID de l'expérience LSTM
                order_by=["start_time DESC"],
                max_results=1
            )
            
            # Enregistrer le modèle
            model_version = client.create_model_version(
                name="rakuten-classifier",
                source=f"runs:/{lstm_runs[0].info.run_id}/lstm_model",
                run_id=lstm_runs[0].info.run_id
            )
            
            # 2. Transition du modèle en production
            client.transition_model_version_stage(
                name="rakuten-classifier",
                version=model_version.version,
                stage="Production"
            )
            
            # 3. Mettre à jour les fichiers de modèle pour l'API
            # Dans un scénario réel, vous pourriez avoir un mécanisme pour mettre à jour l'API
            # Par exemple, en redémarrant un service ou en mettant à jour un point de montage
            
            print(f"Modèle v{model_version.version} déployé avec succès en production!")
            
            # 4. Journaliser le déploiement
            with open(f"{MODELS_DIR}/deployment_log.txt", "a") as f:
                f.write(f"{datetime.now().isoformat()} - Modèle v{model_version.version} déployé en production\n")
                
        except Exception as e:
            print(f"Erreur lors du déploiement du modèle: {str(e)}")
        
    deployment_task = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model,
        provide_context=True,
    )
    
    # Définir le flux de tâches
    check_data_task >> drift_check_task >> decision_task
    decision_task >> retrain_lstm_task >> optimize_model_task
    decision_task >> retrain_vgg16_task >> optimize_model_task
    optimize_model_task >> evaluation_task >> deployment_task
