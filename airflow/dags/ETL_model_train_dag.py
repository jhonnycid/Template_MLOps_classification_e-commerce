#Libraries Airflow
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

with DAG(
    dag_id='Rakuten_ETL_training_dag',
    schedule_interval=None,
    tags =['ETL', 'Model_training'],
    start_date= days_ago(0)
) as my_dag:
        
    data_processing = DockerOperator(
        task_id="run_script_Data_Processing",
        image = "data_processing:latest",
        auto_remove=True,
        command='python3 data_processing.py',
        mount_tmp_dir=False,
        #mounts=[Mount(source="C:/Users/jhonn/data", target="/app/data", type="bind")],
        mounts=[Mount(source="C:/Users/jhonn/Projets_DataScientest/Projet MLOPs/Template_MLOps_classification_e-commerce/airflow/data", target="/app/data", type="bind")],
        docker_url="tcp://host.docker.internal:2375",  # for windows
        network_mode="bridge")
    
    model_training = DockerOperator(
        task_id="run_script_training_data",
        image = "training_model:latest",
        auto_remove=True,
        command='python3 train_model_v2.py',
        mount_tmp_dir=False,
        #mounts=[Mount(source="C:/Users/jhonn/data", target="/app/data", type="bind")],
        mounts=[Mount(source="C:/Users/jhonn/Projets_DataScientest/Projet MLOPs/Template_MLOps_classification_e-commerce/airflow/data", target="/app/data", type="bind")],
        docker_url="tcp://host.docker.internal:2375",  # for windows
        network_mode="bridge")

    data_processing >> model_training