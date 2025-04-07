from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.docker_operator import DockerOperator


with DAG(
    dag_id='hello_world',
    tags=['docker', 'datascientest'],
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=1),
    },
    catchup=False
) as dag:
    
    hello_world = DockerOperator(
        task_id="hello_world",
        image="hello-world",
        docker_url="tcp://host.docker.internal:2375",  # Pour Windows, et la comm entre container
        network_mode="bridge",
    ) 
