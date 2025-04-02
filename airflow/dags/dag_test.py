#Libraries Airflow
from airflow import DAG
#from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import pandas as pd

#def prepare_data(path_to_data='/app/data/preprocessed'):
#    # reading data
#    data = pd.read_csv(f"{path_to_data}/X_train_update.csv")
#    print(data.head())

    
with DAG(
    dag_id='Rakuten_tests',
    schedule_interval=None,
    tags =['Rakuten', 'Test'],
    start_date= days_ago(0)
) as my_dag:
        
    task_1 = BashOperator(
        task_id="run_script_df",
        bash_command="python /app/src/script_test.py")
