from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "Amir",
    "start_date": days_ago(1),  # запуск день назад
    "retries": 5,  # try again if error
    #"retry_delay": datetime.timedelta(minutes=5),  # дельта запуска при повторе 5 минут
    "task_concurency": 1  # 1 task at a time
}

piplines = {'train': {"schedule": "0 0 * * 0"},  # Retrain model every week
            "predict": {"schedule": "*/15 * * * *"}}  # Every 15 minutes predict liquidity


def init_dag(dag, task_id):
    with dag:
        t1 = BashOperator(
            task_id=f"{task_id}",
            bash_command=f'python3 /Users/amir1/PycharmProjects/uniswap-automation/{task_id}.py')
    return dag


for task_id, params in piplines.items():
    dag = DAG(task_id,
              schedule_interval=params['schedule'],
              max_active_runs=1,
              default_args=default_args
              )
    init_dag(dag, task_id)
    globals()[task_id] = dag
