## How to run strategies
#### For effective and seamless execution strategies can be launched with Airflow and MLflow

### VENV

1) Create folder and copy repo `mkdir uniswap-strategies` 
2) Create venv ( or conda ) `python -m venv myvenv`
3) Activate venv `source myvenv/bin/activate`
4) Install sqlite for mlflow `pip install pysqlite3`

### Mlflow intallation and launch

Installation
`pip install mlflow`

Folder for mlflow files`mkdir mlflow` 

Set path `export MLFLOW_REGISTRY_URI=mlflow`

Might be helpfull if have problems: https://www.mlflow.org/docs/latest/tracking.html#tracking-ui

Server launch `mlflow server --host localhost --port 5000 --backend-store-uri sqlite:///${MLFLOW_REGISTRY_URI}/mlflow.db --default-artifact-root ${MLFLOW_REGISTRY_URI}`

If you want stop server 

`ps -A | grep gunicorn`

`kill -9 ps aux | grep mlflow | awk '{print $2}'`

## Airflow intallation and launch

Create folder for airflow files `mkdir airflow`

Installation `airflow pip install apache-airflow==2.0.1
   --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.0.1/constraints-3.7.txt"
   export AIRFLOW_HOME=.`

DB initialisation `airflow db init` 
Also put in airflow.cfg прописать:
   `[webserver]
   rbac = True
   load_examples = False`

Create Airflow user `airflow users create --username Amir --firstname Amir --lastname Amir --role Admin
   --email ***@***.com`

Launch Airflow

 `airflow webserver -p 8080 `

`airflow scheduler`

If you want to stop server

`ps -A | grep gunicorn`

`kill -9 ps aux | grep airflow | awk '{print $2}'`
