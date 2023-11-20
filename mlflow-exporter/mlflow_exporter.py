# mlflow_exporter.py
import mlflow
from prometheus_client import start_http_server
import mlflow.prometheus

# Configura la URI de seguimiento de MLflow
mlflow.set_tracking_uri("http://mlflow_container:80")

# Inicia el servidor Prometheus HTTP
start_http_server(port=8000)

# Configura la exportación de métricas de MLflow a Prometheus
mlflow.prometheus.export_metrics(job_name='mlflow')

# Permite que el servidor siga ejecutándose
input("Press enter to stop the server...")
