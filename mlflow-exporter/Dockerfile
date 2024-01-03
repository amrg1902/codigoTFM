# Dockerfile_mlflow_exporter
FROM python:3.8

WORKDIR /app

COPY ./mlflow-exporter/ .

# Instala MLflow y el exportador de Prometheus desde el requirements.txt
COPY ./mlflow-exporter/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt


CMD ["python", "mlflow_exporter.py"]