# Dockerfile_mlflow_exporter
FROM python:3.8

WORKDIR /app

COPY ./mlflow-exporter/mlflow_exporter.py .

RUN pip install mlflow prometheus_client

CMD ["python", "mlflow_exporter.py"]
