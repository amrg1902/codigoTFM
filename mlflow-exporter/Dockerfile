# Dockerfile_mlflow_exporter
FROM python:3.8

WORKDIR /app

COPY ./mlflow-exporter/mlflow_exporter.py .

RUN pip install Flask prometheus-flask-exporter mlflow scikit-learn

CMD ["python", "mlflow_exporter.py"]
