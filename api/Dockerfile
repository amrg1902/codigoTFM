# Dockerfile_mlflow_exporter
FROM python:3.8

WORKDIR /app

COPY ./api/ .

# Instala MLflow y el exportador de Prometheus desde el requirements.txt
COPY ./api/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt


ENV MLFLOW_TRACKING_URI=http://mlflow:80

EXPOSE 8000
CMD ["sh", "-c", "uvicorn api:app --port 7654 --host 0.0.0.0"]