# Utiliza una imagen base de Python con MLflow preinstalado
FROM python:3.8

# Instala MLflow y otras dependencias
RUN pip install --no-cache-dir mlflow

# Establece un directorio de trabajo en el contenedor
WORKDIR /app

# Expone el puerto en el que se ejecutará el servidor de MLflow
EXPOSE 5005

# Configura MLflow para utilizar una base de datos SQLite
ENV MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT /mlflow/artifacts
ENV MLFLOW_SERVER_DEFAULT_DATABASE_URI sqlite:////mlflow/mlflow.db

# Crea un directorio para almacenar los artefactos de MLflow y la base de datos
RUN mkdir -p /mlflow/artifacts
RUN mkdir -p /mlflow

# Copia el modelo registrado en la imagen del contenedor
COPY ./mlflow-container/mlruns/ /app/mlruns

# Define el comando para iniciar el servidor de MLflow
CMD mlflow server --host 0.0.0.0 --port 5005 --backend-store-uri sqlite:////mlflow/mlflow.db --default-artifact-root /mlflow/artifacts
# CMD ["mlflow", "models", "serve", "-m", "runs:/<RUN_ID>/modelo", "--host", "0.0.0.0", "--port", "5005"]
