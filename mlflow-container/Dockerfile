# # Utiliza una imagen base de Python con MLflow preinstalado
# FROM python:3.9

# # Instala MLflow y otras dependencias
# RUN ["pip", "install", "--upgrade", "pip"]
# RUN pip install --no-cache-dir mlflow

# # Establece un directorio de trabajo en el contenedor
# WORKDIR /app

# # Expone el puerto en el que se ejecutará el servidor de MLflow
# EXPOSE 5005

# # Configura MLflow para utilizar una base de datos SQLite
# ENV MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT /mlflow/artifacts
# ENV MLFLOW_SERVER_DEFAULT_DATABASE_URI sqlite:////mlflow/mlflow.db

# # # Copia el modelo registrado en la imagen del contenedor  CREO QUE TENGO QUE COPIAR ESTO DESDE EL CONTENEDOR DEL TRAINING AL CONTENEDOR DEL MLFLOW
# # COPY ./mlruns/ /app/mlruns

# # Define el comando para iniciar el servidor de MLflow
# CMD mlflow server --host 0.0.0.0 --port 5005 --backend-store-uri sqlite:////mlflow/mlflow.db --default-artifact-root /mlflow/artifacts



FROM python:3.9
WORKDIR /home
RUN ["pip", "install", "--upgrade", "pip"]
# boto3 is required to access artifacts from s3 
RUN ["pip", "install", "psycopg2-binary", "mlflow==1.25.1", "boto3"]
COPY load_env.sh load_env.sh
RUN ["chmod", "+x", "load_env.sh"]
EXPOSE 8000
ENTRYPOINT ["sh", "-c", "bash load_env.sh && mlflow server --backend-store-uri $DB_URI --default-artifact-root $MLRUNS_DIRECTORY -h 0.0.0.0 -p 5001"]
