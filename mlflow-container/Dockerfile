# Utiliza una imagen base de Python con MLflow preinstalado
FROM python:3.9

# Instala MLflow y otras dependencias
RUN ["pip", "install", "--upgrade", "pip"]
RUN pip install --no-cache-dir mlflow

# Establece un directorio de trabajo en el contenedor
WORKDIR /app

# Expone el puerto en el que se ejecutará el servidor de MLflow
EXPOSE 5005

# Configura MLflow para utilizar una base de datos SQLite
ENV MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT /mlflow/artifacts
ENV MLFLOW_SERVER_DEFAULT_DATABASE_URI sqlite:////mlflow/mlflow.db

# # Copia el modelo registrado en la imagen del contenedor  CREO QUE TENGO QUE COPIAR ESTO DESDE EL CONTENEDOR DEL TRAINING AL CONTENEDOR DEL MLFLOW
# COPY ./mlruns/ /app/mlruns

# Copia el script de inicio al contenedor
COPY ./mlflow-container/start_mlflow_server.sh /app/start_mlflow_server.sh

# Concede permisos de ejecución al script
RUN chmod +x /app/start_mlflow_server.sh

# Comando que se ejecutará al iniciar el contenedor
CMD ["/app/start_mlflow_server.sh"]


