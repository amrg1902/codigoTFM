# Utiliza una imagen base de Python con las bibliotecas necesarias
FROM python:3.9

# Establece el directorio de trabajo
WORKDIR /app

# Instala MLflow y el exportador de Prometheus desde el requirements.txt
COPY ./model-training/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copia el script de entrenamiento al contenedor
COPY ./model-training/train_model.py /app/train_model.py

# Establece la variable de entorno para la URI de la base de datos de MLflow
ENV MLFLOW_DATABASE_URI postgresql://mlflow:mlflow@mlflow_postgres:5432/mlflow_db

# Establece la variable de entorno para la dirección del servidor de MLflow
ENV MLFLOW_SERVER_URI http://mlflow_container:80

# Establece el comando por defecto al iniciar el contenedor
CMD ["python", "train_model.py"]
