# Utiliza una imagen base de Python con MLflow preinstalado
FROM python:3.9

#Actualizar pip
RUN pip install --upgrade pip

# Instala MLflow y el exportador de Prometheus desde el requirements.txt
COPY ./mlflow-container/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Instala wait-for-it
ADD https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh /usr/local/bin/wait-for-it
RUN chmod +x /usr/local/bin/wait-for-it

# Actualiza
RUN apt-get update && \
    apt-get install -y wait-for-it

# Creo la conexión entre la base de datos y MLflow
ENV MLFLOW_DATABASE_URI postgresql://mlflow:mlflow@mlflow_postgres:5432/mlflow_db
# Establece el valor de MLFLOW_EXPOSE_PROMETHEUS
ENV MLFLOW_EXPOSE_PROMETHEUS true

# Establece el puerto en el que se ejecutará el servidor de MLflow
EXPOSE 80

# Establecer el directorio de trabajo
WORKDIR /app

# Establece el comando por defecto al iniciar el contenedor
CMD ["sh", "-c", "wait-for-it mlflow_postgres:5432 -- mlflow ui --host 0.0.0.0 --port 80 --backend-store-uri postgresql://mlflow:mlflow@mlflow_postgres:5432/mlflow_db --expose-prometheus $MLFLOW_EXPOSE_PROMETHEUS"]
