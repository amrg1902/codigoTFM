# # Utiliza una imagen base de Python con MLflow preinstalado
# FROM python:3.9

# # Instala MLflow
# RUN pip install mlflow

# #Actualiza
# RUN apt-get update && \
#     apt-get install -y wait-for-it

# #Creo la conexion entre la bd y mlflow
# #ENV MLFLOW_DATABASE_URI postgresql://mlflow:mlflow@localhost:5432/mlflow_db
# ENV MLFLOW_DATABASE_URI postgresql://mlflow:mlflow@mlflow_postgres:5432/mlflow_db


# # Establece el puerto en el que se ejecutará el servidor de MLflow
# EXPOSE 80

# # Establece el comando por defecto al iniciar el contenedor
# #CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "80"]

# CMD ["wait-for-it", "mlflow_postgres:5432", "--", "mlflow", "ui", "--host", "0.0.0.0", "--port", "80"]

# Utiliza una imagen base de Python con MLflow preinstalado
FROM python:3.9

#Actualizar pip
RUN pip install --upgrade pip

# Instala MLflow y el exportador de Prometheus
RUN pip install mlflow mlflow-prometheus-exporter

# Actualiza
RUN apt-get update && \
    apt-get install -y wait-for-it

# Creo la conexión entre la base de datos y MLflow
ENV MLFLOW_DATABASE_URI postgresql://mlflow:mlflow@mlflow_postgres:5432/mlflow_db

# Establece el puerto en el que se ejecutará el servidor de MLflow
EXPOSE 80

# Establece el comando por defecto al iniciar el contenedor
CMD ["wait-for-it", "mlflow_postgres:5432", "--", "mlflow", "ui", "--host", "0.0.0.0", "--port", "80", "--backend-store-uri", "postgresql://mlflow:mlflow@mlflow_postgres:5432/mlflow_db", "--expose-prometheus", "--prometheus-port", "80"]
