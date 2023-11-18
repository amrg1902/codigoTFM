# Utiliza una imagen base de Python con MLflow preinstalado
FROM python:3.9

# Instala MLflow
RUN pip install mlflow

# Establece el puerto en el que se ejecutará el servidor de MLflow
EXPOSE 5000

# Establece el comando por defecto al iniciar el contenedor
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5001"]
