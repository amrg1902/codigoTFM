# Utiliza una imagen base de Python con las bibliotecas necesarias
FROM python:3.9

# Instala las dependencias
RUN pip install mlflow scikit-learn

# Copia el script de entrenamiento al contenedor
COPY ./model-training/train_model.py /app/train_model.py

# Establece el directorio de trabajo
WORKDIR /app

# Establece el comando por defecto al iniciar el contenedor
CMD ["python", "train_model.py"]
