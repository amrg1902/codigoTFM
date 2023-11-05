#Utiliza una imagen base con Python y las bibliotecas requeridas
FROM python:3.8

# Instala las dependencias específicas que necesitas para el entrenamiento del modelo
RUN pip install scikit-learn numpy pandas mlflow  # Agrega las bibliotecas necesarias

# Copia tu código de entrenamiento y los datos necesarios al contenedor
COPY ./modelo.py /app/train_model.py
COPY ./data/ /app/data/

# Establece el directorio de trabajo
WORKDIR /app

# Ejecuta el código de entrenamiento
CMD ["python", "train_model.py"]
