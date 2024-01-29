import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

uri = os.getenv('URI')
print(uri)
nombre_experimento = os.getenv('nombre_experimento')
print(nombre_experimento)

#Configura la URI de la base de datos y la dirección del servidor de MLflow
mlflow.set_tracking_uri(uri)
mlflow.set_experiment(nombre_experimento)

# Cargar el dataset de vino
wine = load_wine()
X, y = wine.data, wine.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Función para evaluar y registrar métricas
def evaluate_model(model, model_name):
    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Definir el nombre del run
    run_name = f"{model_name}"
    
    # Log en MLflow
    with mlflow.start_run(run_name=run_name):
        # Log de métricas
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1", f1)
        mlflow.sklearn.log_model(model, model_name)

# Modelos
models = {
    "RandomForestClassifier": RandomForestClassifier(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "SVC": SVC(),
    "GradientBoostingClassifier": GradientBoostingClassifier()
}

# Evaluación de modelos y registro en MLflow
for model_name, model in models.items():
    evaluate_model(model, model_name)