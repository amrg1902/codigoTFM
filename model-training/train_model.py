# Importa las bibliotecas necesarias
# import os
# import mlflow
# import mlflow.sklearn
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# Configura la URI de la base de datos y la dirección del servidor de MLflow
# mlflow.set_tracking_uri("http://mlflow_container:80")
# mlflow.set_experiment('Entrenamiento prueba data iris')

# # Carga los datos de iris
# iris = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# # Entrenamiento del modelo
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Realiza predicciones en el conjunto de prueba
# y_pred = model.predict(X_test)

# # Calcula la precisión del modelo
# accuracy = accuracy_score(y_test, y_pred)

# # Log en MLflow
# with mlflow.start_run():
#     mlflow.log_param("n_estimators", 100)
#     mlflow.log_metric("accuracy", accuracy)
#     mlflow.sklearn.log_model(model, "model")

# Importa las bibliotecas necesarias
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, jsonify
from prometheus_client import start_http_server, Gauge

# Inicia el servidor de métricas de Prometheus en el puerto 8000
start_http_server(8000)

# Define una aplicación Flask
app = Flask(__name__)

# Define una métrica de gauge para Prometheus
accuracy_gauge = Gauge('mlflow_accuracy', 'Accuracy metric from MLflow')

# Ruta para obtener la métrica de accuracy
@app.route('/metrics')
def metrics():
    return jsonify({'accuracy': accuracy_gauge._value.get()})



if __name__ == '__main__':
    # Carga el conjunto de datos Breast Cancer Wisconsin
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    # Entrenamiento del modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Realiza predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Calcula la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # Configura la URI de seguimiento de MLflow
    mlflow.set_tracking_uri("http://mlflow_container:80")
    mlflow.set_experiment('Entrenamiento de prueba mas complejo')

    # Inicia un nuevo "run" de MLflow
    with mlflow.start_run():
        # Log de parámetros y métricas en MLflow
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        # Log del modelo en MLflow
        mlflow.sklearn.log_model(model, "model")

    # Ejecuta la aplicación Flask
    app.run(host='0.0.0.0', port=5000)
