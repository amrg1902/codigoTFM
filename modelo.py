import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Inicia una corrida de MLflow
with mlflow.start_run():
    # Carga y preprocesa el dataset
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    target = iris.target

    # Divide el dataset en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    # Entrena tu modelo
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Realiza predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Calcula la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)

    # Registra el modelo en MLflow
    mlflow.sklearn.log_model(sk_model=model, artifact_path="modelo")

    # Registra la métrica de precisión en MLflow
    mlflow.log_metric("accuracy", accuracy)
