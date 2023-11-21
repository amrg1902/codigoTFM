# Importa las bibliotecas necesarias
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

