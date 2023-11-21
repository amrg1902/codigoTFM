import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#Configura la URI de la base de datos y la dirección del servidor de MLflow
mlflow.set_tracking_uri("http://mlflow_container:80")
mlflow.set_experiment('Entrenamiento prueba data iris')

# Carga los datos de iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
############################################# Random Forest ###################################
# Entrenamiento del modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Realiza predicciones en el conjunto de prueba
y_pred = model.predict(X_test)
# Calcula la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
# Definir el nombre del run
run_name = "Random Forest Classifier"
# Log en MLflow
with mlflow.start_run(run_name=run_name):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")


################################################ SVM #############################################
# Entrenamiento del modelo SVM
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# Realiza predicciones en el conjunto de prueba
y_pred_svm = svm_model.predict(X_test)

# Calcula la precisión del modelo SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)


# Definir el nombre del run
run_name = "SVM"

# Log en MLflow
with mlflow.start_run(run_name=run_name):
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy_svm)
    mlflow.sklearn.log_model(model, "svm_model")