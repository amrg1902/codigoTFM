# Importa las bibliotecas necesarias
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

#Configura la URI de la base de datos y la dirección del servidor de MLflow
mlflow.set_tracking_uri("http://mlflow_container:80")
mlflow.set_experiment('Entrenamiento Breast Cancer Wisconsin')

# Carga los datos de Breast Cancer Wisconsin
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
############################################# Random Forest ###################################
# Entrenamiento del modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Realiza predicciones en el conjunto de prueba
y_pred = model.predict(X_test)
# Calcula la accuracy del modelo
accuracy = accuracy_score(y_test, y_pred)
# Obtener la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
# Calcular precision y recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
# Definir el nombre del run
run_name = "Random Forest Classifier"
# Log en MLflow
with mlflow.start_run(run_name=run_name):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metric("true_positives", conf_matrix[1, 1])
    mlflow.log_metric("true_negatives", conf_matrix[0, 0])
    mlflow.log_metric("false_positives", conf_matrix[0, 1])
    mlflow.log_metric("false_negatives", conf_matrix[1, 0])
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)


################################################ SVM #############################################
# Entrenamiento del modelo SVM
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
# Realiza predicciones en el conjunto de prueba
y_pred_svm = svm_model.predict(X_test)
# Calcula la precisión del modelo SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
# Obtener la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
# Calcular precision y recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
# Definir el nombre del run
run_name2 = "SVM"
# Log en MLflow
with mlflow.start_run(run_name=run_name2):
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy_svm)
    mlflow.sklearn.log_model(model, "svm_model")
    mlflow.log_metric("true_positives", conf_matrix[1, 1])
    mlflow.log_metric("true_negatives", conf_matrix[0, 0])
    mlflow.log_metric("false_positives", conf_matrix[0, 1])
    mlflow.log_metric("false_negatives", conf_matrix[1, 0])
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)