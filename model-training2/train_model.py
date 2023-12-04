import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Cargar el conjunto de datos
data_diabetes = load_diabetes()
X_diabetes = pd.DataFrame(data_diabetes.data, columns=[f'feature_{i}' for i in range(1, 11)])
y_diabetes = pd.Series(data_diabetes.target, name='target')

# Renombrar las columnas
column_name_mapping = {
    'feature_1': 'age',
    'feature_2': 'sex',
    'feature_3': 'bmi',
    'feature_4': 'bp',  
    'feature_5': 's1',  
    'feature_6': 's2',  
    'feature_7': 's3',  
    'feature_8': 's4',  
    'feature_9': 's5',  
    'feature_10': 's6'  
}

X_diabetes.rename(columns=column_name_mapping, inplace=True)

# Seleccionar columnas específicas
selected_columns = ['age', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6' ]

# Crear un nuevo DataFrame con las columnas seleccionadas
selected_data = X_diabetes[selected_columns]

# División de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(selected_data, y_diabetes, test_size=0.2, random_state=42)

############################################# Random Forest ###################################
# Entrenamiento del modelo
RFCmodel = RandomForestClassifier(n_estimators=100, random_state=42)
RFCmodel.fit(X_train, y_train)
# Realiza predicciones en el conjunto de prueba
y_pred = RFCmodel.predict(X_test)
# Redondear las predicciones
y_pred = pd.Series(y_pred, name='target')

###METRICAS
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Definir el nombre del run
run_name = "Random Forest Classifier"
# Log en MLflow
with mlflow.start_run(run_name=run_name):
    mlflow.log_param("n_estimators", 100)
    # Log de métricas
    mlflow.log_metric("RFC_mse", mse)
    mlflow.log_metric("RFC_rmse", rmse)
    mlflow.log_metric("RFC_r2", r2)
    mlflow.log_metric("RFC_mae", mae)
    mlflow.sklearn.log_model(RFCmodel, "RandomForestClassifier_model")

################################################ SVM #############################################
# Entrenamiento del modelo SVM
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
# Realiza predicciones en el conjunto de prueba
y_pred_svm = svm_model.predict(X_test)
# Redondear las predicciones
y_pred_svm = pd.Series(y_pred_svm, name='target')

# Calcula métricas
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

# Definir el nombre del run
run_name2 = "SVM"
# Log en MLflow
with mlflow.start_run(run_name=run_name2):
    mlflow.log_param("random_state", 42)
    # Log de métricas
    mlflow.log_metric("SVM_accuracy", accuracy_svm)
    mlflow.log_metric("SVM_precision", precision_svm)
    mlflow.log_metric("SVM_recall", recall_svm)
    mlflow.log_metric("SVM_f1_score", f1_svm)

    mlflow.sklearn.log_model(svm_model, "svm_model")