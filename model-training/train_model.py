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
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

#Configura la URI de la base de datos y la dirección del servidor de MLflow
mlflow.set_tracking_uri("http://mlflow_container:80")
mlflow.set_experiment("Entrenamiento dataset Diabetes")

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

############################################# Linear regression ###################################
# Entrenamiento del modelo de regresión lineal
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Predicciones
y_pred_lr = model_lr.predict(X_test)
y_pred_lr = pd.Series(y_pred_lr, name='target')

# Métricas
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)
r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)

# Definir el nombre del run
run_name_lr = "LinearRegression"
# Log en MLflow
with mlflow.start_run(run_name=run_name_lr):
    # Log de métricas
    mlflow.log_metric("MSE", mse_lr)
    mlflow.log_metric("RMSE", rmse_lr)
    mlflow.log_metric("R2", r2_lr)
    mlflow.log_metric("MAE", mae_lr)

    mlflow.sklearn.log_model(model_lr, "model_lr")

################################################ SVR #############################################
# Entrenamiento del modelo SVR
model_svr = SVR()
model_svr.fit(X_train, y_train)

# Predicciones
y_pred_svr = model_svr.predict(X_test)
y_pred_svr = pd.Series(y_pred_svr, name='target')

# Métricas
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = mean_squared_error(y_test, y_pred_svr, squared=False)
r2_svr = r2_score(y_test, y_pred_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)

# Definir el nombre del run
run_name_svr = "SVR"
# Log en MLflow
with mlflow.start_run(run_name=run_name_svr):
    # Log de métricas
    mlflow.log_metric("MSE", mse_svr)
    mlflow.log_metric("RMSE", rmse_svr)
    mlflow.log_metric("R2", r2_svr)
    mlflow.log_metric("MAE", mae_svr)

    mlflow.sklearn.log_model(model_svr, "model_svr")

################################################ Random Forest Regressor #############################################
# Entrenamiento del modelo Random Forest Regressor
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Predicciones
y_pred_rf = model_rf.predict(X_test)
y_pred_rf = pd.Series(y_pred_rf, name='target')

# Métricas
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# Definir el nombre del run
run_name_rf = "RandomForestRegressor"
# Log en MLflow
with mlflow.start_run(run_name=run_name_rf):
    # Log de métricas
    mlflow.log_metric("MSE", mse_rf)
    mlflow.log_metric("RMSE", rmse_rf)
    mlflow.log_metric("R2", r2_rf)
    mlflow.log_metric("MAE", mae_rf)

    mlflow.sklearn.log_model(model_rf, "model_rf")

################################################ Gradient boosting regressor #############################################
# Entrenamiento del modelo Gradient Boosting Regressor
model_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_gb.fit(X_train, y_train)

# Predicciones
y_pred_gb = model_gb.predict(X_test)
y_pred_gb = pd.Series(y_pred_gb, name='target')

# Métricas
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = mean_squared_error(y_test, y_pred_gb, squared=False)
r2_gb = r2_score(y_test, y_pred_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)

# Definir el nombre del run
run_name_gb = "GradientBoostingRegressor"
# Log en MLflow
with mlflow.start_run(run_name=run_name_gb):
    # Log de métricas
    mlflow.log_metric("MSE", mse_gb)
    mlflow.log_metric("RMSE", rmse_gb)
    mlflow.log_metric("R2", r2_gb)
    mlflow.log_metric("MAE", mae_gb)

    mlflow.sklearn.log_model(model_gb, "model_gb")
################################################ Red Neuronal #############################################
# Escalado de características para la red neuronal
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenamiento de la red neuronal
model_mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
model_mlp.fit(X_train_scaled, y_train)

# Predicciones
y_pred_mlp = model_mlp.predict(X_test_scaled)
y_pred_mlp = pd.Series(y_pred_mlp, name='target')

# Métricas
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
rmse_mlp = mean_squared_error(y_test, y_pred_mlp, squared=False)
r2_mlp = r2_score(y_test, y_pred_mlp)
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)

# Definir el nombre del run
run_name_mlp = "MLPRegressor"
# Log en MLflow
with mlflow.start_run(run_name=run_name_mlp):
    # Log de métricas
    mlflow.log_metric("MSE", mse_mlp)
    mlflow.log_metric("RMSE", rmse_mlp)
    mlflow.log_metric("R2", r2_mlp)
    mlflow.log_metric("MAE", mae_mlp)

    mlflow.sklearn.log_model(model_mlp, "model_mlp")