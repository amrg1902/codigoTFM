from flask import Flask
from prometheus_flask_exporter import PrometheusMetrics
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Configura la URI de seguimiento de MLflow
mlflow.set_tracking_uri("http://mlflow_container:80")
mlflow.set_experiment('Entrenamiento de prueba mas complejo')

@metrics.summary('mlflow_accuracy', 'MLflow accuracy')
@app.route('/train_model')
def train_model():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

    return f"Model trained with accuracy: {accuracy}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
