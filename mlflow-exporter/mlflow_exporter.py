from flask import Flask
from prometheus_flask_exporter import PrometheusMetrics
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Configura la URI de seguimiento de MLflow
mlflow.set_tracking_uri("http://mlflow_container:80")

#@metrics.summary('mlflow_accuracy', 'MLflow accuracy')
#@app.route('/train_model')
# def train_model():
#     with mlflow.start_run():
#         mlflow.log_param("n_estimators", 100)
#         mlflow.log_metric("accuracy", accuracy)
#         mlflow.sklearn.log_model(model, "model")

#     return f"Model trained with accuracy: {accuracy}"

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000)

@app.route('/')
def mostrar_experimentos():
    # Obtiene la lista de experimentos
    experimentos = mlflow.search_runs().experiment_name.unique()

    # Renderiza la plantilla HTML con la lista de experimentos
    return render_template('experimentos.html', experimentos=experimentos)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
