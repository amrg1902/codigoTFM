from flask import Flask, render_template, make_response
from prometheus_flask_exporter import PrometheusMetrics
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Configura la URI de seguimiento de MLflow
mlflow.set_tracking_uri("http://mlflow_container:80")

@app.route('/metrics') #Para que prometheus los raspe correctamente
def mostrar_experimentos():

# Nombre del experimento
    nombre_experimento = "Entrenamiento de prueba mas complejo"

    # Obtén el ID del experimento por su nombre
    experimento_id = mlflow.get_experiment_by_name(nombre_experimento).experiment_id
    # Obtén todas las ejecuciones del experimento
    runs = mlflow.search_runs(experiment_ids=experimento_id)
    
    # Inicializa metricas_prometheus
    metricas_prometheus = ""
    # Itera sobre las ejecuciones y muestra las métricas
    for index, run in runs.iterrows():
        run_id = run.run_id
        metrics = mlflow.get_run(run_id).data.metrics
        for metric_name, metric_value in metrics.items():
            metricas_prometheus += f'{metric_name}{{run_id="{run_id}"}} {metric_value}\n'
        print(f"Metrics for run {run_id}: {metrics}")
    # Renderiza la plantilla HTML con la lista de experimentos
    response = make_response(metricas_prometheus)
    response.headers["Content-Type"] = "text/plain"

    return response 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
