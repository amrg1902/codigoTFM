from flask import Flask, render_template, make_response
import pandas as pd
import mlflow

app = Flask(__name__)
# Configura la URI de seguimiento de MLflow
mlflow.set_tracking_uri("http://mlflow_container:80")

def fetch_latest_model():
    client = MlflowClient()
    # Obtener todos los experimentos
    experiments = client.list_experiments()
    # Inicializar variables para el experimento y la métrica más baja
    best_experiment = None
    lowest_mse = float('inf')
    for experiment in experiments:
        # Obtener todas las corridas dentro del experimento
        runs = client.search_runs(experiment.experiment_id)
        for run in runs:
            # Obtener las métricas asociadas a la corrida
            metrics = client.get_run(run.info.run_id).data.metrics
            # Verificar si la métrica "MSE" está presente
            if 'MSE' in metrics:
                current_mse = metrics['MSE']
                # Actualizar el mejor experimento y métrica si encontramos una menor
                if current_mse < lowest_mse:
                    lowest_mse = current_mse
                    best_experiment = experiment.name
    return best_experiment

def fetch_latest_version(model_name):
    client = MlflowClient()
    # Obtener la última versión registrada del modelo en el experimento
    latest_version = client.get_latest_versions(model_name, stages=['Production'])[0].version
    # Cargar el modelo
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{latest_version}"
    )
    return model

@app.route('/predict')
def model_output(age: float, bmi: float, bp: float, s1: float, s2: float, s3: float, s4: float, s5: float, s6: float):
    nombre_experimento = "Entrenamiento dataset Diabetes"
    print("Works I")
    model_name = fetch_latest_model()
    model = fetch_latest_version(model_name)
    print("Works II")
    input = pd.DataFrame({"age": [age], "bmi": [bmi], "bp": [bp], "s1": [s1], "s2": [s2], "s3": [s3], "s4": [s4], "s5": [s5], "s6": [s6]})
    prediction = model.predict(input)
    print(prediction)
    return {"prediction": prediction[0]}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7654)



