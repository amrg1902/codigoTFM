from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import mlflow
import uvicorn
import numpy as np

app = FastAPI()

# Configura la URI de seguimiento de MLflow
mlflow.set_tracking_uri("http://mlflow_container:80")
nombre_experimento = "Entrenamiento dataset vino"

def fetch_best_model_uri():
    lowest_mse = float('inf')
    experimento_id = mlflow.get_experiment_by_name(nombre_experimento).experiment_id
    runs = mlflow.search_runs(experiment_ids=experimento_id)

    for index, run in runs.iterrows():
        run_id = run.run_id
        run_info = mlflow.get_run(run_id).info
        run_name = run_info.run_name
        metrics = mlflow.get_run(run_id).data.metrics

        for metric_name, metric_value in metrics.items():
            if metric_name == "Accuracy":
                current_accuracy = metric_value
                if (current_accuracy > highest_accuracy) & (current_accuracy <= 1):
                    highest_accuracy = current_accuracy
                    best_model = run_name

        if run_name == best_model:
            best_model_run_id = run_id

    model_uri = f"runs:/{best_model_run_id}/{best_model}"
    return model_uri

# Montar la carpeta 'static' para servir archivos estáticos (como el HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def read_form():
    return open("static/index.html", "r").read()

@app.get("/predict/")
def model_output(
    age: float, bmi: float, bp: float, s1: float, s2: float, s3: float, 
    s4: float, s5: float, s6: float):
    logged_model = fetch_best_model_uri()
    if logged_model:
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        input_data = pd.DataFrame({
            "age": [age], "bmi": [bmi], "bp": [bp], "s1": [s1], "s2": [s2],
            "s3": [s3], "s4": [s4], "s5": [s5], "s6": [s6]
        })

        # Aplicar la transformación y estandarización a los datos de entrada
        transformed_data = apply_transformation(input_data)
        # Asegúrate de que transformed_data es un array bidimensional
        transformed_data_2d = np.reshape(transformed_data, (1, -1))
        # Convertir el array a un DataFrame de pandas
        df_transformed = pd.DataFrame(transformed_data_2d, columns=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9'])

        # Crea el DataFrame
        predictions = loaded_model.predict(pd.DataFrame(transformed_data_2d))

        return PlainTextResponse(str(predictions[0]), media_type="text/plain")

    else:
        raise HTTPException(status_code=500, detail="No model available.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7654)
