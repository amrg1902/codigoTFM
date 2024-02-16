from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import mlflow
import uvicorn
import numpy as np
import os

app = FastAPI()
uri = os.getenv('URI')
print(uri)
nombre_experimento = os.getenv('nombre_experimento')
print(nombre_experimento)

#Configura la URI de la base de datos y la dirección del servidor de MLflow
mlflow.set_tracking_uri(uri)

def fetch_best_model_uri():
    lowest_mse = float('inf')
    experimento_id = mlflow.get_experiment_by_name(nombre_experimento).experiment_id
    runs = mlflow.search_runs(experiment_ids=experimento_id)
    highest_accuracy = 0
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
    feature_1: float = Query(..., description="Feature 1"),
    feature_2: float = Query(..., description="Feature 2"),
    feature_3: float = Query(..., description="Feature 3"),
    feature_4: float = Query(..., description="Feature 4"),
    feature_5: float = Query(..., description="Feature 5"),
    feature_6: float = Query(..., description="Feature 6"),
    feature_7: float = Query(..., description="Feature 7"),
    feature_8: float = Query(..., description="Feature 8"),
    feature_9: float = Query(..., description="Feature 9"),
    feature_10: float = Query(..., description="Feature 10"),
    feature_11: float = Query(..., description="Feature 11"),
    feature_12: float = Query(..., description="Feature 12"),
    feature_13: float = Query(..., description="Feature 13"),
):
    logged_model = fetch_best_model_uri()
    if logged_model:
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        input_data = np.array([
            [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
        ])

        # Crea el DataFrame
        predictions = loaded_model.predict(input_data)

        return PlainTextResponse(str(predictions[0]), media_type="text/plain")

    else:
        raise HTTPException(status_code=500, detail="No model available.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7654)
