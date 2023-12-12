from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
import pandas as pd
import mlflow
import uvicorn
import threading, asyncio
import numpy as np


app = FastAPI()


# Configura la URI de seguimiento de MLflow
mlflow.set_tracking_uri("http://mlflow_container:80")
nombre_experimento = "Entrenamiento dataset Diabetes"


# Rangos deseados
desired_ranges = {
    'age': (-0.107226, 0.110727),
    'bmi': (-0.090275, 0.170555),
    'bp': (-0.112399, 0.132044),
    's1': (-0.126781, 0.153914),
    's2': (-0.115613, 0.198788),
    's3': (-0.102307, 0.181179),
    's4': (-0.076395, 0.185234),
    's5': (-0.126097, 0.133597),
    's6': (-0.137767, 0.135612)
}

def apply_transformation(data):
    transform_params = {}
    
    for feature in data.keys():
        original_min = data[feature]
        original_max = data[feature]
        desired_min, desired_max = desired_ranges[feature]

        if original_min == original_max:
            a = 0
            b = desired_min
        else:
            a = (desired_max - desired_min) / (original_max - original_min)
            b = desired_min - a * original_min

        transform_params[feature] = {'a': a, 'b': b}

    # Aplicar la transformación lineal a los datos originales
    mapped_data = {feature: transform_params[feature]['a'] * value + transform_params[feature]['b']
                   for feature, value in data.items()}

    # Convertir el diccionario a un array
    data_array = np.array([list(mapped_data.values())])

    return data_array


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
            if metric_name == "MSE":
                current_mse = metric_value
                if current_mse < lowest_mse:
                    lowest_mse = current_mse
                    best_model = run_name

        if run_name == best_model:
            best_model_run_id = run_id

    model_uri = f"runs:/{best_model_run_id}/{best_model}"
    return model_uri

# Montar la carpeta 'static' para servir archivos estáticos (como el HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Función asincrónica para configurar el Instrumentator
async def configure_instrumentator():
    await asyncio.sleep(1)  # Puedes esperar si es necesario
    Instrumentator().instrument(app).expose(app)

# Función para configurar el Instrumentator en un hilo separado
def configure_instrumentator_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(configure_instrumentator())

# Configurar el Instrumentator en un hilo separado
thread = threading.Thread(target=configure_instrumentator_thread)
thread.start()


@app.get("/", response_class=HTMLResponse)
def read_form():
    return open("static/index.html", "r").read()

@app.get("/predict/")
def model_output(
    age: float, bmi: float, bp: float, s1: float, s2: float, s3: float, 
    s4: float, s5: float, s6: float
):
    logged_model = fetch_best_model_uri()
    if logged_model:
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        input_data = pd.DataFrame({
            "age": [age], "bmi": [bmi], "bp": [bp], "s1": [s1], "s2": [s2],
            "s3": [s3], "s4": [s4], "s5": [s5], "s6": [s6]
        })

        # Aplicar la transformación y estandarización a los datos de entrada
        transformed_data = apply_transformation(input_data)

        # Realizar predicciones en los datos estandarizados
        predictions = loaded_model.predict(pd.DataFrame(transformed_data))

        return PlainTextResponse(str(predictions[0]), media_type="text/plain")

    else:
        raise HTTPException(status_code=500, detail="No model available.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7654)
