from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
import pandas as pd
import mlflow
import uvicorn

app = FastAPI()

# Configura la URI de seguimiento de MLflow
mlflow.set_tracking_uri("http://mlflow_container:80")
nombre_experimento = "Entrenamiento dataset Diabetes"

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

# @app.on_event("startup")
# async def startup():
Instrumentator().instrument(app).expose(app)


@app.get("/", response_class=HTMLResponse)
def read_form():
    return open("static/index.html", "r").read()

@app.get("/predict/")
def model_output(
    age: float, bmi: float, bp: float, s1: float, s2: float, s3: float, 
    s4: float, s5: float, s6: float
):
    def map_age(value):
        return value * 55 + 55

    def map_bmi(value):
        return value * 20 + 20

    def map_bp(value):
        return value * 37.5 + 97.5

    def map_s1(value):
        return value * 50 + 200

    def map_s2(value):
        return value * 80 + 140

    def map_s3(value):
        return value * 30 + 60

    def map_s4(value):
        return value * 3 + 3

    def map_s5(value):
        return value * 35 + 185

    def map_s6(value):
        return value * 50 + 110

    logged_model = fetch_best_model_uri()
    if logged_model:
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        input_data = pd.DataFrame({
            "age": [age], "bmi": [bmi], "bp": [bp], "s1": [s1], "s2": [s2],
            "s3": [s3], "s4": [s4], "s5": [s5], "s6": [s6]
        })

        input_data['age'] = input_data['age'].apply(map_age)
        input_data['bmi'] = input_data['bmi'].apply(map_bmi)
        input_data['bp'] = input_data['bp'].apply(map_bp)
        input_data['s1'] = input_data['s1'].apply(map_s1)
        input_data['s2'] = input_data['s2'].apply(map_s2)
        input_data['s3'] = input_data['s3'].apply(map_s3)
        input_data['s4'] = input_data['s4'].apply(map_s4)
        input_data['s5'] = input_data['s5'].apply(map_s5)
        input_data['s6'] = input_data['s6'].apply(map_s6)

        prediction = loaded_model.predict(pd.DataFrame(input_data))

        return PlainTextResponse(str(prediction), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7654)
