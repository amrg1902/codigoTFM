from flask import Flask,  render_template, request
import pandas as pd
import mlflow

app = Flask(__name__)
# Configura la URI de seguimiento de MLflow
mlflow.set_tracking_uri("http://mlflow_container:80")
nombre_experimento = "Entrenamiento dataset Diabetes"

def fetch_best_model():
    #Inicializacion metrica mse
    lowest_mse = float('inf')
    # Obtén el ID del experimento por su nombre
    experimento_id = mlflow.get_experiment_by_name(nombre_experimento).experiment_id
    # Obtén todas las ejecuciones del experimento
    runs = mlflow.search_runs(experiment_ids=experimento_id)

    for index, run in runs.iterrows():
        run_id = run.run_id
        run_info = mlflow.get_run(run_id).info
        run_name = run_info.run_name

        metrics = mlflow.get_run(run_id).data.metrics
        for metric_name, metric_value in metrics.items():
            if(metric_name == "MSE"):
                current_mse = metric_value
                if (current_mse < lowest_mse):
                    lowest_mse = current_mse
                    best_model = run_name
    return best_model

def fetch_best_model_uri():
    best_model_name = fetch_best_model()
    
    if best_model_name:
        experimento_id = mlflow.get_experiment_by_name(nombre_experimento).experiment_id
        best_model_run = mlflow.search_runs(experiment_ids=experimento_id, filter_string=f"run_name = '{best_model_name}'").iloc[0]
        
        # Obtiene la URI del modelo
        model_uri = f"runs:/{best_model_run.run_id}/model"
        return model_uri
    else:
        return None


@app.route('/')
def index():
     # Nombre del experimento
    nombre_experimento = "Entrenamiento dataset Diabetes"

    # Obtén el ID del experimento por su nombre
    experimento_id = mlflow.get_experiment_by_name(nombre_experimento).experiment_id
    # Obtén todas las ejecuciones del experimento
    runs = mlflow.search_runs(experiment_ids=experimento_id)
    
    # Inicializa metricas_prometheus
    metricas_prometheus = ""

    # Itera sobre las ejecuciones y muestra las métricas
    for index, run in runs.iterrows():
        run_id = run.run_id
        run_info = mlflow.get_run(run_id).info
        metrics = mlflow.get_run(run_id).data.metrics
        run_name = run_info.run_name
    #POR TANTO LO QUE TENGO QUE SACAR ES EL RUN_ID CUYO RUN_NAME SEA IGUAL AL ANTERIOR, Y SOBRE ESE, REALIZAR LAS PREDICCIONES

        for metric_name, metric_value in metrics.items():
            # Incluye el run_name en las métricas Prometheus
            metricas_prometheus += f'{{run_id="{run_id}"}} {{run_name="{run_name}"}} {{run_info="{run_info}"}}\n' 
        print(f"Metrics for run {run_id} ({run_name}): {metrics}")
    
    # Renderiza la plantilla HTML con la lista de experimentos
    response = make_response(metricas_prometheus)
    response.headers["Content-Type"] = "text/plain"
    return response 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7654)





# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['GET'])
# def model_output():
        
#         age = float(request.args.get('age'))
#         bmi = float(request.args.get('bmi'))
#         bp = float(request.args.get('bp'))
#         s1 = float(request.args.get('s1'))
#         s2 = float(request.args.get('s2'))
#         s3 = float(request.args.get('s3'))
#         s4 = float(request.args.get('s4'))
#         s5 = float(request.args.get('s5'))
#         s6 = float(request.args.get('s6'))
#         # Carga la URI del mejor modelo
#         logged_model = fetch_best_model_uri()
#         if logged_model:
#             # Load model as a PyFuncModel.
#             loaded_model = mlflow.pyfunc.load_model(logged_model)
#             input_data = pd.DataFrame({"age": [age], "bmi": [bmi], "bp": [bp], "s1": [s1], "s2": [s2], "s3": [s3], "s4": [s4], "s5": [s5], "s6": [s6]})
#             prediction = loaded_model.predict(pd.DataFrame(input_data))
#             return prediction


# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=7654)



    