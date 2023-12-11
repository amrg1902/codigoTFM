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

def fetch_model(model_name):
    try:
        # Carga el modelo más reciente
        model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None       

@app.route('/')
def index():
    # Obtén el mejor modelo y su métrica asociada
    best_model_name = fetch_best_model()
    best_model_metric = None

    if best_model_name:
        # Obtén el valor de la métrica (MSE) del mejor modelo
        experimento_id = mlflow.get_experiment_by_name(nombre_experimento).experiment_id
        best_model_run = mlflow.search_runs(experiment_ids=experimento_id, filter_string=f"run_name = '{best_model_name}'").iloc[0]
        best_model_metric = best_model_run["MSE"]

    return render_template('index.html', best_model=best_model_name, best_model_metric=best_model_metric)

@app.route('/model_details')
def model_details():
    # Obtén el mejor modelo y carga los detalles del modelo
    best_model_name = fetch_best_model()
    model = fetch_model(best_model_name)

    return render_template('model_details.html', best_model=best_model_name, model_details=model)


# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['GET'])
# def model_output():
#     try:
#         age = float(request.args.get('age'))
#         bmi = float(request.args.get('bmi'))
#         bp = float(request.args.get('bp'))
#         s1 = float(request.args.get('s1'))
#         s2 = float(request.args.get('s2'))
#         s3 = float(request.args.get('s3'))
#         s4 = float(request.args.get('s4'))
#         s5 = float(request.args.get('s5'))
#         s6 = float(request.args.get('s6'))

#         model_name = fetch_best_model()
#         if model_name:
#             model = fetch_model(model_name)
#             print(model)
#             if model:
#                 input_data = pd.DataFrame({"age": [age], "bmi": [bmi], "bp": [bp], "s1": [s1], "s2": [s2], "s3": [s3], "s4": [s4], "s5": [s5], "s6": [s6]})
#                 prediction = model.predict(input_data)
#                 print(prediction)
#                 return render_template('index.html', prediction=f"Prediction: {prediction[0]}")
#             else:
#                 return render_template('index.html', prediction="Error: Model not loaded")
#         else:
#             return render_template('index.html', prediction="Error: No best model found")
#     except Exception as e:
#         return render_template('index.html', prediction=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7654)
