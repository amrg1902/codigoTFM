# Utiliza una imagen base de Python con MLflow preinstalado
FROM python:3.9

WORKDIR /api

COPY ./api/requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV MLFLOW_TRACKING_URI=http://mlflow:80

EXPOSE 8000
CMD ["sh", "-c", "uvicorn api:app --port 8086 --host 0.0.0.0"]