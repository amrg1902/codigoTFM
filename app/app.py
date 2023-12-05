import streamlit as st
import requests as rs


age = st.text_input("Age")
bmi = st.text_input("Body Mass index")
bp = st.text_input("Blood Pressure")
s1 = st.text_input("s1")
s2 = st.text_input("s2")
s3 = st.text_input("s3")
s4 = st.text_input("s4")
s5 = st.text_input("s5")
s6 = st.text_input("s6")

#Función para Obtener Respuestas de la API FastAPI
def get_api(params):
    #url = f"http://api:8086/predict/"
    # Reemplaza 'http://api:8086' con la URL correcta de tu servidor Flask
    url = "http://localhost:7000/predict/"
    response = rs.get(url, params=params)
    return response.content

#Lógica Streamlit para Enviar Datos a la API y Mostrar Respuesta
if st.button("Get response"):
    params = {
        "age": float(age),
        "bmi": float(bmi),
        "bp": float(bp),
        "s1": float(s1),
        "s2": float(s2),
        "s3": float(s3),
        "s4": float(s4),
        "s5": float(s5),
        "s6": float(s6)
    }

    data = get_api(params)
    st.write(data)


