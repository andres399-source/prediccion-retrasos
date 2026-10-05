import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Predicción de Retrasos", page_icon="🚚")
st.title("🚚 Predicción de Retrasos en Entregas")
st.write("Ingresa los datos del envío para predecir si llegará a tiempo o se retrasará.")

@st.cache_resource
def cargar_modelo():
    return load_model('modelo_retrasos.keras')

modelo = cargar_modelo()

with st.form("formulario"):
    peso = st.number_input("Peso del paquete (gramos):", min_value=100, max_value=8000, value=3000)
    distancia = st.number_input("Distancia (km):", min_value=50, max_value=3000, value=1000)
    costo = st.number_input("Costo del producto:", min_value=50, max_value=2000, value=500)
    compras = st.number_input("Compras previas:", min_value=0, max_value=15, value=3)
    tipo_envio = st.selectbox("Tipo de envío:", ['Barco', 'Vuelo', 'Carretera'])
    importancia = st.selectbox("Importancia del producto:", ['Baja', 'Media', 'Alta'])
    
    submit = st.form_submit_button("Predecir")

if submit:
    datos = pd.DataFrame({
        'Weight_in_gms': [peso],
        'Distance_km': [distancia],
        'Cost_of_the_Product': [costo],
        'Prior_purchases': [compras],
        'Mode_of_Shipment': [tipo_envio],
        'Product_importance': [importancia]
    })
    
    datos_proc = pd.get_dummies(datos, drop_first=True)
    
    columnas_esperadas = ['Weight_in_gms', 'Distance_km', 'Cost_of_the_Product', 
                          'Prior_purchases', 'Mode_of_Shipment_Vuelo',
                          'Product_importance_Baja', 'Product_importance_Media']
    
    for col in columnas_esperadas:
        if col not in datos_proc.columns:
            datos_proc[col] = 0
    
    datos_proc = datos_proc[columnas_esperadas]
    
    prediccion = modelo.predict(datos_proc.values)
    
    st.write(f"**Probabilidad de retraso:** {prediccion[0][0]*100:.2f}%")
    
    if prediccion[0][0] > 0.5:
        st.error("⚠️ PREDICCIÓN: El envío SE RETRASARÁ.")
    else:
        st.success("✅ PREDICCIÓN: El envío LLEGARÁ A TIEMPO.")