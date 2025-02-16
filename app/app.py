import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")
global sequence_length
sequence_length = 30

# Función para cargar los objetos
@st.cache_resource
def load_forecasting_objects():
    model = load_model(os.path.join(MODELS_DIR, "forecasting", "modelo_forecasting.keras"))
    with open(os.path.join(MODELS_DIR, "forecasting","scaler_features_forecasting.pkl"), "rb") as file:
        scaler_features = pickle.load(file)
    with open(os.path.join(MODELS_DIR, "forecasting","scaler_target_forecasting.pkl"), "rb") as file:
        scaler_target = pickle.load(file)
    
    return model, scaler_features, scaler_target

def load_forecasting_data(data_path, scaler_features):
    # Cargar los datos nuevos
    data = pd.read_csv(data_path)
    data["Date"] = pd.to_datetime(data["Date"])
    data['weekday'] = data['Date'].dt.weekday
    data['month'] = data['Date'].dt.month
    data.set_index("Date", inplace=True)
    data.fillna(method='ffill', inplace=True)

    # Seleccionar las mismas características usadas en entrenamiento
    features = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "IsHoliday", "Size", "weekday", "month"]

    # Asegurar consistencia en las columnas
    temp = data[features]

    # Escalar los datos de entrada
    data_scaled = scaler_features.transform(temp) 

    # Tomar solo las últimas 30 semanas para la predicción
    last_sequence = data_scaled[-sequence_length:]

    return data, last_sequence

# Función para predecir los próximos valores
def make_forecasting_predictions(model, last_sequence, steps_ahead, scaler_target, data):
    future_predictions = []
    input_seq = last_sequence.copy()

    for _ in range(steps_ahead):
        # Hacer una predicción
        pred = model.predict(input_seq.reshape(1, sequence_length, -1))[0, 0]

        # Guardar la predicción
        future_predictions.append(pred)

        # Actualizar la secuencia eliminando el primer elemento y agregando la nueva predicción
        next_input = np.hstack(([pred], input_seq[-1, 1:]))  # Mantiene las demás características
        input_seq = np.vstack((input_seq[1:], next_input))

    # Desescalar las predicciones
    predicted_sales = scaler_target.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    df_predictions = pd.DataFrame({
        "Date": pd.date_range(start=data.index[-1] + pd.Timedelta(weeks=1), periods=steps_ahead, freq="W"),
        "Predicted_Weekly_Sales": predicted_sales.flatten()
    })
    return df_predictions

# Configuración de la página
st.set_page_config(page_title="Sistema Integrado Gestión de Ventas", layout="wide")

# Título
st.title("📊 Sistema Integrado Gestión de Ventas")

# Diseño de tarjetas para los módulos
col1, col2, col3 = st.columns(3)

with col1:
    st.image("./app/resources/forecasting.png", use_container_width=True)
    st.subheader("📈 Predicción de Demanda")
    if st.button(label="Ir a Predicción de Demanda", key="forecasting"):
        st.session_state["page"] = "prediccion"

with col2:
    st.image("./app/resources/classification.png", use_container_width=True)
    st.subheader("🖼️ Clasificación de Productos")
    st.button(label="Próximamente", key="clasificacion", disabled=True)

with col3:
    st.image("./app/resources/recommendation.png", use_container_width=True)
    st.subheader("🛒 Recomendación Personalizada")
    st.button(label="Próximamente", key="recommendation", disabled=True)

# Página de predicción de demanda
if "page" in st.session_state and st.session_state["page"] == "prediccion":
    st.subheader("📈 Predicción de Demanda")
    
    # Cargar objetos
    model, scaler_features, scaler_target = load_forecasting_objects()
    
    # Cargar data
    data, last_sequence = load_forecasting_data(os.path.join(DATA_DIR, "forecasting", "data.csv"), scaler_features)

    # Input de usuario para número de semanas a predecir
    steps = st.slider("Selecciona el número de semanas predecir", min_value=1, max_value=16, value=6)

    # Botón para hacer predicción
    if st.button("Generar Predicción"):
        forecast = make_forecasting_predictions(model, last_sequence, steps, scaler_target, data)

        # Mostrar tabla con predicciones
        st.write("### Tabla de Predicciones")
        st.dataframe(forecast)

        # Gráfico de predicción
        st.write("### Gráfico de Predicción")
        fig, ax = plt.subplots(figsize=(10, 5))

        # Mostrar solo los últimos 6 meses de datos históricos
        last_6_months = data.last('26W')  # 26 semanas = ~6 meses
        
        # Obtener el último punto de los datos históricos
        last_historical_date = last_6_months.index[-1]
        last_historical_value = last_6_months["Weekly_Sales"].iloc[-1]

        # Crear un punto de conexión
        connection_dates = [last_historical_date, forecast["Date"].iloc[0]]
        connection_values = [last_historical_value, forecast["Predicted_Weekly_Sales"].iloc[0]]

        # Graficar datos originales, conexión y predicción
        ax.plot(last_6_months.index, last_6_months["Weekly_Sales"], 
                label="Datos Originales", color="black")
        ax.plot(connection_dates, connection_values, 
                color="blue") 
        ax.plot(forecast["Date"], forecast["Predicted_Weekly_Sales"], 
                label="Predicción", color="blue")

        # Ajustar límites del eje x usando solo el período relevante
        all_dates = pd.concat([pd.Series(last_6_months.index), pd.Series(forecast["Date"])])
        ax.set_xlim([all_dates.min(), all_dates.max()])

        # Ajustar formato de fechas en eje x
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)  # Agregar cuadrícula
        plt.tight_layout()  # Ajustar layout para evitar corte de etiquetas

        ax.set_title("Predicción de Demanda")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Ventas")
        ax.legend()
        st.pyplot(fig)
