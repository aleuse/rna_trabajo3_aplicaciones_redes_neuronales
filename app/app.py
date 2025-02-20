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
import os

# Ruta donde están las partes del modelo
ruta_modelo = "app\models\classification"

# Obtener y ordenar los archivos de modelo por nombre
partes = sorted(
    [f for f in os.listdir(ruta_modelo) if f.startswith("modelo_completo_part_")]
)

# Unir todas las partes en un solo archivo
with open(os.path.join(ruta_modelo, "modelo_reconstruido.keras"), "wb") as salida:
    for parte in partes:
        with open(os.path.join(ruta_modelo, parte), "rb") as fragmento:
            salida.write(fragmento.read())


# Función para cargar los objetos
@st.cache_resource
def load_forecasting_objects():
    model = load_model(
        os.path.join(MODELS_DIR, "forecasting", "modelo_forecasting.keras")
    )
    with open(
        os.path.join(MODELS_DIR, "forecasting", "scaler_forecasting.pkl"), "rb"
    ) as file:
        scaler = pickle.load(file)

    return model, scaler

def prepare_data(df):
	# Convert weekly data to daily using linear interpolation
	df = df.copy()
	df['Date'] = pd.to_datetime(df['Date'])
	
	# Aggregate sales across all stores and departments
	df = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
	
	# Create daily dates
	daily_dates = pd.date_range(df['Date'].min(), df['Date'].max(), freq='D')
	daily_df = pd.DataFrame({'Date': daily_dates})
	
	# Merge with original data
	daily_df = daily_df.merge(df, on='Date', how='left')
	
	# Interpolate missing values
	daily_df['Weekly_Sales'] = daily_df['Weekly_Sales'].interpolate(method='linear')
	
	return daily_df

def prepare_features(df, scaler):
    """Prepara y escala las características para el modelo."""
    # Solo escalar Weekly_Sales
    scaled_features = scaler.transform(df[["Weekly_Sales"]])
    return scaled_features

def recursive_predict(model, initial_sequence, n_steps, future_features, scaler):
    """Realiza predicciones recursivas."""
    predictions = []
    current_sequence = initial_sequence.copy()
    
    for i in range(n_steps):
        # Predicción
        next_pred = model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)[0]
        predictions.append(next_pred)
        
        # Actualizar secuencia
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = next_pred
    
    return np.array(predictions)

def predict_future(scaled_features, model, scaler, last_date, n_days):
    """Genera predicciones futuras."""    
    # Crear fechas futuras
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=n_days,
        freq='D'
    )
    
    # Escalar características
    initial_sequence = scaled_features[-30:]  # Use last 30 days
    
    # Hacer predicciones
    predictions = recursive_predict(model, initial_sequence, n_days, None, scaler)
    predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1))[:, 0]
    
    return pd.DataFrame({
        'Date': future_dates,
        'Predicted_Sales': predictions_rescaled
    })



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
    if st.button(label="Clasificación de Productos", key="clasificacion"):
        st.session_state["page"] = "clasificacion"

with col3:
    st.image("./app/resources/recommendation.png", use_container_width=True)
    st.subheader("🛒 Recomendación Personalizada")
    st.button(label="Próximamente", key="recommendation", disabled=True)

# Página de predicción de demanda
if "page" in st.session_state and st.session_state["page"] == "prediccion":
    st.subheader("📈 Predicción de Demanda")

    # Cargar objetos
    model, scaler = load_forecasting_objects()

    # Cargar los datos 
    data = pd.read_csv(os.path.join(DATA_DIR, "forecasting", "data.csv"))
    # Asegurar que la columna Date esté en formato datetime
    data['Date'] = pd.to_datetime(data['Date'])
    last_date = data['Date'].max()
    
    # Preparar datos históricos
    daily_df = prepare_data(data)
    daily_df = daily_df[daily_df['Date'] <= last_date]
    scaled_features = prepare_features(daily_df, scaler)

    # Input de usuario para número de días a predecir
    steps = st.slider(
        "Selecciona el número de días predecir", min_value=1, max_value=60, value=30
    )

    # Botón para hacer predicción
    if st.button("Generar Predicción"):
        df_predictions = predict_future(scaled_features, model, scaler, last_date, steps)

        # Mostrar tabla con predicciones
        st.write("### Tabla de Predicciones")
        st.dataframe(df_predictions)

        # Gráfico de predicción
        st.write("### Gráfico de Predicción")
        fig, ax = plt.subplots(figsize=(10, 5))


        # Graficar datos originales, conexión y predicción
        ax.plot(
            daily_df["Date"],
            daily_df["Weekly_Sales"],
            label="Datos Originales",
            color="black",
        )
        # ax.plot(connection_dates, connection_values, color="blue")
        ax.plot(
            df_predictions["Date"],
            df_predictions["Predicted_Sales"],
            label="Predicción",
            color="blue",
        )
        # Ajustar formato de fechas en eje x
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.7)  # Agregar cuadrícula
        plt.tight_layout()  # Ajustar layout para evitar corte de etiquetas

        ax.set_title("Predicción de Demanda")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Ventas")
        ax.legend()
        st.pyplot(fig)

# Página de clasificación de productos
if "page" in st.session_state and st.session_state["page"] == "clasificacion":
    import streamlit as st
    import numpy as np
    import tensorflow as tf
    from PIL import Image

    # Cargar el modelo
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model(
            os.path.join(MODELS_DIR, "classification", "modelo_reconstruido.keras")
        )

    model = load_model()

    # Definir clases
    CLASSES = ["jeans", "tv", "tshirt", "sofa"]

    # Función para preprocesar la imagen
    def preprocess_image(image):
        try:
            image = image.convert("RGB").resize(
                (256, 256)
            )  # Convertir a RGB y redimensionar
            img_array = np.array(image) / 255.0  # Normalizar valores
            img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión de batch
            return img_array
        except Exception as e:
            st.error(f"Error al procesar la imagen: {e}")
            return None

    # Interfaz de usuario
    st.title("🖼️ Clasificación de Productos")
    st.write(
        "Sube una imagen en formato jpg, jpeg o png y el modelo la clasificará en una de las siguientes categorías:"
    )
    st.write(f"{', '.join(CLASSES)}")

    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", width=300)

        if st.button("Clasificar imagen"):
            img_array = preprocess_image(image)
            if img_array is not None:
                with st.spinner("Clasificando..."):
                    prediction = model.predict(img_array)
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    resultado = CLASSES[predicted_class]

                    st.success(f"La imagen fue clasificada como: {resultado}")
                    # Redondear a tres decimales y convertir en porcentaje
                    probabilidades_legibles = [
                        round(p * 100, 2) for p in prediction.flatten()
                    ]

                    # Mostrar de manera más clara
                    for i, p in enumerate(probabilidades_legibles):
                        st.write(f"Clase {CLASSES[i]}: {round(p,2)}%")
