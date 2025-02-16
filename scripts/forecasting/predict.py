import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Rutas de archivos
MODEL_PATH = "./models/forecasting/model.keras"
SCALER_FEATURES_PATH = "./models/forecasting/scaler_features.pkl"
SCALER_TARGET_PATH = "./models/forecasting/scaler_target.pkl"
OUTPUT_PATH = "./data/output/forecasting/predictions.csv"
DATA_PATH = "./data/input/forecasting/new_data.csv"  
STEPS_AHEAD = 5

# Cargar el modelo
model = load_model(MODEL_PATH)

# Cargar los escaladores entrenados
with open(SCALER_FEATURES_PATH, "rb") as file:
    scaler_features = pickle.load(file)
with open(SCALER_TARGET_PATH, "rb") as file:
    scaler_target = pickle.load(file)

# Cargar los datos nuevos
data = pd.read_csv(DATA_PATH)
data["Date"] = pd.to_datetime(data["Date"])
data['weekday'] = data['Date'].dt.weekday
data['month'] = data['Date'].dt.month
data.set_index("Date", inplace=True)
data.fillna(method='ffill', inplace=True)

# Seleccionar las mismas características usadas en entrenamiento
features = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "IsHoliday", "Size", "weekday", "month"]

# Asegurar consistencia en las columnas
data = data[features]

# Escalar los datos de entrada
data_scaled = scaler_features.transform(data) 

# Tomar solo las últimas 30 semanas para la predicción
sequence_length = 30
last_sequence = data_scaled[-sequence_length:]

# Verificar forma de entrada
num_features = last_sequence.shape[1]  # Debe coincidir con el entrenamiento
# Verificar que tenga la forma correcta
if last_sequence.shape != (sequence_length, last_sequence.shape[1]):
    raise ValueError(f"La forma de 'last_sequence' es incorrecta: {last_sequence.shape}, debería ser {(sequence_length, len(features))}.")


# Función para predecir los próximos 30 valores
def predict_future(model, last_sequence, steps_ahead, scaler_target):
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
    return scaler_target.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Predecir los próximos 5 valores
predicted_sales = predict_future(model, last_sequence, STEPS_AHEAD, scaler_target)

# Guardar las predicciones en un CSV
df_predictions = pd.DataFrame({
    "Date": pd.date_range(start=data.index[-1] + pd.Timedelta(weeks=1), periods=STEPS_AHEAD, freq="W"),
    "Predicted_Weekly_Sales": predicted_sales.flatten()
})
df_predictions.to_csv(OUTPUT_PATH, index=False)

print(f"Predicciones guardadas en {OUTPUT_PATH}")
