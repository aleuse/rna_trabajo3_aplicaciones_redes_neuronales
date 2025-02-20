import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Rutas de archivos
MODEL_PATH = "./models/forecasting/model.keras"
SCALER_PATH = "./models/forecasting/scaler.pkl"
OUTPUT_PATH = "./data/output/forecasting/predictions.csv"
DATA_PATH = "./data/input/forecasting/data_processed.csv"  
STEPS_AHEAD = 30

# Cargar el modelo
model = load_model(MODEL_PATH)

# Cargar los escaladores entrenados
with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)

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

def predict_future(df, model, scaler, last_date, n_days):
    """Genera predicciones futuras."""
    if isinstance(last_date, str):
        last_date = pd.to_datetime(last_date)
    
    # Preparar datos históricos
    daily_df = prepare_data(df)
    daily_df = daily_df[daily_df['Date'] <= last_date]
    
    # Crear fechas futuras
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=n_days,
        freq='D'
    )
    
    # Escalar características
    scaled_features = prepare_features(daily_df, scaler)
    initial_sequence = scaled_features[-30:]  # Use last 30 days
    
    # Hacer predicciones
    predictions = recursive_predict(model, initial_sequence, n_days, None, scaler)
    predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1))[:, 0]
    
    return pd.DataFrame({
        'Date': future_dates,
        'Predicted_Sales': predictions_rescaled
    })

# Cargar los datos 
data = pd.read_csv(DATA_PATH)
# Asegurar que la columna Date esté en formato datetime
data['Date'] = pd.to_datetime(data['Date'])

df_predictions = predict_future(data, model, scaler, data['Date'].max(), STEPS_AHEAD)

df_predictions.to_csv(OUTPUT_PATH, index=False)

print(f"Predicciones guardadas en {OUTPUT_PATH}")
