import pickle
import argparse
import os
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Definir rutas
DATA_PATH = "./data/input/forecasting/"
MODEL_PATH = "./models/forecasting/model.keras"
SCALER_PATH = "./models/forecasting/scaler.pkl"
OUTPUT_PATH = "./data/output/forecasting/"

seq_length = 30
pred_length = 30

# Crear directorios si no existen
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Cargar datos
data = pd.read_csv(os.path.join(DATA_PATH, 'data_processed.csv'))


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
	# Create feature matrix
    feature_columns = [
		'Weekly_Sales', 
	]
	
    scaled_features = scaler.transform(df[feature_columns])
	
    return scaled_features

def create_sequences(data, seq_length=30):
	"""Create sequences without fixed prediction length"""
	X = []
	for i in range(len(data) - seq_length + 1):
		X.append(data[i:(i + seq_length)])
	return np.array(X)


def train_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    # Compilación del modelo
    model.compile(optimizer='adam', loss='mse')
    checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True)
    history = model.fit(X_train, y_train, epochs=300, batch_size=48, validation_split=0.2, callbacks=[checkpoint])
    return history, model

def recursive_predict(model, initial_sequence, n_steps, future_features, scaler):
    """
    Realiza predicciones recursivas utilizando un modelo de series temporales.
    
    Args:
        model: Modelo entrenado
        initial_sequence: Secuencia inicial de entrada (shape: [sequence_length, n_features])
        n_steps: Número de pasos futuros a predecir
        future_features: Características futuras conocidas
    
    Returns:
        np.array: Array con las predicciones
    """
    predictions = []
    current_sequence = initial_sequence.copy()
    
    for i in range(n_steps):
        # Realizar predicción
        next_pred = model.predict(current_sequence.reshape(1, *current_sequence.shape))[0]
        predictions.append(next_pred)
        
        # Actualizar la secuencia
        current_sequence = np.roll(current_sequence, -1, axis=0)
        
        # Actualizar el último valor con la predicción
        current_sequence[-1, 0] = next_pred
        
        # Actualizar características temporales si están disponibles
        if i < len(future_features):
            current_sequence[-1, 1:] = future_features[i, 1:]  # Mantener todas las características excepto weekly_sales
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))


daily_df = prepare_data(data)
scaler = MinMaxScaler().fit(daily_df[['Weekly_Sales']])
scaled_features = prepare_features(daily_df, scaler)

# Guardar escaladores
with open(SCALER_PATH, "wb") as file:
    pickle.dump(scaler, file)

# Create sequences
X = create_sequences(scaled_features, seq_length)
y = scaled_features[seq_length:, 0]

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size-pred_length:train_size]

# Entrenar modelo
history, model = train_model()

# Guardar gráfico de pérdida
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.legend()
plt.title('Loss durante el entrenamiento')
plt.savefig(os.path.join(OUTPUT_PATH, 'loss.png'))
plt.close()

# Evaluar modelo
initial_sequence = X_test[0]
y_pred = recursive_predict(
    model,
    initial_sequence,
    pred_length,
    scaled_features[train_size:train_size+pred_length],
    scaler
)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred))
mae = mean_absolute_error(y_test_inv, y_pred)
mape = mean_absolute_percentage_error(y_test_inv, y_pred)
print(f"RMSE: {rmse}, MAE: {mae}, MAPE: {mape}")

# Guardar gráfico de predicciones
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Valores Reales')
plt.plot(y_pred, label='Predicciones')
plt.title('Predicciones vs Valores Reales')
plt.legend()
plt.savefig(os.path.join(OUTPUT_PATH, 'predictions.png'))
plt.close()
