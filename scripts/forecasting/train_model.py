import pickle
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Definir rutas
DATA_PATH = "./data/input/forecasting/"
MODEL_PATH = "./models/forecasting/model.keras"
SCALER_FEATURES_PATH = "./models/forecasting/scaler_features.pkl"
SCALER_TARGET_PATH = "./models/forecasting/scaler_target.pkl"
OUTPUT_PATH = "./data/output/forecasting/"

# Crear directorios si no existen
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Cargar datos
data = pd.read_csv(os.path.join(DATA_PATH, 'data_processed.csv'))

# Agrupar las ventas semanales
temp_data = data.groupby('Date')['Weekly_Sales'].sum().reset_index()

# Fusionar con el resto de las variables
data = temp_data.merge(data[['Date', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday', 'Size']]
                    .groupby('Date').mean(), on='Date')

# Crear nuevas características
data['Date'] = pd.to_datetime(data['Date'])
data['weekday'] = data['Date'].dt.weekday
data['month'] = data['Date'].dt.month
data.set_index('Date', inplace=True)
data.fillna(method='ffill', inplace=True)

# Separar la variable objetivo antes de escalar
target = data[['Weekly_Sales']]
features = data.drop(columns=['Weekly_Sales'])

# Escalar características y variable objetivo por separado
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))

features_scaled = scaler_features.fit_transform(features)
target_scaled = scaler_target.fit_transform(target)

# Guardar escaladores
with open(SCALER_FEATURES_PATH, "wb") as file:
    pickle.dump(scaler_features, file)
with open(SCALER_TARGET_PATH, "wb") as file:
    pickle.dump(scaler_target, file)


def create_sequences(features, target, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        # Todas las variables features
        X.append(features[i:i + sequence_length])
        # Solo la variable target
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 30
X, y = create_sequences(features_scaled, target_scaled, sequence_length)

# División en entrenamiento y prueba
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Definir el modelo
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

# Compilar modelo
model.compile(optimizer='adam', loss='mse')

# Callback para guardar el mejor modelo
checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min')

# Entrenar modelo
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint], verbose=1)

# Guardar gráfico de pérdida
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.legend()
plt.title('Loss durante el entrenamiento')
plt.savefig(os.path.join(OUTPUT_PATH, 'loss.png'))
plt.close()

# Evaluar modelo
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"RMSE: {rmse}, MAE: {mae}, MAPE: {mape}")

# Guardar gráfico de predicciones
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Valores Reales')
plt.plot(y_pred, label='Predicciones')
plt.title('Predicciones vs Valores Reales')
plt.legend()
plt.savefig(os.path.join(OUTPUT_PATH, 'predictions.png'))
plt.close()
