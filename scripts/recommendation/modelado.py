# Este script fue convertido desde un Jupyter Notebook
# Descarga el dataset desde internet en lugar de usar un archivo local

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Flatten, Dense, Concatenate, Dropout

df=pd.read_csv("/kaggle/input/cleaned-dataset/data_processed.csv")

# Codificamos userId y productId
user_encoder = LabelEncoder()
product_encoder = LabelEncoder()

df['userId_encoded'] = user_encoder.fit_transform(df['productId'])  # Usamos productId como usuario ficticio
df['productId_encoded'] = product_encoder.fit_transform(df['productId'])

# División en conjuntos de entrenamiento y prueba
train_data, test_data = train_test_split(df, test_size=0.3, random_state=42)

# Parámetros del modelo
num_users = df['userId_encoded'].nunique()
num_products = df['productId_encoded'].nunique()
embedding_size = 50  # Tamaño del embedding


def build_nn_model(embedding_size=50, dropout_rate=0.3):
    user_input = Input(shape=(1,))
    product_input = Input(shape=(1,))

    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size)(user_input)
    product_embedding = Embedding(input_dim=num_products, output_dim=embedding_size)(product_input)

    user_vector = Flatten()(user_embedding)
    product_vector = Flatten()(product_embedding)

    concat = Concatenate()([user_vector, product_vector])

    dense = Dense(128, activation='relu')(concat)
    dense = Dropout(dropout_rate)(dense)
    dense = Dense(64, activation='relu')(dense)
    dense = Dropout(dropout_rate)(dense)
    output = Dense(1, activation='linear')(dense)  # Predicción de rating

    model = Model(inputs=[user_input, product_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Construimos el modelo
nn_model = build_nn_model()
nn_model.summary()

history = nn_model.fit(
    [train_data['userId_encoded'], train_data['productId_encoded']],
    train_data['ratings'],
    validation_data=([test_data['userId_encoded'], test_data['productId_encoded']], test_data['ratings']),
    epochs=50, batch_size=128, verbose=1
)


# Definición de modelo más profundo
def build_deep_nn_model(embedding_size=50, dropout_rate=0.3):
    user_input = Input(shape=(1,))
    product_input = Input(shape=(1,))
    
    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size)(user_input)
    product_embedding = Embedding(input_dim=num_products, output_dim=embedding_size)(product_input)
    
    user_vector = Flatten()(user_embedding)
    product_vector = Flatten()(product_embedding)
    
    concat = Concatenate()([user_vector, product_vector])
    
    dense = Dense(256, activation='relu')(concat)
    dense = Dropout(dropout_rate)(dense)
    dense = Dense(128, activation='relu')(dense)
    dense = Dropout(dropout_rate)(dense)
    dense = Dense(64, activation='relu')(dense)
    dense = Dropout(dropout_rate)(dense)
    output = Dense(1, activation='linear')(dense)

    model = Model(inputs=[user_input, product_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

deep_nn_model = build_deep_nn_model()
deep_nn_model.summary()

history_deep = deep_nn_model.fit(
    [train_data['userId_encoded'], train_data['productId_encoded']], 
    train_data['ratings'], 
    validation_data=([test_data['userId_encoded'], test_data['productId_encoded']], test_data['ratings']),
    epochs=50, batch_size=128, verbose=1
)

#modelo basado en autoencoders
def build_autoencoder(embedding_size=50, dropout_rate=0.3):
    user_input = Input(shape=(1,))
    product_input = Input(shape=(1,))
    
    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size)(user_input)
    product_embedding = Embedding(input_dim=num_products, output_dim=embedding_size)(product_input)
    
    user_vector = Flatten()(user_embedding)
    product_vector = Flatten()(product_embedding)
    
    concat = Concatenate()([user_vector, product_vector])
    
    encoder = Dense(128, activation='relu')(concat)
    encoder = Dropout(dropout_rate)(encoder)
    encoder = Dense(64, activation='relu')(encoder)
    
    decoder = Dense(128, activation='relu')(encoder)
    decoder = Dropout(dropout_rate)(decoder)
    output = Dense(1, activation='linear')(decoder)

    model = Model(inputs=[user_input, product_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

autoencoder_model = build_autoencoder()
autoencoder_model.summary()


history_autoencoder = autoencoder_model.fit(
    [train_data['userId_encoded'], train_data['productId_encoded']], 
    train_data['ratings'], 
    validation_data=([test_data['userId_encoded'], test_data['productId_encoded']], test_data['ratings']),
    epochs=50, batch_size=128, verbose=1
)


# Guardar los modelos en formato .h5
nn_model.save("modelo_nn_simple.h5")
deep_nn_model.save("modelo_nn_profundo.h5")
autoencoder_model.save("modelo_autoencoder.h5")

print("Modelos guardados exitosamente.")


import matplotlib.pyplot as plt

def plot_training_history(histories, labels):
    plt.figure(figsize=(12,6))
    for history, label in zip(histories, labels):
        plt.plot(history.history['val_mae'], label=f'Val MAE {label}')
    plt.title('Comparación de Modelos')
    plt.xlabel('Épocas')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()

plot_training_history(
    [history, history_deep, history_autoencoder], 
    ["Modelo Simple", "Modelo Profundo", "Autoencoder"]
)


from sklearn.model_selection import train_test_split

# Separar características y etiquetas
X = df[['userId_encoded', 'productId_encoded']]
y = df['ratings']

# División en entrenamiento y prueba (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extraer las columnas separadas para usuarios y productos
X_train_user, X_train_product = X_train['userId_encoded'], X_train['productId_encoded']
X_test_user, X_test_product = X_test['userId_encoded'], X_test['productId_encoded']

from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def evaluar_modelo(model, X_test, y_test, threshold=3.5):
    """
    Calcula métricas de evaluación: Precision, Recall y F1-Score.

    Parámetros:
    - model: Modelo de red neuronal entrenado.
    - X_test: Datos de entrada de prueba (usuarios y productos).
    - y_test: Valores reales de ratings.
    - threshold: Umbral para clasificar recomendaciones positivas.

    Retorna:
    - Precisión, Recall y F1-Score
    """
    # Hacer predicciones con el modelo
    y_pred = model.predict(X_test)

    # Convertir las predicciones en clases binarias (buenas/malas recomendaciones)
    y_pred_bin = (y_pred >= threshold).astype(int)
    y_test_bin = (y_test >= threshold).astype(int)

    # Calcular métricas
    precision = precision_score(y_test_bin, y_pred_bin, average='weighted')
    recall = recall_score(y_test_bin, y_pred_bin, average='weighted')
    f1 = f1_score(y_test_bin, y_pred_bin, average='weighted')

    return precision, recall, f1

# Evaluamos los tres modelos
precision_nn, recall_nn, f1_nn = evaluar_modelo(nn_model, [X_test_user, X_test_product], y_test)
precision_deep, recall_deep, f1_deep = evaluar_modelo(deep_nn_model, [X_test_user, X_test_product], y_test)
precision_autoencoder, recall_autoencoder, f1_autoencoder = evaluar_modelo(autoencoder_model, [X_test_user, X_test_product], y_test)

# Imprimir resultados
print(f"Red Neuronal Simple - Precision: {precision_nn:.4f}, Recall: {recall_nn:.4f}, F1-Score: {f1_nn:.4f}")
print(f"Red Profunda - Precision: {precision_deep:.4f}, Recall: {recall_deep:.4f}, F1-Score: {f1_deep:.4f}")
print(f"Autoencoder - Precision: {precision_autoencoder:.4f}, Recall: {recall_autoencoder:.4f}, F1-Score: {f1_autoencoder:.4f}")


import matplotlib.pyplot as plt
import numpy as np

# Datos de las métricas
modelos = ["Red Neuronal Simple", "Red Profunda", "Autoencoder"]
precision_scores = [precision_nn, precision_deep, precision_autoencoder]
recall_scores = [recall_nn, recall_deep, recall_autoencoder]
f1_scores = [f1_nn, f1_deep, f1_autoencoder]

x = np.arange(len(modelos))  # Posiciones en el eje X

# Crear gráfico de barras
plt.figure(figsize=(10,6))
plt.bar(x - 0.2, precision_scores, width=0.2, label="Precision", color='royalblue')
plt.bar(x, recall_scores, width=0.2, label="Recall", color='seagreen')
plt.bar(x + 0.2, f1_scores, width=0.2, label="F1-Score", color='tomato')

plt.xticks(x, modelos)
plt.xlabel("Modelos de Recomendación")
plt.ylabel("Puntuación")
plt.title("Comparación de Métricas de Evaluación")
plt.legend()
plt.show()

def plot_learning_curves(history, title):
    """
    Grafica la curva de aprendizaje de un modelo.

    Parámetros:
    - history: Historial de entrenamiento de Keras (loss y métricas).
    - title: Título de la gráfica.
    """
    plt.figure(figsize=(12,5))

    # Gráfico de Pérdida
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label="Train Loss", color='royalblue')
    plt.plot(history.history['val_loss'], label="Validation Loss", color='tomato')
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.title(f"{title} - Curva de Pérdida")
    plt.legend()

    # Gráfico de MAE
    plt.subplot(1,2,2)
    plt.plot(history.history['mae'], label="Train MAE", color='seagreen')
    plt.plot(history.history['val_mae'], label="Validation MAE", color='darkorange')
    plt.xlabel("Épocas")
    plt.ylabel("MAE")
    plt.title(f"{title} - Curva de MAE")
    plt.legend()

    plt.show()

# Graficar curvas de aprendizaje para cada modelo
plot_learning_curves(history, "Red Neuronal Simple")
plot_learning_curves(history_deep, "Red Profunda")
plot_learning_curves(history_autoencoder, "Autoencoder")


