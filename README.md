# Sistema Inteligente Integrado para Predicción, Clasificación y Recomendación en Comercio Electrónico

---

## **Objetivo del Proyecto**

Desarrollar un sistema inteligente basado en aprendizaje profundo que integre la predicción de demanda, la clasificación de productos mediante imágenes y un sistema de recomendaciones personalizadas para mejorar la gestión y el rendimiento de un comercio electrónico.

## **Objetivos Específicos**

1. **Predicción de Demanda:**
    - Entrenar y evaluar un modelo de series de tiempo utilizando datos históricos de ventas para anticipar la demanda de los próximos 30 días.
    - Validar el modelo utilizando métricas como **RMSE** y **MAE**.
2. **Clasificación Automática de Productos:**
    - Implementar un modelo de clasificación de imágenes basado en redes neuronales convolucionales (CNNs) para categorizar automáticamente productos en categorías específicas como Televisores, Sofás, Jeans y Camisetas.
    - Evaluar el modelo con métricas como **accuracy** y **F1-score**.
3. **Sistema de Recomendación Personalizada:**
    - Construir un sistema de recomendaciones basado en datos de interacción usuario-producto.
    - Evaluar el sistema utilizando métricas como **Precision**, **Recall** y **Coverage**.
4. **Desarrollo de una Herramienta Web:**
    - Crear una interfaz web funcional que permita interactuar con las soluciones desarrolladas.
    - Incluir funcionalidades para cargar imágenes, visualizar gráficos de predicción y probar recomendaciones personalizadas.

---

## 📂 **Estructura del Proyecto**

### Predicción de la Demanda

#### 📝 Análisis Exploratorio de Datos
- **Ubicación:** [`notebooks/forecasting/1_exploracion.ipynb`](notebooks/forecasting/1_exploracion.ipynb) 
- **Descripción:**
  - Análisis de series temporales de ventas semanales.
  - Identificación de tendencias y estacionalidad.
  - Análisis de variables predictoras (*temperatura, precio combustible, CPI, etc.*).
  - Preprocesamiento y limpieza de datos.

#### 🤖 Modelado y Evaluación
- **Ubicación:** [`notebooks/forecasting/2_modelado.ipynb`](notebooks/forecasting/2_modelado.ipynb) 
- **Descripción:**
  - Implementación de modelo **SARIMA**.
  - Desarrollo de modelos de redes neuronales (**LSTM, GRU**).
  - Evaluación comparativa de modelos.
  - Análisis de predicciones.

#### 📊 Script de Entrenamiento 
- **Ubicación:** [`scripts/forecasting/train_model.py`](scripts/forecasting/train_model.py) 
- **Descripción:**
  - Script de entrenamiento del modelo.
  - Utilidades de preprocesamiento.

#### 📊 Script de Predicción 
- **Ubicación:** [`scripts/forecasting/train_model.py`](scripts/forecasting/train_model.py) 
- **Descripción:**
  - Script de predicción para nuevos datos.
  - Utilidades de graficación.

### Clasificación Automática de Productos

### Recomendación Personalizada

### 🌐 **Aplicación Web**
- **Ubicación:** [`app/`](app/)  
- **Descripción:** 
  - Aplicación web desarrollada con Streamlit.
  - Permite a los usuarios interactuar con las soluciones desarrolladas.

---

## ⚙️ Requisitos del Sistema
- **Lenguaje:** Python 3.10+
- **Librerías principales:**
  - `TensorFlow` (para redes neuronales)
  - `Pandas` y `NumPy` (para análisis de datos)
  - `Scikit-learn` (para preprocesamiento y evaluación)
  - `Statsmodels` (para modelos SARIMA)
- Ver el archivo [`requirements.txt`](requirements.txt) para más detalles.

---


## 🚀 Cómo Ejecutar el Proyecto

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/aleuse/rna_trabajo3_aplicaciones_redes_neuronales.git
   cd rna_trabajo3_aplicaciones_redes_neuronales
   ```

2. **Crear y activar entorno virtual:**
   ```bash
    python -m venv venv
    source venv/bin/activate  # En Mac/Linux
    venv\Scripts\activate     # En Windows
   ```

3. **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
   ```

### Predicción de Demanda

1. **Entrenar modelo**:
    ```bash
    python scripts/forecasting/train_model.py
   ```

2. **Ejecutar predicciones**:
    ```bash
    python scripts/forecasting/predict.py
   ```

### Clasificación Automática de Productos

### Recomendación Personalizada

### Aplicación Web

1. **Ejecutar app:**
    ```bash
    streamlit run app/app.py
   ```