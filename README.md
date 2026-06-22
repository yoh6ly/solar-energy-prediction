# 🌞 Predicción de Generación Solar con LSTM
### Santo Domingo, República Dominicana

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Azure ML](https://img.shields.io/badge/Azure-ML%20Notebooks-0078D4)
![R2](https://img.shields.io/badge/R²-0.9924-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)

---
## 📖 Sobre el Proyecto

La energía solar representa una de las alternativas más prometedoras para resolver 
el déficit energético que históricamente ha afectado a República Dominicana. Sin 
embargo, su naturaleza intermitente — condicionada por factores meteorológicos como 
la nubosidad, la temperatura y la humedad — representa un desafío técnico significativo 
para su integración eficiente en la red eléctrica nacional.

Este proyecto nace de la necesidad de dotar a los operadores de sistemas fotovoltaicos 
de una herramienta predictiva que les permita anticipar, con una hora de anticipación, 
cuánta energía solar llegará a la superficie en Santo Domingo. Para lograrlo, se 
desarrolló un modelo de red neuronal recurrente tipo LSTM (Long Short-Term Memory), 
entrenado con seis años de datos meteorológicos horarios obtenidos de la API NASA POWER 
— una de las fuentes de datos climáticos más confiables y completas disponibles 
gratuitamente.

El modelo recibe como entrada las últimas 24 horas de cuatro variables meteorológicas 
— irradiancia solar, temperatura, humedad relativa y velocidad del viento — y produce 
como salida la irradiancia esperada para la hora siguiente. Con un coeficiente de 
determinación R² de 0.9924, el modelo demuestra una capacidad predictiva excepcional, 
explicando más del 99% de la variabilidad en la generación solar.

Desde una perspectiva económica, la predicción precisa de generación solar tiene 
implicaciones directas en la rentabilidad de los sistemas fotovoltaicos. Permite 
optimizar la compra y venta de energía en mercados eléctricos, gestionar eficientemente 
sistemas de almacenamiento por baterías y reducir la dependencia de fuentes de 
generación convencionales durante horas pico. En un país donde el costo de la 
electricidad representa una carga significativa para hogares y empresas, este tipo 
de herramientas tiene el potencial de generar un impacto económico y social concreto.

El análisis de errores reveló patrones interpretables y coherentes con la realidad 
climática dominicana: el mayor error ocurre a las 13:00h, cuando la irradiancia 
alcanza su máximo y la variabilidad por nubosidad es más alta, y en el mes de 
septiembre, coincidiendo con el pico de la temporada ciclónica en el Caribe. Estos 
hallazgos no solo validan el modelo, sino que abren líneas de mejora futuras, como 
la incorporación de índices de nubosidad o datos de satélite en tiempo real.

Este proyecto fue desarrollado íntegramente en Azure ML Notebooks como parte de un 
portafolio profesional en Ingeniería Electrónica, con enfoque en Machine Learning 
aplicado a sistemas de energía renovable.

## 🎯 Objetivo

Predecir la irradiancia solar (kWh/m²) con una hora de anticipación en Santo Domingo,
República Dominicana, utilizando datos meteorológicos históricos de 6 años y redes
neuronales LSTM (Long Short-Term Memory).

---

## 💡 Contexto y Motivación

República Dominicana enfrenta históricamente un déficit energético significativo.
La transición hacia energías renovables, especialmente la solar, representa una
oportunidad estratégica para el país. Sin embargo, la naturaleza intermitente de
la energía solar dificulta su integración eficiente en la red eléctrica.

Este proyecto desarrolla un modelo de predicción que permite a operadores de plantas
solares anticipar la generación de energía con una hora de anticipación, facilitando:

- ✅ Mejor planificación del consumo energético
- ✅ Optimización de la compra/venta de energía en la red
- ✅ Gestión eficiente de sistemas de almacenamiento (baterías)
- ✅ Reducción de desperdicio energético

---

## 📊 Resultados del Modelo

| Métrica | Valor |
|---------|-------|
| **R²**  | **0.9924** 🏆 |
| RMSE    | 26.48 kWh/m² |
| MAE     | 13.17 kWh/m² |
| MAPE    | 13.05% (horas diurnas) |

### 🔍 Hallazgos Clave
- **Hora con mayor error:** 13:00h — variabilidad máxima por nubosidad en el mediodía
- **Mes con mayor error:** Septiembre — pico de temporada ciclónica en el Caribe

---

## 🗂️ Estructura del Proyecto
solar-energy-prediction/

│

├── 📓 Prediccion_Solar_LSTM.ipynb   # Notebook principal (Fases 1-4)

├── 🌐 appLSTM.py                    # Dashboard Streamlit

├── 📄 requirements.txt              # Dependencias

├── 📄 README.md                     # Este archivo

│

├── 📁 data/

│   └── santo_domingo_solar_2020_2026.csv

│

└── 📁 models/

├── mejor_modelo_solar.keras

└── scaler_solar.pkl
---

## 🔧 Pipeline del Proyecto
NASA POWER API

↓

Descarga de datos horarios (54,768 registros)

↓

EDA — Análisis Exploratorio

↓

Preprocesamiento

• Imputación (interpolación lineal)

• Normalización (MinMaxScaler)

• Sliding Window (24h → 1h)

↓

Modelado LSTM

• LSTM(128) → LSTM(64) → Dense(32) → Dense(1)

• Early Stopping + ReduceLROnPlateau

↓

Evaluación (R² = 0.9924)
↓

Dashboard Streamlit
---

## 📍 Datos

| Parámetro | Valor |
|-----------|-------|
| Fuente | NASA POWER API |
| Ubicación | Santo Domingo, R.D. (18.49°N, 69.93°W) |
| Período | Enero 2020 – Marzo 2026 |
| Frecuencia | Horaria |
| Total registros | 54,768 |

**Variables utilizadas:**
| Variable | Código NASA | Unidad |
|----------|-------------|--------|
| Irradiancia solar | ALLSKY_SFC_SW_DWN | kWh/m² |
| Temperatura | T2M | °C |
| Humedad relativa | RH2M | % |
| Velocidad del viento | WS2M | m/s |

---

## 🧠 Arquitectura LSTM
Input (24h × 4 variables)

↓

LSTM(128) + BatchNorm + Dropout(0.2)

↓

LSTM(64)  + BatchNorm + Dropout(0.2)

↓

Dense(32, ReLU) + Dropout(0.1)

↓

Dense(1, Linear) → Predicción
| Hiperparámetro | Valor |
|----------------|-------|
| Optimizador | Adam (lr=0.001) |
| Loss | MSE |
| Batch size | 64 |
| Épocas máx. | 100 |
| Early Stopping | Patience=10 |

---

## 🛠️ Stack Tecnológico

- **Lenguaje:** Python 3.10
- **Modelado:** TensorFlow / Keras
- **Procesamiento:** Pandas, NumPy, scikit-learn
- **Visualización:** Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Entorno:** Azure ML Notebooks
- **Datos:** NASA POWER API

---

## 🚀 Ejecutar el Dashboard

```bash
# Clonar repositorio
git clone https://github.com/yoh6ly/solar-energy-prediction.git
cd solar-energy-prediction

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar dashboard
streamlit run appLSTM.py
```

---

## 👤 Autor

**Yohaly** — Ingeniería Electrónica  
Universidad O&M, Santo Domingo, República Dominicana

[![GitHub](https://img.shields.io/badge/GitHub-yoh6ly-181717?logo=github)](https://github.com/yoh6ly)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-yoh6ly-0A66C2?logo=linkedin)](https://linkedin.com/in/yoh6ly)

---

## 📄 Licencia

MIT License — libre para usar y modificar con atribución.
