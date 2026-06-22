# Dashboard Streamlit
# Proyecto: Predicción de Generación Solar con LSTM
# Ubicación: Santo Domingo, República Dominicana
# Archivo: app.LSTMpy

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime

# Configuración de página 
st.set_page_config(
    page_title="Predicción Solar — Santo Domingo",
    page_icon="🌞",
    layout="wide"
)

#  Estilos 
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #F5A623;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #F5A623;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 4px;
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #F5A623;
    }
</style>
""", unsafe_allow_html=True)

# Cargar modelo y scaler
@st.cache_resource
def cargar_modelo():
    model  = load_model("mejor_modelo_solar.keras")
    scaler = joblib.load("scaler_solar.pkl")
    return model, scaler

@st.cache_data
def cargar_datos():
    df = pd.read_csv(
        "santo_domingo_solar_2020_2026.csv",
        index_col="datetime",
        parse_dates=True
    )
    df.index = pd.to_datetime(df.index)
    df = df.interpolate(method="linear")
    return df

model, scaler = cargar_modelo()
df = cargar_datos()

# Header 
st.markdown('<p class="header-title">🌞 Predicción de Generación Solar</p>', unsafe_allow_html=True)
st.markdown("**Ubicación:** Santo Domingo, República Dominicana &nbsp;|&nbsp; **Modelo:** LSTM &nbsp;|&nbsp; **Datos:** NASA POWER API")
st.divider()

# Métricas globales 
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">0.9924</div>
        <div class="metric-label">R² Score</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">26.48</div>
        <div class="metric-label">RMSE (kWh/m²)</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">13.17</div>
        <div class="metric-label">MAE (kWh/m²)</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">13.05%</div>
        <div class="metric-label">MAPE (diurno)</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# Sidebar — controles 
st.sidebar.header("⚙️ Controles")

# Selector de fecha para predicción
fechas_disponibles = df.index.normalize().unique()
fecha_sel = st.sidebar.date_input(
    "Selecciona una fecha",
    value=fechas_disponibles[-2].date(),
    min_value=fechas_disponibles[1].date(),
    max_value=fechas_disponibles[-1].date()
)

# Selector de rango para serie histórica
st.sidebar.markdown("---")
st.sidebar.markdown("**Serie histórica**")
meses_opciones = {
    "Último mes"    : 30,
    "Últimos 3 meses": 90,
    "Últimos 6 meses": 180,
    "Último año"    : 365
}
rango_sel = st.sidebar.selectbox("Rango de visualización", list(meses_opciones.keys()))

# Sección 1: Predicción del día seleccionado 
st.subheader(f"📅 Predicción Horaria — {fecha_sel.strftime('%d %b %Y')}")

fecha_dt = pd.Timestamp(fecha_sel)

# Obtener las 24h previas como input del modelo
try:
    idx_inicio = df.index.get_loc(fecha_dt, method="nearest") - 24
    if idx_inicio < 0:
        st.warning("No hay suficientes datos previos para esta fecha.")
    else:
        ventana = df.iloc[idx_inicio : idx_inicio + 24][
            ["irradiancia_kwh_m2", "temperatura_c", "humedad_pct", "viento_m_s"]
        ].values

        ventana_scaled = scaler.transform(ventana)
        X_input = ventana_scaled.reshape(1, 24, 4)
        pred_scaled = model.predict(X_input, verbose=0)[0][0]

        # Invertir escala
        dummy = np.zeros((1, 4))
        dummy[0, 0] = pred_scaled
        pred_real = scaler.inverse_transform(dummy)[0][0]

        # Datos reales del día para comparar
        dia_real = df[df.index.date == fecha_sel]["irradiancia_kwh_m2"]

        fig, ax = plt.subplots(figsize=(12, 4))
        if len(dia_real) > 0:
            ax.plot(dia_real.index.hour, dia_real.values,
                    label="Real", color="#F5A623", linewidth=2, marker="o", markersize=4)
        ax.axhline(pred_real, color="#E74C3C", linestyle="--", linewidth=2,
                   label=f"Predicción próxima hora: {pred_real:.2f} kWh/m²")
        ax.set_xlabel("Hora del día")
        ax.set_ylabel("Irradiancia (kWh/m²)")
        ax.set_xticks(range(0, 24))
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.success(f"⚡ Predicción para la próxima hora: **{pred_real:.2f} kWh/m²**")

except Exception as e:
    st.error(f"Error al generar predicción: {e}")

st.divider()

# Sección 2: Serie histórica 
st.subheader(f"📈 Serie Histórica de Irradiancia — {rango_sel}")

dias = meses_opciones[rango_sel]
df_rango = df.last(f"{dias}D")

fig2, ax2 = plt.subplots(figsize=(14, 4))
df_rango["irradiancia_kwh_m2"].resample("D").mean().plot(
    ax=ax2, color="#F5A623", linewidth=1.2
)
ax2.set_ylabel("Irradiancia promedio diaria (kWh/m²)")
ax2.set_xlabel("")
ax2.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.divider()

# Sección 3: Perfil solar promedio por hora 
st.subheader("⏰ Perfil Solar Promedio por Hora del Día")

col_a, col_b = st.columns(2)

with col_a:
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    irr_hora = df.groupby(df.index.hour)["irradiancia_kwh_m2"].mean()
    ax3.bar(irr_hora.index, irr_hora.values, color="#F5A623", edgecolor="white")
    ax3.set_title("Irradiancia promedio por hora")
    ax3.set_xlabel("Hora")
    ax3.set_ylabel("kWh/m²")
    ax3.set_xticks(range(0, 24))
    ax3.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

with col_b:
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    irr_mes = df.groupby(df.index.month)["irradiancia_kwh_m2"].mean()
    meses_labels = ["Ene","Feb","Mar","Abr","May","Jun",
                    "Jul","Ago","Sep","Oct","Nov","Dic"]
    ax4.bar(range(1, 13), irr_mes.values, color="#E67E22", edgecolor="white")
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels(meses_labels)
    ax4.set_title("Irradiancia promedio por mes")
    ax4.set_xlabel("Mes")
    ax4.set_ylabel("kWh/m²")
    ax4.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

st.divider()

# Footer 
st.markdown("""
<div style='text-align:center; color:#666; font-size:0.8rem; padding:10px'>
    Proyecto de Portafolio — Ingeniería Electrónica | Universidad O&M, Santo Domingo, R.D.<br>
    Datos: NASA POWER API | Modelo: LSTM (TensorFlow/Keras) | R² = 0.9924
</div>
""", unsafe_allow_html=True)
