"""
app/app.py

App interactiva con Streamlit para la predicción de riesgo académico (SIPRE).
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "outputs", "scaler.pkl")
CSV_PATH = os.path.join(BASE_DIR, "data", "student-mat.csv")

FEATURE_COLUMNS = ["studytime", "failures", "absences", "G1", "G2"]

st.set_page_config(
    page_title="SIPRE - Predicción Académica",
    page_icon="🎓",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(124, 58, 237, 0.30),
                transparent 35%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(37, 99, 235, 0.28),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #080b1a,
                #11152d,
                #17113a
            );
    }

    .main .block-container {
        max-width: 950px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(
            90deg,
            #c084fc,
            #818cf8,
            #60a5fa
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }

    h2, h3 {
        color: #f8fafc !important;
    }

    p, label {
        color: #dbe4f0 !important;
    }

    .subtitle {
        text-align: center;
        color: #aebbd0;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            rgba(124, 58, 237, 0.20),
            rgba(37, 99, 235, 0.12)
        );
        border: 1px solid rgba(139, 92, 246, 0.25);
        padding: 1rem;
        border-radius: 18px;
    }

    div[data-testid="stMetricLabel"] {
        color: #aebbd0 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 750;
    }

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 14px;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        font-weight: 700;
        color: white;
        background: linear-gradient(
            90deg,
            #7c3aed,
            #4f46e5,
            #2563eb
        );
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.35);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
    }

    .footer {
        text-align: center;
        color: #718096;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def cargar_modelo_y_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    modelo = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return modelo, scaler

@st.cache_data
def cargar_dataset():
    if not os.path.exists(CSV_PATH):
        return None
    return pd.read_csv(CSV_PATH, sep=";")

@st.cache_data
def perfil_del_dataset():
    df = cargar_dataset()
    if df is None:
        return None

    numericos = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    alfanumericos = df.select_dtypes(include=["object"]).columns.tolist()

    return {
        "registros": len(df),
        "campos_totales": len(df.columns),
        "campos_numericos": len(numericos),
        "campos_alfanumericos": len(alfanumericos),
        "nulos": int(df.isnull().sum().sum()),
        "duplicados": int(df.duplicated().sum()),
        "lista_numericos": numericos,
        "lista_alfanumericos": alfanumericos
    }

def obtener_probabilidad(modelo, scaler, entrada):
    entrada_arr = np.array([entrada])
    entrada_escalada = scaler.transform(entrada_arr)

    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(entrada_escalada)[0]
        if 1 in modelo.classes_:
            indice_riesgo = list(modelo.classes_).index(1)
            return float(probabilidades[indice_riesgo])
        return float(probabilidades[-1])

    prediccion = modelo.predict(entrada_escalada)[0]
    return float(prediccion)

def pagina_prediccion(modelo, scaler):
    st.title("🎓 SIPRE")
    st.markdown(
        """
        <p class="subtitle">
        Sistema Inteligente de Predicción de Riesgo Estudiantil
        <br>
        Analiza indicadores académicos para estimar el riesgo de bajo rendimiento.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.subheader("📋 Perfil del estudiante")
    col1, col2 = st.columns(2)

    with col1:
        studytime = st.slider(
            "Tiempo de estudio semanal", 1, 4, 2,
            help="1: <2h | 2: 2-5h | 3: 5-10h | 4: >10h"
        )
        failures = st.slider("Materias reprobadas previamente", 0, 4, 0)
        absences = st.slider("Ausencias acumuladas", 0, 93, 4)

    with col2:
        g1 = st.slider("Nota primer periodo (G1)", 0, 20, 12)
        g2 = st.slider("Nota segundo periodo (G2)", 0, 20, 12)

    st.divider()

    # Usar session_state para mantener la predicción activa
    if "prediccion_activa" not in st.session_state:
        st.session_state.prediccion_activa = False

    if st.button("🔍 Predecir riesgo", type="primary", use_container_width=True):
        st.session_state.prediccion_activa = True
        st.session_state.entrada_actual = [studytime, failures, absences, g1, g2]

    if st.session_state.prediccion_activa:
        entrada = st.session_state.entrada_actual
        probabilidad = obtener_probabilidad(modelo, scaler, entrada)
        probabilidad = min(max(probabilidad, 0.0), 1.0)
        en_riesgo = probabilidad >= 0.5

        st.subheader("Resultado")

        if en_riesgo:
            st.error(
                f"🔴 **Riesgo académico detectado**\n\n"
                f"Probabilidad de riesgo: **{probabilidad:.1%}**\n\n"
                "Los indicadores ingresados sugieren una probabilidad elevada de riesgo académico."
            )
        else:
            st.success(
                f"🟢 **Estudiante con riesgo bajo**\n\n"
                f"Probabilidad de riesgo: **{probabilidad:.1%}**\n\n"
                "Los indicadores ingresados se encuentran asociados con una menor probabilidad de riesgo."
            )

        st.progress(probabilidad)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probabilidad de riesgo", f"{probabilidad:.1%}")
        with col2:
            st.metric("Probabilidad de no riesgo", f"{1 - probabilidad:.1%}")

        st.markdown("### 📋 Datos ingresados")
        datos = pd.DataFrame({"Variable": FEATURE_COLUMNS, "Valor": entrada})
        st.dataframe(datos, use_container_width=True, hide_index=True)

        st.markdown("### 📈 Análisis de sensibilidad")
        variable_a_variar = st.selectbox(
            "Selecciona una variable para observar cómo cambia la predicción:",
            FEATURE_COLUMNS,
            index=4
        )

        rangos = {
            "studytime": list(range(1, 5)),
            "failures": list(range(0, 5)),
            "absences": list(range(0, 41, 2)),
            "G1": list(range(0, 21)),
            "G2": list(range(0, 21))
        }

        valores_actuales = {
            "studytime": entrada[0],
            "failures": entrada[1],
            "absences": entrada[2],
            "G1": entrada[3],
            "G2": entrada[4]
        }

        resultados = []
        for valor in rangos[variable_a_variar]:
            fila = valores_actuales.copy()
            fila[variable_a_variar] = valor
            entrada_variada = [fila["studytime"], fila["failures"], fila["absences"], fila["G1"], fila["G2"]]
            probabilidad_variada = obtener_probabilidad(modelo, scaler, entrada_variada)
            resultados.append({variable_a_variar: valor, "Riesgo": probabilidad_variada * 100})

        df_sensibilidad = pd.DataFrame(resultados)
        st.line_chart(df_sensibilidad, x=variable_a_variar, y="Riesgo")
        st.caption(f"Cada punto representa una predicción nueva variando únicamente {variable_a_variar}.")

    with st.expander("ℹ️ Acerca de este modelo"):
        st.write("SIPRE utiliza un modelo de aprendizaje automático entrenado para estimar el riesgo académico.")

def pagina_dashboard():
    st.title("📊 Datos del proyecto")
    st.markdown('<p class="subtitle">Resumen del dataset utilizado para desarrollar SIPRE.</p>', unsafe_allow_html=True)

    perfil = perfil_del_dataset()
    if perfil is None:
        st.warning("No se encontró data/student-mat.csv.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Registros", perfil["registros"])
    col2.metric("Campos", perfil["campos_totales"])
    col3.metric("Valores nulos", perfil["nulos"])

    col4, col5, col6 = st.columns(3)
    col4.metric("Numéricos", perfil["campos_numericos"])
    col5.metric("Categóricos", perfil["campos_alfanumericos"])
    col6.metric("Duplicados", perfil["duplicados"])

    st.markdown("### 🔢 Campos numéricos")
    st.write(", ".join(perfil["lista_numericos"]))

    st.markdown("### 🔤 Campos alfanuméricos")
    st.write(", ".join(perfil["lista_alfanumericos"]))

    st.markdown("### 🎯 Variables utilizadas por SIPRE")
    st.write(", ".join(FEATURE_COLUMNS))

def main():
    modelo, scaler = cargar_modelo_y_scaler()

    if modelo is None or scaler is None:
        st.error("No se encontró el modelo o escalador entrenado.")
        st.info("Ejecuta `python model/train.py` para regenerar `model.pkl` y `scaler.pkl` en `outputs/`.")
        st.stop()

    tab1, tab2 = st.tabs(["🔍 Predicción", "📊 Datos del proyecto"])

    with tab1:
        pagina_prediccion(modelo, scaler)

    with tab2:
        pagina_dashboard()

    st.markdown('<div class="footer">SIPRE · Sistema de Predicción de Riesgo Estudiantil</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()