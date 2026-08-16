"""
pipeline/preprocess.py
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["studytime", "failures", "absences", "G1", "G2"]
TARGET_COLUMN = "riesgo"


def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")
    
    df = pd.read_csv(ruta_csv, sep=";", skipinitialspace=True)
    df.columns = df.columns.str.replace('"', '').str.strip()
    return df


def crear_variable_riesgo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df["G3"].dtype == object:
        df["G3"] = df["G3"].astype(str).str.replace('"', '').astype(float)
    
    df[TARGET_COLUMN] = np.where(df["G3"] < 10, 1, 0)
    return df


def seleccionar_variables(df: pd.DataFrame) -> pd.DataFrame:
    for col in FEATURE_COLUMNS:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace('"', '').astype(float)
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().dropna().reset_index(drop=True)


def dividir_y_normalizar(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
    outputs_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(outputs_dir, "scaler.pkl"))

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def preparar_pipeline_completo(ruta_csv: str, test_size: float = 0.2, random_state: int = 42):
    df_crudo = cargar_datos(ruta_csv)
    df_limpio_completo = limpiar_datos(df_crudo)
    df_con_riesgo = crear_variable_riesgo(df_limpio_completo)
    df_limpio = seleccionar_variables(df_con_riesgo)

    X_train, X_test, y_train, y_test, scaler = dividir_y_normalizar(
        df_limpio, test_size=test_size, random_state=random_state
    )

    return {
        "df_limpio": df_limpio,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }