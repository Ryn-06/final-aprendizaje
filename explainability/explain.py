"""
explainability/explain.py

Extrae los pesos de la primera capa de la red para estimar qué tanto
influye cada variable de entrada en la predicción de riesgo.

Es una explicabilidad aproximada, no un método riguroso como SHAP o
LIME. La importancia de cada variable se calcula como la suma de los
valores absolutos de los pesos que la conectan a las 32 neuronas de la
primera capa oculta.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def calcular_importancia_variables(modelo, nombres_variables: list[str]) -> dict:
    """Extrae los pesos de la primera capa Dense y calcula la importancia
    relativa de cada variable de entrada."""
    primera_capa_densa = None
    for capa in modelo.layers:
        if capa.__class__.__name__ == "Dense":
            primera_capa_densa = capa
            break

    if primera_capa_densa is None:
        raise ValueError("No se encontró ninguna capa Dense en el modelo.")

    pesos, _bias = primera_capa_densa.get_weights()
    importancia_bruta = np.sum(np.abs(pesos), axis=1)
    importancia_normalizada = importancia_bruta / importancia_bruta.sum()

    return dict(zip(nombres_variables, importancia_normalizada))


def graficar_importancia(importancia: dict, carpeta_salida: str) -> str:
    """Genera un gráfico de barras con la importancia de cada variable."""
    os.makedirs(carpeta_salida, exist_ok=True)

    items_ordenados = sorted(importancia.items(), key=lambda x: x[1])
    nombres = [n for n, _ in items_ordenados]
    valores = [v for _, v in items_ordenados]

    plt.figure(figsize=(8, 5))
    barras = plt.barh(nombres, valores, color="#2563eb")
    for barra, valor in zip(barras, valores):
        plt.text(
            valor + 0.005,
            barra.get_y() + barra.get_height() / 2,
            f"{valor*100:.1f}%",
            va="center",
            fontsize=9,
        )
    plt.title("Importancia de variables, factores de riesgo estudiantil")
    plt.xlabel("Importancia relativa (pesos de la primera capa)")
    plt.tight_layout()

    ruta = os.path.join(carpeta_salida, "importancia_variables.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    return ruta


if __name__ == "__main__":
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from pipeline.preprocess import preparar_pipeline_completo
    from model.train import entrenar_modelo

    BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
    ruta_csv = os.path.join(BASE_DIR, "data", "student-mat.csv")

    datos = preparar_pipeline_completo(ruta_csv)
    modelo, historial = entrenar_modelo(
        datos["X_train"], datos["y_train"], datos["X_test"], datos["y_test"], verbose=0
    )

    importancia = calcular_importancia_variables(modelo, datos["feature_columns"])
    for nombre, valor in sorted(importancia.items(), key=lambda x: -x[1]):
        print(f"{nombre:12s}: {valor*100:5.1f}%")

    graficar_importancia(importancia, os.path.join(BASE_DIR, "outputs"))
