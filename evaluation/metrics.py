"""
evaluation/metrics.py

Genera la Matriz de Confusión y el reporte de Precision, Recall y F1-Score.

En este proyecto, un Falso Negativo (decir que el estudiante está bien
cuando en realidad reprueba) es más grave que un Falso Positivo (decir
que está en riesgo cuando en realidad iba bien), porque el objetivo es
detectar a tiempo a los estudiantes que sí necesitan ayuda. Por eso el
Recall de la clase "riesgo" es la métrica más importante a vigilar.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluar_modelo(modelo, X_test, y_test, umbral: float = 0.5) -> dict:
    """Calcula las métricas del modelo sobre el set de prueba."""
    y_proba = modelo.predict(X_test, verbose=0).ravel()
    y_pred = (y_proba >= umbral).astype(int)

    matriz = confusion_matrix(y_test, y_pred)

    resultados = {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "matriz_confusion": matriz,
        "reporte_texto": classification_report(
            y_test, y_pred, target_names=["Bajo Riesgo (0)", "Riesgo Alto (1)"], zero_division=0
        ),
    }
    return resultados


def graficar_matriz_confusion(matriz, carpeta_salida: str) -> str:
    """Genera y guarda la imagen de la Matriz de Confusión."""
    os.makedirs(carpeta_salida, exist_ok=True)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz, display_labels=["Bajo Riesgo (0)", "Riesgo Alto (1)"]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Matriz de Confusión, proyecto SIPRE")
    plt.tight_layout()

    ruta = os.path.join(carpeta_salida, "matriz_confusion.png")
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

    resultados = evaluar_modelo(modelo, datos["X_test"], datos["y_test"])
    print("Accuracy :", round(resultados["accuracy"], 3))
    print("Precision:", round(resultados["precision"], 3))
    print("Recall   :", round(resultados["recall"], 3))
    print("F1-Score :", round(resultados["f1"], 3))
    print("Matriz de Confusión:")
    print(resultados["matriz_confusion"])
    print(resultados["reporte_texto"])

    graficar_matriz_confusion(resultados["matriz_confusion"], os.path.join(BASE_DIR, "outputs"))
