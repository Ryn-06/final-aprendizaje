"""
main.py

Fase 5: Orquestación.

Archivo maestro del proyecto SIPRE. Al ejecutar:

    python main.py

el sistema:
    1. Carga y limpia los datos (pipeline/preprocess.py).
    2. Entrena la red neuronal, o la carga si ya existe un modelo guardado
       (model/build_model.py + model/train.py).
    3. Muestra/guarda las gráficas de rendimiento (Accuracy y Loss).
    4. Ejecuta la evaluación de métricas (evaluation/metrics.py).
    5. Muestra la importancia de las variables (explainability/explain.py).

Todas las gráficas e imágenes generadas se guardan en la carpeta `outputs/`.
La app interactiva (app/app.py) se corre por separado con:

    streamlit run app/app.py
"""

from __future__ import annotations

import argparse
import os

import tensorflow as tf

from pipeline.preprocess import preparar_pipeline_completo
from model.train import entrenar_modelo, graficar_entrenamiento, guardar_modelo
from evaluation.metrics import evaluar_modelo, graficar_matriz_confusion
from explainability.explain import calcular_importancia_variables, graficar_importancia

BASE_DIR = os.path.dirname(__file__)
RUTA_CSV = os.path.join(BASE_DIR, "data", "student-mat.csv")
CARPETA_OUTPUTS = os.path.join(BASE_DIR, "outputs")
RUTA_MODELO = os.path.join(CARPETA_OUTPUTS, "model.h5")


def linea(titulo: str) -> None:
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)


def main(forzar_reentrenamiento: bool = False, epochs: int = 50):
    # ------------------------------------------------------------------
    # 1. Carga y limpieza de datos
    # ------------------------------------------------------------------
    linea("FASE 1: Cargando y preparando los datos")
    datos = preparar_pipeline_completo(RUTA_CSV)
    print(f"Estudiantes tras limpieza: {len(datos['df_limpio'])}")
    print(f"Variables de entrada: {datos['feature_columns']}")
    print(f"Train: {datos['X_train'].shape} | Test: {datos['X_test'].shape}")

    # ------------------------------------------------------------------
    # 2. Entrenar (o cargar) la red neuronal
    # ------------------------------------------------------------------
    linea("FASE 2: Entrenando la red neuronal (o cargando modelo existente)")
    if os.path.exists(RUTA_MODELO) and not forzar_reentrenamiento:
        print(f"Se encontró un modelo guardado en {RUTA_MODELO}. Cargando...")
        modelo = tf.keras.models.load_model(RUTA_MODELO)
        historial = None
    else:
        modelo, historial = entrenar_modelo(
            datos["X_train"], datos["y_train"], datos["X_test"], datos["y_test"], epochs=epochs
        )
        guardar_modelo(modelo, RUTA_MODELO)
        print(f"Modelo entrenado y guardado en {RUTA_MODELO}")

    # ------------------------------------------------------------------
    # 3. Gráficas de rendimiento
    # ------------------------------------------------------------------
    if historial is not None:
        linea("FASE 2b: Generando gráficas de Accuracy y Loss")
        ruta_acc, ruta_loss = graficar_entrenamiento(historial, CARPETA_OUTPUTS)
        print(f"Guardadas: {ruta_acc}, {ruta_loss}")
    else:
        print("\n(Modelo cargado desde disco: no hay historial nuevo que graficar. "
              "Usa --reentrenar para forzar un nuevo entrenamiento y regenerar las gráficas.)")

    # ------------------------------------------------------------------
    # 4. Evaluación de métricas
    # ------------------------------------------------------------------
    linea("FASE 3: Evaluando el modelo")
    resultados = evaluar_modelo(modelo, datos["X_test"], datos["y_test"])
    print(f"Accuracy : {resultados['accuracy']:.3f}")
    print(f"Precision: {resultados['precision']:.3f}")
    print(f"Recall   : {resultados['recall']:.3f}")
    print(f"F1-Score : {resultados['f1']:.3f}")
    print("\nMatriz de Confusión:\n", resultados["matriz_confusion"])
    print("\n" + resultados["reporte_texto"])
    ruta_matriz = graficar_matriz_confusion(resultados["matriz_confusion"], CARPETA_OUTPUTS)
    print(f"Guardada: {ruta_matriz}")

    # ------------------------------------------------------------------
    # 5. Explicabilidad: importancia de variables
    # ------------------------------------------------------------------
    linea("FASE 3b: Calculando importancia de variables")
    importancia = calcular_importancia_variables(modelo, datos["feature_columns"])
    for nombre, valor in sorted(importancia.items(), key=lambda x: -x[1]):
        print(f"  {nombre:12s}: {valor*100:5.1f}%")
    ruta_importancia = graficar_importancia(importancia, CARPETA_OUTPUTS)
    print(f"Guardada: {ruta_importancia}")

    linea("SIPRE: proceso completo. Corre 'streamlit run app/app.py' para la app interactiva.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orquestador del proyecto SIPRE")
    parser.add_argument(
        "--reentrenar",
        action="store_true",
        help="Fuerza un nuevo entrenamiento aunque ya exista outputs/model.h5",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas de entrenamiento")
    args = parser.parse_args()

    main(forzar_reentrenamiento=args.reentrenar, epochs=args.epochs)
