# SIPRE, Sistema Inteligente de Predicción de Riesgo Estudiantil

Proyecto final de Aprendizaje Automatizado (Prof. Ronald Ponce, ITSE).

## Cómo correr el proyecto

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Correr el pipeline completo (entrena, evalúa, genera gráficas)
python main.py
# usa --reentrenar si ya existe outputs/model.h5 y quieres entrenar de nuevo
# usa --epochs N para cambiar el número de épocas

# 3. Levantar la app interactiva (en otra terminal, con el modelo ya entrenado)
streamlit run app/app.py

# 4. Generar el informe final
#    Si tienes Quarto instalado (https://quarto.org/docs/get-started/):
quarto render informe/reporte_final.qmd
#    Si no, ya existe un HTML equivalente generado en:
informe/reporte_final.html
```

Todos los outputs (gráficas y el modelo `model.h5`) quedan en `outputs/`.

## Estructura

```
AI_Aprendizaje_Auto_L6/
├── data/student-mat.csv          # Dataset UCI (Student Performance - Math)
├── pipeline/preprocess.py        # Fase 1: carga, limpieza, riesgo, normalización
├── model/build_model.py          # Fase 2: arquitectura MLP
├── model/train.py                # Fase 2: entrenamiento + gráficas
├── evaluation/metrics.py         # Fase 3: matriz de confusión, precision/recall/f1
├── explainability/explain.py     # Fase 3: importancia de variables
├── app/app.py                    # Fase 4: app Streamlit
├── main.py                       # Fase 5: orquestador
├── informe/reporte_final.qmd     # Fase 6: informe fuente (Quarto)
├── informe/reporte_final.html    # Fase 6: informe ya renderizado (respaldo)
├── outputs/                      # Gráficas + modelo entrenado (se genera al correr main.py)
└── requirements.txt
```

## Resultados obtenidos (última corrida)

| Métrica | Valor |
|---|---|
| Accuracy | 0.91 |
| Precision (Riesgo Alto) | 0.85 |
| Recall (Riesgo Alto) | 0.88 |
| F1-Score (Riesgo Alto) | 0.87 |

Variable más influyente: **G2** (nota del segundo parcial, 32.8% de importancia).

## Sugerencia de reparto para 7 integrantes

Aunque el proyecto se corre completo con `main.py`, cada módulo es
independiente y se puede repartir así para la presentación/documentación:

1. **Persona 1**, `pipeline/preprocess.py` (Fase 1: datos)
2. **Persona 2**, `model/build_model.py` + `model/train.py` (Fase 2: arquitectura y entrenamiento)
3. **Persona 3**, `evaluation/metrics.py` (Fase 3a: métricas y matriz de confusión)
4. **Persona 4**, `explainability/explain.py` (Fase 3b: explicabilidad)
5. **Persona 5**, `app/app.py` (Fase 4: app Streamlit)
6. **Persona 6**, `main.py` + integración/pruebas de todo el pipeline (Fase 5)
7. **Persona 7**, `informe/reporte_final.qmd` (Fase 6: informe, y preparar la exposición/discusión en clase sobre Falso Positivo vs Falso Negativo)

Fuente del dataset: Cortez, P., & Silva, A. (2008). *Student Performance*.
UCI Machine Learning Repository.
