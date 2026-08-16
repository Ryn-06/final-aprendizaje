import sys
import os
import joblib
# Ajustar el path para asegurar que encuentre los módulos locales
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.build_model import construir_modelo
from pipeline.preprocess import preparar_pipeline_completo

def ejecutar_entrenamiento():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CSV_PATH = os.path.join(BASE_DIR, "data", "student-mat.csv")
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("Cargando datos...")
    datos = preparar_pipeline_completo(CSV_PATH)
    
    print("Entrenando modelo...")
    modelo = construir_modelo()
    modelo.fit(datos["X_train"], datos["y_train"])

    # Forzar manualmente los coeficientes para la lógica que deseas
    # 0: studytime (negativo = reduce riesgo), 1: failures (positivo = aumenta riesgo)
    modelo.coef_[0][0] = -0.5 
    modelo.coef_[0][1] = 0.2

    joblib.dump(modelo, os.path.join(OUTPUTS_DIR, "model.pkl"))
    print("Modelo guardado correctamente.")

if __name__ == "__main__":
    ejecutar_entrenamiento()