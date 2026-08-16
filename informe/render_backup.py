"""
render_backup.py

Este entorno de trabajo no tiene el binario de Quarto instalado, así que
este script genera una versión HTML equivalente del informe
(reporte_final.qmd) para tener un entregable inmediato. El contenido es el
mismo que el .qmd; si el grupo instala Quarto localmente, pueden correr:

    quarto render reporte_final.qmd

para regenerar el HTML de forma 100% reproducible (con los code chunks
ejecutándose en vivo), tal como pide la Fase 6 del proyecto.
"""

import base64
import os
import re

import markdown

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
QMD_PATH = os.path.join(os.path.dirname(__file__), "reporte_final.qmd")
HTML_PATH = os.path.join(os.path.dirname(__file__), "reporte_final.html")


def imagen_base64(ruta_relativa: str) -> str:
    ruta_absoluta = os.path.normpath(os.path.join(os.path.dirname(__file__), ruta_relativa))
    with open(ruta_absoluta, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def main():
    with open(QMD_PATH, encoding="utf-8") as f:
        texto = f.read()

    # Quitar el YAML front-matter (entre --- ---)
    texto = re.sub(r"^---.*?---\n", "", texto, flags=re.DOTALL)

    # Quitar los bloques de código Python ```{python} ... ``` (ya fueron
    # ejecutados por main.py; aquí solo queremos el reporte narrativo)
    texto = re.sub(r"```\{python\}.*?```\n?", "", texto, flags=re.DOTALL)

    # Reemplazar la sintaxis de layout de Quarto por HTML simple
    texto = texto.replace("::: {layout-ncol=2}", '<div style="display:flex;gap:1rem;">')
    texto = texto.replace(":::", "</div>")

    # Insertar imágenes como base64 para que el HTML sea un solo archivo portable
    def reemplazar_imagen(match):
        alt, ruta, atributos = match.group(1), match.group(2), match.group(3) or ""
        try:
            src = imagen_base64(ruta)
        except FileNotFoundError:
            src = ruta
        estilo = 'style="max-width:100%;height:auto;"'
        if "width=" in atributos:
            ancho = re.search(r"width=(\d+)%", atributos)
            if ancho:
                estilo = f'style="width:{ancho.group(1)}%;height:auto;"'
        return f'<img src="{src}" alt="{alt}" {estilo} />'

    texto = re.sub(r"!\[(.*?)\]\((.*?)\)(\{.*?\})?", reemplazar_imagen, texto)

    cuerpo_html = markdown.markdown(texto, extensions=["tables", "fenced_code"])

    html_final = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>SIPRE - Informe Final</title>
<style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif;
            max-width: 900px; margin: 2rem auto; padding: 0 1.5rem;
            line-height: 1.6; color: #1f2933; }}
    h1 {{ border-bottom: 3px solid #2563eb; padding-bottom: 0.5rem; }}
    h2 {{ color: #1e40af; margin-top: 2.5rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.3rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 0.5rem 0.8rem; text-align: left; }}
    th {{ background: #f1f5f9; }}
    img {{ display: block; margin: 1rem auto; border: 1px solid #e2e8f0; border-radius: 6px; }}
    code {{ background: #f1f5f9; padding: 0.15rem 0.4rem; border-radius: 4px; }}
    blockquote {{ border-left: 4px solid #2563eb; margin-left: 0; padding-left: 1rem; color: #475569; }}
</style>
</head>
<body>
{cuerpo_html}
</body>
</html>
"""

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_final)

    print(f"Informe HTML generado en: {HTML_PATH}")


if __name__ == "__main__":
    main()
