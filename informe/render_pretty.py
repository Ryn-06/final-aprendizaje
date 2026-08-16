"""
render_pretty.py

Genera una versión visualmente cuidada del informe SIPRE (reporte_final.qmd)
como un único HTML autocontenido (imágenes embebidas en base64).

Dirección de diseño ("dossier técnico de riesgo académico"):
  - Paleta: papel marfil + tinta azul-pizarra + rojo "riesgo" / verde "estable"
    como acentos semánticos (coherentes con el tema del proyecto: un modelo
    que clasifica entre esos dos estados).
  - Tipografía: serif editorial para títulos (Source Serif 4), sans para
    cuerpo (Inter), mono para datos/métricas (IBM Plex Mono).
  - Elemento de firma: un velocímetro/gauge semicircular en el hero que
    muestra el Accuracy del modelo, un dial que va de "riesgo" a "estable",
    igual que la predicción que hace el propio sistema.
  - Los marcadores numerados (01–09) sí tienen sentido aquí: el informe es
    literalmente una secuencia de 9 fases pedidas por la rúbrica.
"""

import base64
import os
import re

import markdown

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
QMD_PATH = os.path.join(os.path.dirname(__file__), "reporte_final.qmd")
HTML_PATH = os.path.join(os.path.dirname(__file__), "reporte_final.html")

# Métricas conocidas de la última corrida (ver outputs/ y consola de main.py)
METRICS = {
    "accuracy": 0.911,
    "precision": 0.852,
    "recall": 0.885,
    "f1": 0.868,
    "n_estudiantes": 395,
    "n_riesgo": 130,
    "n_bajo_riesgo": 265,
}

SECTION_TITLES = [
    "Planteamiento del problema",
    "Análisis exploratorio de datos (EDA)",
    "Preparación y limpieza de datos",
    "Ingeniería de características (Feature Engineering)",
    "Entrenamiento del modelo",
    "Evaluación y métricas",
    "Interpretación de resultados (Explicabilidad)",
    "Conclusiones y recomendaciones",
    "Trabajo futuro",
]


def imagen_base64(ruta_relativa: str) -> str:
    ruta_absoluta = os.path.normpath(os.path.join(os.path.dirname(__file__), ruta_relativa))
    with open(ruta_absoluta, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def gauge_svg(valor: float, size: int = 220) -> str:
    """Dial semicircular 'riesgo -> estable' con aguja en el valor dado (0-1)."""
    import math

    cx, cy, r = size / 2, size / 2 + 6, size / 2 - 18
    ang = math.pi * (1 - valor)  # 0 -> derecha (estable), 1 -> izquierda... invertimos
    ang = math.pi - math.pi * valor
    x2 = cx + r * 0.78 * math.cos(ang)
    y2 = cy - r * 0.78 * math.sin(ang)

    # Arco de fondo dividido en 3 tramos de color
    def punto(frac):
        a = math.pi - math.pi * frac
        return cx + r * math.cos(a), cy - r * math.sin(a)

    p0 = punto(0.0)
    p33 = punto(0.5)
    p100 = punto(1.0)

    return f"""
    <svg viewBox="0 0 {size} {size*0.82:.0f}" width="{size}" height="{size*0.82:.0f}" class="gauge">
      <path d="M {p0[0]:.1f} {p0[1]:.1f} A {r} {r} 0 0 1 {p33[0]:.1f} {p33[1]:.1f}"
            fill="none" stroke="var(--risk)" stroke-width="14" stroke-linecap="round" opacity="0.85"/>
      <path d="M {p33[0]:.1f} {p33[1]:.1f} A {r} {r} 0 0 1 {p100[0]:.1f} {p100[1]:.1f}"
            fill="none" stroke="var(--stable)" stroke-width="14" stroke-linecap="round" opacity="0.85"/>
      <line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="var(--ink)" stroke-width="3" stroke-linecap="round"/>
      <circle cx="{cx}" cy="{cy}" r="6" fill="var(--ink)"/>
      <text x="{cx}" y="{cy+34}" text-anchor="middle" class="gauge-label">{valor*100:.1f}%</text>
      <text x="{cx}" y="{cy+52}" text-anchor="middle" class="gauge-caption">Accuracy del modelo</text>
    </svg>
    """


def build_sections_html(texto: str):
    """Parte el markdown del informe en secciones por h2 y devuelve HTML por sección."""
    texto = re.sub(r"^---.*?---\n", "", texto, flags=re.DOTALL)
    texto = re.sub(r"```\{python\}.*?```\n?", "", texto, flags=re.DOTALL)
    texto = texto.replace("::: {layout-ncol=2}", '<div class="img-row">')
    texto = texto.replace(":::", "</div>")

    def reemplazar_imagen(match):
        alt, ruta, atributos = match.group(1), match.group(2), match.group(3) or ""
        try:
            src = imagen_base64(ruta)
        except FileNotFoundError:
            src = ruta
        ancho = ""
        m = re.search(r"width=(\d+)%", atributos or "")
        if m:
            ancho = f' style="width:{m.group(1)}%;"'
        return f'<figure class="fig"><img src="{src}" alt="{alt}"{ancho}/><figcaption>{alt}</figcaption></figure>'

    texto = re.sub(r"!\[(.*?)\]\((.*?)\)(\{.*?\})?", reemplazar_imagen, texto)

    # Cortar el pie de página final (línea "---" + fuente de datos)
    texto = re.split(r"\n---\n\n\*Fuente de datos", texto)[0]

    partes = re.split(r"\n## ", texto)
    secciones = []
    for parte in partes[1:]:
        titulo_linea, _, resto = parte.partition("\n")
        titulo = re.sub(r"^\d+\.\s*", "", titulo_linea).strip()
        cuerpo_html = markdown.markdown(resto.strip(), extensions=["tables", "fenced_code"])
        secciones.append((titulo, cuerpo_html))
    return secciones


def main():
    with open(QMD_PATH, encoding="utf-8") as f:
        texto = f.read()

    secciones = build_sections_html(texto)

    nav_items = "\n".join(
        f'<a href="#s{i}" class="nav-item"><span class="nav-num">{i:02d}</span>{titulo}</a>'
        for i, (titulo, _) in enumerate(secciones, start=1)
    )

    secciones_html = "\n".join(
        f"""
        <section class="card" id="s{i}">
          <div class="section-eyebrow"><span class="num">{i:02d}</span><span class="rule"></span></div>
          <h2>{titulo}</h2>
          <div class="prose">{cuerpo}</div>
        </section>
        """
        for i, (titulo, cuerpo) in enumerate(secciones, start=1)
    )

    metric_cards = f"""
      <div class="metric" style="--accent: var(--stable);">
        <span class="metric-value">{METRICS['accuracy']*100:.1f}%</span>
        <span class="metric-label">Accuracy</span>
      </div>
      <div class="metric" style="--accent: var(--indigo);">
        <span class="metric-value">{METRICS['precision']*100:.1f}%</span>
        <span class="metric-label">Precision (riesgo)</span>
      </div>
      <div class="metric" style="--accent: var(--risk);">
        <span class="metric-value">{METRICS['recall']*100:.1f}%</span>
        <span class="metric-label">Recall (riesgo)</span>
      </div>
      <div class="metric" style="--accent: var(--muted-ink);">
        <span class="metric-value">{METRICS['f1']*100:.1f}%</span>
        <span class="metric-label">F1-Score</span>
      </div>
    """

    html_final = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SIPRE, Informe Final</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --paper: #F7F4EC;
    --paper-raised: #FFFFFF;
    --ink: #23241F;
    --muted-ink: #726C5E;
    --indigo: #2B4560;
    --risk: #B5453A;
    --stable: #3C7A5D;
    --line: #E1D9C7;
    --serif: 'Source Serif 4', Georgia, serif;
    --sans: 'Inter', -apple-system, Segoe UI, sans-serif;
    --mono: 'IBM Plex Mono', ui-monospace, monospace;
  }}

  * {{ box-sizing: border-box; }}

  body {{
    margin: 0;
    background: var(--paper);
    color: var(--ink);
    font-family: var(--sans);
    font-size: 16px;
    line-height: 1.65;
  }}

  a {{ color: var(--indigo); }}

  ::selection {{ background: var(--risk); color: #fff; }}

  .layout {{
    display: grid;
    grid-template-columns: 260px minmax(0, 1fr);
    max-width: 1160px;
    margin: 0 auto;
    gap: 0;
  }}

  /* ---------- HERO ---------- */
  .hero {{
    grid-column: 1 / -1;
    padding: 4.5rem 2rem 3rem;
    border-bottom: 1px solid var(--line);
    display: grid;
    grid-template-columns: 1.4fr auto;
    gap: 2rem;
    align-items: center;
  }}
  .hero-eyebrow {{
    font-family: var(--mono);
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted-ink);
    margin-bottom: 1rem;
  }}
  .hero h1 {{
    font-family: var(--serif);
    font-weight: 700;
    font-size: clamp(2rem, 4vw, 3.1rem);
    line-height: 1.08;
    margin: 0 0 0.6rem;
    letter-spacing: -0.01em;
  }}
  .hero .subtitle {{
    font-family: var(--serif);
    font-style: italic;
    color: var(--muted-ink);
    font-size: 1.15rem;
    margin: 0 0 1.5rem;
  }}
  .hero-meta {{
    font-family: var(--mono);
    font-size: 0.85rem;
    color: var(--muted-ink);
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
  }}
  .hero-meta b {{ color: var(--ink); }}

  .gauge-wrap {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-self: end;
  }}
  .gauge-label {{ font-family: var(--serif); font-weight: 700; font-size: 26px; fill: var(--ink); }}
  .gauge-caption {{ font-family: var(--mono); font-size: 10px; fill: var(--muted-ink); letter-spacing: 0.05em; text-transform: uppercase; }}

  .metric-row {{
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border-bottom: 1px solid var(--line);
  }}
  .metric {{
    padding: 1.5rem 2rem;
    border-right: 1px solid var(--line);
    border-top: 3px solid var(--accent);
    background: var(--paper-raised);
  }}
  .metric:last-child {{ border-right: none; }}
  .metric-value {{
    display: block;
    font-family: var(--serif);
    font-weight: 700;
    font-size: 2rem;
    color: var(--ink);
  }}
  .metric-label {{
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--muted-ink);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }}

  /* ---------- SIDEBAR NAV ---------- */
  .sidebar {{
    padding: 2.5rem 1.5rem;
    border-right: 1px solid var(--line);
    position: sticky;
    top: 0;
    align-self: start;
    height: 100vh;
    overflow-y: auto;
  }}
  .sidebar-title {{
    font-family: var(--mono);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted-ink);
    margin-bottom: 1rem;
  }}
  .nav-item {{
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    text-decoration: none;
    color: var(--ink);
    font-size: 0.88rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--line);
    transition: color 0.15s ease;
  }}
  .nav-item:hover {{ color: var(--risk); }}
  .nav-num {{
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted-ink);
  }}

  /* ---------- MAIN CONTENT ---------- */
  .main {{
    padding: 2.5rem 2rem 5rem;
    min-width: 0;
  }}
  .card {{
    background: var(--paper-raised);
    border: 1px solid var(--line);
    border-radius: 3px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.6rem;
  }}
  .section-eyebrow {{
    display: flex;
    align-items: center;
    gap: 0.9rem;
    margin-bottom: 0.6rem;
  }}
  .section-eyebrow .num {{
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--risk);
    font-weight: 600;
  }}
  .section-eyebrow .rule {{
    flex: 1;
    height: 1px;
    background: var(--line);
  }}
  .card h2 {{
    font-family: var(--serif);
    font-weight: 600;
    font-size: 1.65rem;
    margin: 0 0 1.1rem;
    letter-spacing: -0.01em;
  }}
  .prose p {{ margin: 0 0 1rem; color: #3A392F; }}
  .prose strong {{ color: var(--ink); }}
  .prose ul, .prose ol {{ padding-left: 1.3rem; margin-bottom: 1rem; }}
  .prose li {{ margin-bottom: 0.4rem; }}
  .prose code {{
    font-family: var(--mono);
    background: #EFE9D8;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    font-size: 0.88em;
  }}
  .prose table {{
    width: 100%;
    border-collapse: collapse;
    margin: 1.2rem 0;
    font-size: 0.92rem;
  }}
  .prose th, .prose td {{
    border: 1px solid var(--line);
    padding: 0.6rem 0.9rem;
    text-align: left;
  }}
  .prose th {{
    font-family: var(--mono);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    background: #EFE9D8;
    color: var(--muted-ink);
  }}

  .fig {{
    margin: 1.4rem 0;
    text-align: center;
  }}
  .fig img {{
    max-width: 100%;
    border: 1px solid var(--line);
    border-radius: 3px;
  }}
  .fig figcaption {{
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--muted-ink);
    margin-top: 0.5rem;
  }}
  .img-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }}
  @media (max-width: 720px) {{
    .img-row {{ grid-template-columns: 1fr; }}
  }}

  footer {{
    grid-column: 1 / -1;
    padding: 2rem;
    text-align: center;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--muted-ink);
    border-top: 1px solid var(--line);
  }}

  @media (max-width: 880px) {{
    .layout {{ grid-template-columns: 1fr; }}
    .sidebar {{ position: static; height: auto; border-right: none; border-bottom: 1px solid var(--line); }}
    .hero {{ grid-template-columns: 1fr; }}
    .gauge-wrap {{ justify-self: start; }}
    .metric-row {{ grid-template-columns: 1fr 1fr; }}
  }}

  @media print {{
    .sidebar {{ display: none; }}
    .layout {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="layout">

  <div class="hero">
    <div>
      <div class="hero-eyebrow">Informe final · Aprendizaje Automatizado · ITSE</div>
      <h1>SIPRE</h1>
      <p class="subtitle">Sistema Inteligente de Predicción de Riesgo Estudiantil</p>
      <div class="hero-meta">
        <span><b>Profesor:</b> Ronald Ponce</span>
        <span><b>Dataset:</b> UCI Student Performance ({METRICS['n_estudiantes']} estudiantes)</span>
        <span><b>Modalidad:</b> Trabajo grupal</span>
      </div>
    </div>
    <div class="gauge-wrap">
      {gauge_svg(METRICS['accuracy'])}
    </div>
  </div>

  <div class="metric-row">
    {metric_cards}
  </div>

  <nav class="sidebar">
    <div class="sidebar-title">Contenido</div>
    {nav_items}
  </nav>

  <main class="main">
    {secciones_html}
  </main>

  <footer>
    Fuente de datos: Cortez, P., &amp; Silva, A. (2008). <em>Student Performance</em>. UCI Machine Learning Repository.
  </footer>

</div>
</body>
</html>
"""

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_final)

    print(f"Informe HTML (versión con diseño) generado en: {HTML_PATH}")


if __name__ == "__main__":
    main()
