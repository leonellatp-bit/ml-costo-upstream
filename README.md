# Predicción del costo unitario de producción en activos upstream de Oil & Gas

**Leonella Torreglosa Portillo** · Machine Learning · Maestría en Ingeniería Industrial · Octubre 2026

Regresión del costo unitario de producción (`BrentEquivalentProductionCost`, USD/Bbl) de 695 activos upstream de
petróleo (99 países, 2014–2023, S&P Global Commodity Insights). Incluye 7 modelos × 4 optimizadores con
validación cruzada anidada por activo, un **modelo original (HGSB)** con validación estadística y un dashboard
desplegado.

- **Dashboard:** <https://ml-costo-upstream-dash.onrender.com>
- **Informe (Jupyter Book):** <https://leonellatp-bit.github.io/ml-costo-upstream>

## Estructura

```
├── app.py                    Punto de entrada del dashboard (Dash multipágina, expone `server`)
├── pages/                    contexto.py · eda.py · modelos.py
├── src/
│   ├── original_model.py     Modelo original HGSB (numpy; compatible con scikit-learn)
│   ├── config.py             Semilla global, rutas, variables, escenarios, esquema de validación
│   ├── data.py               Carga, partición por grupos, preprocesamiento en Pipeline
│   ├── models.py             7 modelos, espacios de búsqueda y complejidad
│   ├── optimizers.py         Grid, Random, Bayesiana (Optuna-TPE), Genética (DEAP)
│   ├── nested_cv.py          CV anidada + tabla maestra
│   ├── stats_tests.py        Friedman, Nemenyi, MCS, GW, CW, DM-HLN, SPA, Holm/BH, Cliff, Cohen
│   ├── evaluation.py · computational.py · interpret.py · viz.py
│   ├── dash_data.py          Carga y caché para el dashboard
│   ├── plots.py              Gráficos Plotly reutilizables
│   ├── explain.py            Shapley por grupos e intervalos conformales
│   └── theory_content.py     Teoría de los modelos y conclusiones del dashboard
├── artifacts/<Modelo>/       model.joblib + metadata.json (8 modelos), leaderboard.csv, conformal.json
├── tests/                    Pruebas unitarias del modelo original (pytest)
├── run_nested.py             Benchmarks (≈ 80 min, 2 núcleos)
├── run_original.py           HGSB con el mismo protocolo (≈ 16 min)
├── run_original_stats.py     Pruebas de hipótesis, MCS, ablación, robustez
├── run_blend.py, run_blend2.py  Modelos híbridos (capítulo 3.5)
├── run_weighted_forest.py    Bosques con pesos por árbol (capítulo 5.9)
├── tools/                    build_artifacts.py, nbtools.py
├── book/                     Jupyter Book (capítulos 0–9, references.bib)
├── results/                  Tablas y resultados precalculados (caché y modelos pesados en .gitignore)
├── assets/                   CSS e ilustración del dashboard
├── requirements.txt          Versiones exactas (experimentos)
├── requirements-render.txt   Versiones exactas (despliegue)
└── render.yaml               Web Service de Render: gunicorn app:server
```

## Reproducir

```bash
conda create -n ml_venv2 python=3.13 -y && conda activate ml_venv2
pip install -r requirements.txt
# copiar DatasetOilCostProjectWorldwide.csv en data/  (licencia: no se publica)
python run_nested.py
python run_original.py
python run_original_stats.py
python tools/build_artifacts.py
pytest -q tests/
jupyter-book build book/
python app.py        # http://127.0.0.1:8050
```

Semilla global `SEED = 42` (`src/config.py`). Los notebooks se entregan ejecutados.

## Despliegue

Render lee `render.yaml`: `pip install -r requirements-render.txt` y `gunicorn app:server`. El CSV con licencia se
sube como *Secret File* (`/etc/secrets/DatasetOilCostProjectWorldwide.csv`).
