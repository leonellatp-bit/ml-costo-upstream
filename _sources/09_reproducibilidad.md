# 9. Reproducibilidad y bibliografía

## 9.1 Entorno

- Python 3.13 con versiones exactas en `requirements.txt` (experimentos) y `requirements-render.txt`
  (dashboard): numpy, pandas, scipy, scikit-learn, xgboost, optuna, deap, shap, plotly, dash,
  dash-bootstrap-components, joblib y gunicorn.
- Entorno conda local: `ml_venv2`.
- **Semilla global única `SEED = 42`** (`src/config.py`), propagada a numpy, random, scikit-learn, XGBoost, Optuna,
  DEAP y al modelo original.

## 9.2 Orden de ejecución

| Paso | Comando | Produce | Tiempo aprox. |
|---|---|---|---|
| 1 | `python run_nested.py` | 385 búsquedas de los benchmarks en `results/cache/`, `results/master_table.csv` | 80 min (2 núcleos) |
| 2 | `python run_original.py` | 11 búsquedas del HGSB con el mismo protocolo | 16 min |
| 3 | `python run_original_stats.py` | Pruebas, MCS, ablación, convergencia y robustez en `results/original/` | 15 min |
| 4 | `python run_blend.py`, `python run_blend2.py` y `python run_weighted_forest.py` | Modelos híbridos (capítulo 3.5) y bosques con pesos por árbol (5.9) | 6 min |
| 5 | `python tools/build_artifacts.py` | `artifacts/` (8 modelos + metadata), tablas del dashboard | 3 min |
| 6 | `pytest -q tests/` | 19 pruebas unitarias del modelo original | 5 s |
| 7 | `jupyter-book build book/` | Informe HTML | 2 min |
| 8 | `python app.py` | Dashboard local en <http://127.0.0.1:8050> | — |

Los notebooks se entregan ejecutados (`execute_notebooks: "off"`): leen los resultados precalculados de
`results/`, que están versionados en el repositorio, y el código que los genera está en los scripts del paso 1 al
5. Para reconstruir todo desde cero basta con ejecutar los pasos en orden; las semillas fijas reproducen las mismas
cifras.

## 9.3 Tabla maestra de experimentos

`results/master_table.csv` contiene las 35 combinaciones del Entregable 2 (28 del escenario A + 7 del B) y
`results/original/folds.csv` las métricas por pliegue de las 28 combinaciones y del HGSB. Cada búsqueda guarda en
`results/cache/` hiperparámetros, métricas externas, tiempos, historial y semilla.

## 9.4 Datos

El conjunto de S&P Global Commodity Insights se usa bajo licencia corporativa con fines académicos y **no se
publica**: `data/` está en `.gitignore`. Para reproducir, se copia `DatasetOilCostProjectWorldwide.csv` en `data/`.
En Render, se sube como *Secret File*.

## 9.5 Bibliografía

```{bibliography}
```
