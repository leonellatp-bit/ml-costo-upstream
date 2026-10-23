# Predicción del costo unitario de producción en activos upstream de Oil & Gas — Entrega final

**Leonella Torreglosa Portillo** · Machine Learning · Maestría en Ingeniería Industrial · Octubre 2026

Regresión del `BrentEquivalentProductionCost` (USD/Bbl) de 695 activos upstream de petróleo (99 países, 2014–2023,
S&P Global Commodity Insights) con 7 modelos × 4 optimizadores de hiperparámetros, validación cruzada anidada por
activo y comparación estadística jerárquica.

## Estructura

```
proyecto/
├── data/                         Dataset (licencia corporativa S&P Global; uso académico)
├── src/                          Código modular (docstrings estilo NumPy)
│   ├── config.py                 Semilla global, rutas, variables, escenarios y esquema de validación
│   ├── data.py                   Carga, partición por grupos, preprocesamiento en Pipeline
│   ├── models.py                 7 modelos, espacios de búsqueda y complejidad O(·)
│   ├── optimizers.py             Grid, Random, Bayesiana (Optuna-TPE), Genética (DEAP)
│   ├── nested_cv.py              CV anidada combinatoria + tabla maestra de experimentos
│   ├── computational.py          Escalamiento, estándar vs optimizado, Successive Halving, paralelismo, profiling
│   ├── evaluation.py             White, Ljung-Box, BDS, normalidad, bootstrap BCa
│   ├── stats_tests.py            Friedman, Nemenyi/CD, MCS, Giacomini-White, Clark-West, DM-HLN, SPA, Holm/BH, Cliff, Cohen
│   ├── interpret.py              SHAP y LIME
│   └── viz.py                    Paleta y estilo comunes
├── run_nested.py                 Etapa pesada: 28 + 7 combinaciones (≈ 2 h en 2 núcleos)
├── book/                         Jupyter Book (notebooks ejecutados + _config.yml + _toc.yml)
│   └── src/                      Fuente de cada notebook en formato "percent" (.py)
├── dashboard/                    App Dash final (app.py, build_data.py, assets/)
├── results/                      Tabla maestra, caché de cada corrida, modelos finales, CSV de análisis
├── tools/nbtools.py              Conversión .py → .ipynb y ejecutor mínimo
└── requirements.txt              Versiones exactas
```

## Reproducir

```bash
conda create -n ml_proyecto python=3.13 -y
conda activate ml_proyecto
pip install -r requirements.txt

python run_nested.py                         # 1) CV anidada (usa caché si se interrumpe)
jupyter nbconvert --execute --inplace book/0*.ipynb   # 2) notebooks (o: jupyter-book build book/)
python dashboard/build_data.py               # 3) datos del dashboard
python dashboard/app.py                      # 4) http://127.0.0.1:8050
```

Semilla global única: `SEED = 42` en `src/config.py`, propagada a numpy, random, scikit-learn, XGBoost, Optuna y DEAP.
Cada corrida queda registrada en `results/cache/` y resumida en `results/master_table.csv`
(modelo, optimizador, métricas del bucle externo media ± d.e., métricas de prueba, tiempos, evaluaciones,
hiperparámetros finales y semilla).

## Nota sobre el entorno en que se generaron los resultados

Los resultados incluidos se generaron en un entorno Linux con 2 núcleos, sin acceso a PyPI. Por eso:

- Las pruebas estadísticas (White, BDS, Ljung-Box, MCS, GW, CW, DM, SPA) están implementadas en `src/` con
  numpy/scipy, sin depender de statsmodels ni arch.
- Los notebooks se ejecutaron con `tools/nbtools.py` (ejecutor mínimo); en una máquina con Jupyter se
  pueden re-ejecutar con `nbconvert` o `jupyter-book build`.
