# Predicción del costo unitario de producción en activos upstream de Oil & Gas

**Proyecto final — Segundo entregable** · Machine Learning · Maestría en Ingeniería Industrial
**Estudiante:** Leonella Torreglosa Portillo · Octubre 2026

## Problema

Estimar el costo unitario de producción, medido como costo de producción equivalente Brent
(`BrentEquivalentProductionCost`, USD/Bbl), de un activo upstream de petróleo **que el modelo no ha visto** a partir de sus características geográficas, operativas y temporales. Los datos provienen de S&P Global
Commodity Insights (Supply Cost and Ownership): 4.595 registros activo-año, 695 activos con costo observado,
99 países y 14 regiones entre 2014 y 2023.

## Qué cambió respecto al primer entregable

1. **EDA corregido.** Se descubrió que el target es una suma exacta: `BrentEquivalentProductionCost =
   WellheadProductionCost + BrentDifferential`. Por eso el modelo principal (escenario A) excluye
   `BrentDifferential`, que el modelo base del entregable 1 usaba como predictor. El escenario B lo incluye
   como análisis de sensibilidad.
2. **28 combinaciones** (7 modelos × 4 optimizadores; el balanceo de clases no aplica en regresión) con
   **validación cruzada anidada** por activo (10 pliegues externos × 3 internos) y presupuesto igual de 30
   evaluaciones por búsqueda.
3. **Optimización computacional** con análisis de complejidad O(·) contrastado con tiempos medidos, variantes
   eficientes de cada algoritmo, Successive Halving, paralelización y perfilamiento.
4. **Evaluación** con pruebas de supuestos (White, BDS, Ljung-Box), robustez entre pliegues y semillas, e
   interpretabilidad con SHAP y LIME.
5. **Comparación estadística jerárquica:** Friedman → Nemenyi (diagrama CD) → MCS → Giacomini-White /
   Clark-West → Diebold-Mariano (HLN) con bootstrap estacionario, SPA, corrección de Holm/BH, tamaños de
   efecto e intervalos BCa.

## Capítulos

```{tableofcontents}
```

## Reproducibilidad

- Semilla global única `SEED = 42` (`src/config.py`), propagada a numpy, random, scikit-learn, XGBoost, Optuna y DEAP.
- Versiones exactas en `requirements.txt`.
- Registro estructurado: cada corrida (escenario, modelo, optimizador, pliegue) queda en `results/cache/` y la
  tabla maestra en `results/master_table.csv`, con hiperparámetros finales, métricas, tiempos y semilla.
- Código modular en `src/` con docstrings; `run_nested.py` ejecuta la etapa pesada y los notebooks leen sus
  resultados.
