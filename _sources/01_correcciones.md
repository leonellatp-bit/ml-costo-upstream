# 1. Trazabilidad de correcciones

Este capítulo registra cada observación sobre el Entregable 2 y la acción realizada.

## 1.1 Observaciones y acciones realizadas

Revisión crítica del Entregable 2 y de su sustentación. Cada fila indica la acción tomada, dónde se ve en el libro y
la evidencia.

| # | Observación | Acción realizada | Ubicación en el Jupyter Book | Evidencia |
|---|---|---|---|---|
| P1 | El título usaba el nombre técnico de la columna objetivo ("costo de producción equivalente Brent") y no el concepto de negocio. | Título unificado: "costo unitario de producción", con la definición técnica en la introducción. | {doc}`intro` | `_config.yml`, `intro.md`, `app.py` |
| P2 | No se justificaba por qué Naive Bayes y la regresión logística (sección 7.2 de la guía del E2) no estaban entre los modelos. | Se explica que son clasificadores y que sus equivalentes en regresión (Ridge/Lasso, SVR) sí están cubiertos. | {doc}`02_modelos_optimizacion` (introducción) | Párrafo "¿Y Naive Bayes y Regresión Logística L1/L2?" |
| P3 | Faltaba evaluar si combinar los mejores modelos mejoraba el desempeño. | Tres mezclas con pesos estimados sin fuga (Bates-Granger, RF + SVR, Super Learner) y sus pruebas estadísticas. | {doc}`05b_modelos_hibridos` | Tabla resumen y matriz de correlación de errores |
| P4 | El dashboard era una sola página con pestañas, entrenaba desde pickles sin metadatos y no daba intervalos ni explicación local. | Dashboard multipágina (Contexto, EDA, Modelos) con artefactos `joblib` + `metadata.json`, prueba de ida y vuelta, intervalos conformales y SHAP local. | {doc}`07_dashboard` | `app.py`, `pages/`, `artifacts/` |
| P5 | Las predicciones del EDA no tenían frase de interpretación en cada gráfico. | Cada gráfico del EDA del dashboard muestra una frase calculada sobre la selección actual. | {doc}`07_dashboard` | `pages/eda.py` |
| P6 | La comparación estadística del E2 usaba pruebas de series de tiempo (DM, GW, CW) sin discutir que el problema es un panel de corte transversal. | Para el modelo original se usan las pruebas de la columna de corte transversal de la guía (Wilcoxon, Nadeau-Bengio, permutación, bootstrap por activo) y se mantiene el MCS como procedimiento global, con bootstrap por bloques en orden activo-año. | {doc}`06_validacion` | Tabla de pruebas de hipótesis |

## 1.2 Resultados actualizados y cambios en el ranking

- Ninguna corrección cambió los datos ni las particiones: el hold-out (139 activos), los 10 pliegues externos y la
  semilla 42 son los mismos del Entregable 2. Por eso las métricas de los 28 benchmarks no cambian.
- La tabla maestra se amplió con el modelo original (HGSB), evaluado con el mismo protocolo.
- Al incluir el HGSB el ranking cambia en validación cruzada: el HGSB pasa al primer lugar (RMSE externo 1.93 y mejor
  rango medio de Friedman), pero en el conjunto de prueba queda detrás de XGBoost y Random Forest y la diferencia no
  es significativa. Ambos hechos se reportan en los capítulos 5, 6 y 8.
