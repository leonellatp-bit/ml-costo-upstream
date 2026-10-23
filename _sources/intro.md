# Predicción del costo unitario de producción en activos upstream de Oil & Gas

**Proyecto final, Tercer entregable** · Machine Learning · Maestría en Ingeniería Industrial
**Estudiante:** Leonella Torreglosa Portillo · Octubre 2026

**Dashboard desplegado:** <https://ml-costo-upstream-dash.onrender.com> ·
**Repositorio:** <https://github.com/leonellatp-bit/ml-costo-upstream>

## Resumen ejecutivo

Se estima el costo unitario de producción, medido como costo de producción equivalente Brent
(`BrentEquivalentProductionCost`, USD/Bbl), de activos upstream de petróleo **que el modelo no ha visto**, a partir
de su geografía, tipo de reserva, ubicación, año y producción. Los datos son de S&P Global Commodity Insights:
4.595 registros activo-año, 695 activos con costo, 99 países y 14 regiones entre 2014 y 2023.

- **Benchmarks:** 28 combinaciones (7 modelos × 4 optimizadores) con validación cruzada anidada por activo. Los
  mejores son XGBoost y Random Forest: RMSE de 1.65–1.72 USD/Bbl y R² ≈ 0.92 en 139 activos nunca vistos, sin
  usar componentes del objetivo.
- **Modelo original (HGSB):** *gradient boosting* programado en numpy que recodifica en cada iteración las
  categóricas geográficas con la media del residuo **contraída hacia su categoría padre** (región → país). Tiene
  el **menor RMSE medio de validación de los 29 modelos** (1.93 frente a 2.01) y el mejor rango medio, pero la
  diferencia **no es estadísticamente significativa** (0 de 5 pruebas tras Holm) y en el hold-out queda por detrás
  de los dos mejores benchmarks. Veredicto: cumple R1 y R2, no cumple R3.
- **Producción:** 8 artefactos serializados con metadatos y prueba de ida y vuelta, y un dashboard multipágina en
  Render con EDA interactivo, predicción en vivo con intervalo conformal y explicación SHAP local.

## Objetivos

**General.** Construir, validar y desplegar un modelo que estime el costo unitario de producción de activos
upstream nuevos.

**Específicos.** (1) Corregir y documentar las observaciones del Entregable 2. (2) Seleccionar con criterio
estadístico la mejor combinación de cada modelo y llevarla a producción. (3) Diseñar, implementar y validar
estadísticamente un modelo original. (4) Publicar un dashboard que permita entender el problema, explorar los
datos y predecir.

## Mapa del libro

| Capítulo | Contenido |
|---|---|
| 1. Trazabilidad de correcciones | Observaciones del Entregable 2 y acciones |
| 2. Datos: ETL y EDA | Identidad contable, faltantes MAR, efectos geográficos, deriva temporal |
| 3. Modelos benchmark | Selección y artefactos (3.0); optimización (3.1), cómputo (3.2), evaluación (3.3), estadística (3.4) e híbridos (3.5) |
| 4. Revisión de literatura y novedad | Cadenas de búsqueda, matriz de novedad, declaración |
| 5. Modelo original | Formulación, proposiciones, pseudocódigo, complejidad, código, pruebas y ablación |
| 6. Validación estadística | Protocolo, supuestos, pruebas con Holm, MCS, efectos, robustez y veredicto |
| 7. Dashboard y despliegue | Arquitectura, decisiones de diseño, guía de Render |
| 8. Conclusiones y limitaciones | Hallazgos, veredicto R1–R3, trabajo futuro, declaración de uso de IA |
| 9. Reproducibilidad y bibliografía | Entorno, orden de ejecución, semillas, referencias |

```{tableofcontents}
```
