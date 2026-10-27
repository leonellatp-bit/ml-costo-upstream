# 8. Conclusiones y limitaciones

## Respuesta a la pregunta de investigación

**Sí es posible estimar el costo unitario de producción (equivalente Brent, USD/Bbl) de un activo upstream que el modelo nunca ha
visto**, con un error típico de 1.6–1.7 USD/Bbl (RMSE en el hold-out de 139 activos; MAE ≈ 0.95 USD/Bbl;
MAPE ≈ 6.7 %; R² ≈ 0.91–0.92), usando solo características estructurales del activo (región, país, cuenca,
tipo y subtipo de reserva, ubicación, OPEP, año y producción). Lo logran los ensambles de árboles, Random
Forest y XGBoost, sin una diferencia estadísticamente demostrable entre ellos.

## Hallazgos principales

1. **Corrección metodológica del primer entregable.** El target cumple la identidad exacta
   `BrentEquivalentProductionCost = WellheadProductionCost + BrentDifferential`. El modelo base del primer
   entregable (R² = 0.857) usaba `BrentDifferential`, un sumando del target. Sin esa variable, el mismo SVR
   lineal cae a R² = 0.78; los ensambles del escenario estructural recuperan y superan ese nivel
   (R² ≈ 0.92 en prueba) sin usar ningún componente del target. Con `BrentDifferential` (escenario B) el error
   baja otro 30 % (Clark-West p < 10⁻¹⁵), lo cual es legítimo solo si el diferencial se conoce antes que el costo.
2. **La familia de modelo importa más que el optimizador.** Friedman entre los 7 modelos rechaza la igualdad
   (p = 5.5 × 10⁻⁶) y Nemenyi separa ensambles de lineales con tamaños de efecto grandes (δ de Cliff de −0.56 a
   −0.76). Entre los 4 optimizadores, Friedman no encuentra diferencias (p = 0.89).
3. **La optimización Bayesiana es la más eficiente por presupuesto, pero la ganancia es interna.** Con 30
   evaluaciones, Optuna-TPE llega a ≤ 1 % del mejor óptimo en el 93 % de las búsquedas (Random 61 %, GA 71 %,
   Grid 51 %; Wilcoxon p < 0.001, efecto grande frente a Random) y es la más estable entre semillas. Grid es el
   peor en rapidez y no escala a espacios de 7 dimensiones. Esa ventaja (décimas de punto porcentual en el RMSE
   interno) queda por debajo del ruido entre pliegues externos. La excepción es SVR: con 3 hiperparámetros
   continuos en escala log que interactúan, la Bayesiana reduce el RMSE externo de 2.35–2.47 a 2.19.
4. **El error es persistente por activo.** Los residuos están fuertemente correlacionados dentro de cada activo
   (ρ = 0.75 entre años consecutivos) y crecen con el nivel de costo (White p ≈ 10⁻²⁶). El modelo acierta el
   nivel de la región y del tipo de activo, pero cada campo tiene un efecto propio no observado. Esto justifica
   la validación por grupos y el bootstrap por bloques, y marca el límite de lo que pueden explicar las variables
   disponibles.
5. **Interpretabilidad.** La región es el factor dominante (media |SHAP| ≈ 2.2 USD/Bbl: Norteamérica hasta
   +7, Medio Oriente ≈ −2 a −3), seguida del año (ciclo 2015–2023), la cuenca y el país. Ser no convencional es
   poco frecuente pero suma +3 a +10 USD/Bbl. La producción aporta poco una vez conocida la geografía. SHAP y
   LIME coinciden en los casos extremos y divergen en casos con efectos compensados, por las perturbaciones
   irreales de LIME en variables jerárquicas.
6. **Optimización computacional.** SVR-RBF es el único modelo con escalamiento cuadrático medido (exponente
   ≈ 2.0, como predice la teoría). XGBoost `hist` es 8.4× más rápido que `exact` sin perder precisión. KD-Tree y
   Ball-Tree son más lentos que la fuerza bruta en 173 dimensiones. Successive Halving ahorra 75 % del tiempo en
   Random Forest sin pérdida, pero en XGBoost empeora el resultado porque "pocos árboles" no preserva el ranking
   de configuraciones con *learning rate* bajo.

7. **Combinar modelos no rompe el límite.** Se propusieron tres modelos híbridos con pesos estimados sin fuga:
   RF + XGBoost (peso óptimo de Bates-Granger, w = 0.83), RF + SVR y un Super Learner convexo de los 7 modelos.
   Ninguno supera de forma significativa al mejor modelo individual (Diebold-Mariano p ≥ 0.39 frente a XGBoost;
   en CV, RF + SVR logra 1.992 frente a 2.008 de RF, Wilcoxon p = 0.16). La teoría lo anticipa: los errores de
   RF y XGBoost tienen correlación 0.96, así que la ganancia máxima de combinarlos es de 0.2 % del MSE. Todos los
   modelos fallan en los mismos activos (correlación de errores ≥ 0.57): el límite está en la información, no
   en el algoritmo.

8. **El modelo original (HGSB) tiene el menor error de validación, pero no demuestra superioridad.** Con el
   mismo protocolo que los 28 benchmarks, el HGSB logra el menor RMSE medio del bucle externo de los 29 modelos
   (1.927 ± 0.579 frente a 2.008 ± 0.549 de Random Forest), es mejor en 7 de 10 pliegues y tiene el mejor rango
   medio de Friedman (1.70). Sin embargo, ninguna de las cinco pruebas rechaza la igualdad (Wilcoxon p = 0.14 y
   Nadeau-Bengio p = 0.21 antes de Holm; MCS deja al mejor benchmark dentro), el δ de Cliff (−0.14) queda justo
   bajo el umbral de 0.147 y en el hold-out el HGSB es peor (1.884 frente a 1.717). La ablación sí respalda el
   diseño: quitar la jerarquía, la contracción o la codificación empeora el RMSE externo (+0.08, +0.09 y +0.18).

## Veredicto sobre el modelo original

| Condición | Resultado | Evidencia |
|---|---|---|
| **R1** Novedad en la aplicación | Cumple (según la búsqueda documentada) | Capítulo 4: no se encontraron estudios que apliquen boosting con recodificación jerárquica contraída al costo unitario upstream |
| **R2** Implementación algorítmica propia | Cumple | `src/original_model.py`: codificación, árboles por histogramas y boosting programados en numpy; 19 pruebas unitarias |
| **R3** Superioridad estadísticamente verificada | **No cumple** | Menor RMSE medio externo, pero 0 de 5 pruebas significativas tras Holm y δ de Cliff despreciable (capítulo 6) |

Por la regla de la sección 12 de la guía, el tope que corresponde es **4.0** (R1 y R2 sí, R3 no).

**Causas probables de la falta de significancia.** (i) Potencia baja: con 10 pliegues y una desviación entre
pliegues de 0.55 USD/Bbl, una mejora media de 0.08 USD/Bbl (4 %) no alcanza significancia. (ii) El error dominante
es un efecto propio del activo que ninguna variable observa: los errores de los 8 modelos tienen correlación de al
menos 0.57. (iii) El HGSB elige árboles poco profundos con contracción fuerte: generaliza mejor en promedio, pero
ajusta peor los activos no convencionales extremos, que son más frecuentes en el hold-out (6 % frente a 3.4 %).

## Limitaciones

- **Tamaño efectivo.** 3.834 filas con target, pero solo 695 activos independientes (por debajo de las 20.000
  observaciones sugeridas por la guía), con 45 países representados por un único activo.
- **Representatividad.** Predominan activos convencionales (96 %) y de Norteamérica, África y Latinoamérica; las
  predicciones para no convencionales y regiones poco representadas tienen más incertidumbre (el peor error en
  prueba es una mina de bitumen canadiense).
- **Faltantes estructurales.** `Production` no está disponible para África, Asia Central ni casi toda Asia
  Oriental; para esos activos el modelo nunca usa la producción real.
- **Presupuesto de optimización.** 30 evaluaciones por búsqueda favorecen a métodos que exploran rápido; el
  algoritmo genético, que necesita más generaciones, queda en desventaja estructural.
- **Licencia de datos.** Los datos de S&P Global se usan bajo licencia corporativa con fines académicos; los
  resultados se reportan de forma agregada.

## Trabajo futuro

- Modelar `log(costo)` o usar pérdidas relativas para atender la heterocedasticidad.
- Para activos con historia (pronóstico, no activos nuevos), incorporar el costo del año anterior o efectos
  aleatorios por activo para capturar el componente persistente.
- Bootstrap por conglomerados (activo) para intervalos que respeten totalmente la dependencia intra-activo.
- Mayor presupuesto para el GA (población más grande, mutación adaptativa) y una fidelidad para Successive
  Halving que preserve el ranking en XGBoost (submuestreo de filas).
- Validación cruzada externa **repetida** (por ejemplo 5 × 5) para todos los modelos, que daría más potencia a las
  pruebas del HGSB frente a Random Forest.
- HGSB con **codificación por nodo** (como LightGBM, pero con prior jerárquico) y pérdida relativa, para no perder
  ajuste en los activos de costo extremo.
- Evaluar el HGSB en otros conjuntos con categóricas anidadas (costos por planta o región comercial) para medir su
  transferibilidad.

## Declaración de autoría y uso de herramientas de IA

- **Autora:** Leonella Torreglosa Portillo, responsable del planteamiento del problema, las decisiones de
  modelado, la interpretación de resultados y la sustentación.
- **Herramientas de IA:** se usó un asistente de IA (Claude, de Anthropic) para apoyar la escritura y depuración
  de código (pipeline de validación anidada, dashboard, implementación del HGSB y sus pruebas), la redacción de
  borradores del informe y la búsqueda de literatura (OnePetro, Semantic Scholar, arXiv y web). Todo el código fue revisado,
  ejecutado y validado por la autora, que puede explicar y modificar cualquier parte del algoritmo y de las pruebas.
- **Datos:** S&P Global Commodity Insights, bajo licencia corporativa con fines académicos.
