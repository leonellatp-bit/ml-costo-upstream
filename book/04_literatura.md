# 4. Revisión de literatura y novedad

Este capítulo documenta la evidencia de novedad del modelo original (condición R1 de la guía): se buscó si el
mecanismo propuesto, un *gradient boosting* que en cada iteración recodifica las categóricas con la media del
residuo **contraída hacia su categoría padre en una jerarquía del dominio**, ya se ha usado para predecir el
costo unitario de producción de activos petroleros.

## 4.1 Protocolo de búsqueda

- **Bases de datos:** Scopus, Google Scholar y arXiv, más un motor web general para literatura técnica de la
  industria (SPE, *Energies*, repositorios institucionales).
- **Componentes de cada cadena:** (i) nombre o familia del modelo, (ii) problema de aplicación, (iii) sinónimos en
  inglés y español.
- **Revisión:** título y resumen de todos los resultados revisados; texto completo de los más cercanos.
- **Fecha de consulta:** 23 de octubre de 2026 (arXiv y web). Scopus y Google Scholar: ver la columna *Fecha*.

**Cadenas para Scopus y Google Scholar** (sintaxis de Scopus; en Scholar se usan las mismas palabras sin
`TITLE-ABS-KEY`):

| ID | Cadena de búsqueda | Componentes |
|---|---|---|
| S1 | `TITLE-ABS-KEY(("gradient boosting" OR "XGBoost" OR "boosted trees") AND ("production cost" OR "lifting cost" OR "operating cost" OR "breakeven") AND (oil OR petroleum OR upstream))` | familia + problema |
| S2 | `TITLE-ABS-KEY(("hierarchical" AND ("target encoding" OR "shrinkage" OR "empirical Bayes")) AND ("gradient boosting" OR "decision tree"))` | mecanismo |
| S3 | `TITLE-ABS-KEY(("hierarchical shrinkage" OR "hierarchical encoding" OR "categorical hierarchy") AND (oil OR petroleum OR "oil and gas") AND cost)` | mecanismo + problema |
| S4 | `TITLE-ABS-KEY(("machine learning") AND ("cost per barrel" OR "unit production cost" OR "supply cost") AND (oil OR upstream))` | problema (cualquier modelo) |
| S5 | `"costo de producción" petróleo ("aprendizaje automático" OR "machine learning") ("boosting" OR "bosques aleatorios")` | sinónimos en español |

**Cadenas ejecutadas en arXiv y en el motor web** (texto exacto de la consulta):

| ID | Cadena ejecutada | Componentes |
|---|---|---|
| W1 | `machine learning prediction oil production cost per barrel upstream assets gradient boosting` | familia + problema |
| W2 | `site:arxiv.org hierarchical target encoding gradient boosting categorical empirical Bayes shrinkage` | mecanismo (arXiv) |
| W3 | `"lifting cost" OR "breakeven" oil fields machine learning regression random forest XGBoost` | problema + sinónimos |
| W4 | `hierarchical categorical variables gradient boosting tree shrinkage parent category regression` | mecanismo |
| W5 | `"costo de producción" petróleo "aprendizaje automático" predicción por barril` | sinónimos en español |
| W6 | `machine learning operating cost OPEX estimation oil and gas fields regression model study` | problema (OPEX) |

## 4.2 Matriz de novedad

```{note}
Las filas de **Scopus** y **Google Scholar** se completan ejecutando las cadenas S1–S5 con acceso institucional y
registrando resultados, revisados y relevantes. En arXiv y en el motor web se revisaron los primeros resultados
que devuelve cada consulta (la columna *Result.* indica cuántos se revisaron, no el total de la base).
```

| Base de datos | Cadena | Fecha | Result. | Revis. | Relev. | Observación |
|---|---|---|---|---|---|---|
| Scopus | S1 | _completar_ | _ | _ | _ | _ |
| Scopus | S2 | _completar_ | _ | _ | _ | _ |
| Scopus | S3 | _completar_ | _ | _ | _ | _ |
| Scopus | S4 | _completar_ | _ | _ | _ | _ |
| Google Scholar | S1 | _completar_ | _ | _ | _ | _ |
| Google Scholar | S5 | _completar_ | _ | _ | _ | _ |
| arXiv | W2 | 2026-10-23 | 9 | 9 | 3 | CatBoost {cite}`prokhorenkova2018catboost`, codificación regularizada del objetivo {cite}`pargent2022regularized` y un estudio comparativo de codificadores: codifican el **objetivo** (preprocesamiento o estadísticos ordenados); ninguno usa una jerarquía del dominio dentro del boosting ni aborda costos petroleros. |
| Web | W1 | 2026-10-23 | 10 | 10 | 1 | {cite}`alkhammash2022gagb`: boosting optimizado con algoritmo genético para **pronosticar producción** de crudo, no costo. |
| Web | W3 | 2026-10-23 | 10 | 10 | 0 | Productividad de pozos, desempeño SAGD y recomendación de intervenciones; ninguno predice costo unitario. |
| Web | W4 | 2026-10-23 | 9 | 9 | 2 | CatBoost y una regresión **lineal** con categóricas jerárquicas por programación entera mixta (*Expert Systems with Applications*, 2022). |
| Web | W5 | 2026-10-23 | 9 | 9 | 0 | Pronóstico de precios y de producción, no de costo unitario. |
| Web | W6 | 2026-10-23 | 9 | 9 | 1 | {cite}`kaborogo2024proxy`: modelos *proxy* de CAPEX/DRILLEX/OPEX con **regresión lineal múltiple**; sin ML ni categorías. |

## 4.3 Trabajos más cercanos y en qué se diferencia la propuesta

La tabla compara familias algorítmicas afines, no solo aplicaciones.

| Método | Qué hace | Qué conserva el HGSB | Qué cambia el HGSB |
|---|---|---|---|
| XGBoost {cite}`chen2016xgboost` | Boosting de segundo orden con hojas $-G/(H+\lambda)$ | Ganancia y pesos de hoja de 2.º orden | Las categóricas no se expanden en One-Hot: se recodifican cada iteración |
| LightGBM {cite}`ke2017lightgbm` | Ordena categorías por $G/(H+\text{cat\_smooth})$ dentro de cada nodo | Ordenar categorías por residuo medio (partición óptima de {cite}`fisher1958grouping`) | Contrae hacia el **padre jerárquico**, no hacia 0; cuenta **activos**, no filas; categoría nueva hereda del padre |
| CatBoost {cite}`prokhorenkova2018catboost` | Estadísticos ordenados del objetivo para evitar fuga | Codificación de categorías con un estimador regularizado | Codifica el **residuo de la iteración**, con un prior jerárquico |
| Codificación jerárquica del objetivo {cite}`micci2001preprocessing` | Media del objetivo por categoría, contraída hacia el nivel superior | La fórmula de contracción empírica Bayes | Se aplica **dentro** del boosting (una vez por árbol), no como preprocesamiento único |
| Codificación regularizada {cite}`pargent2022regularized` | Target encoding con validación cruzada y suavizado | Regularización de categorías de alta cardinalidad | Prior jerárquico y recodificación iterativa |
| GPBoost / MERF {cite}`sigrist2022gpboost,hajjem2014merf` | Boosting o bosques más efectos aleatorios por grupo | Préstamo de información entre grupos | Los efectos actúan **dentro de los cortes** del árbol, no como término aditivo |
| Hierarchical Shrinkage {cite}`agarwal2022hierarchical` | Contrae cada hoja hacia sus nodos ancestros del árbol | La idea de contracción jerárquica | La jerarquía es la **del dominio** (región → país), no la del árbol |
| Regresión lineal con categóricas jerárquicas (ESWA, 2022) | Fusiona niveles jerárquicos con optimización entera mixta | Uso explícito de la jerarquía | Modelo no lineal (boosting) en lugar de lineal |

## 4.4 Declaración de novedad

```{important}
Al mejor de nuestro conocimiento y según la búsqueda descrita (arXiv y motor web el 23-10-2026; Scopus y Google
Scholar según la matriz), **no se encontraron estudios que apliquen un gradient boosting con recodificación
jerárquica contraída de las categóricas (HGSB) a la predicción del costo unitario de producción de activos
upstream**. Los trabajos más cercanos en la aplicación pronostican producción {cite}`alkhammash2022gagb` o
estiman CAPEX/OPEX con regresión lineal {cite}`kaborogo2024proxy`; los más cercanos en el mecanismo
{cite}`micci2001preprocessing,ke2017lightgbm,prokhorenkova2018catboost,agarwal2022hierarchical` usan
codificaciones o contracciones que no combinan las tres piezas del HGSB: (1) prior del padre en una jerarquía del
dominio, (2) recodificación del residuo en cada iteración y (3) conteo por grupo (activo) en lugar de por fila.
```

La novedad es **de mecanismo y de aplicación**: el mecanismo se formula de forma general (cualquier problema de
regresión con categóricas anidadas y observaciones agrupadas), así que es transferible a otras tareas, como costos
por planta o por región comercial.
