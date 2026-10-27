# 4. Revisión de literatura y novedad

Este capítulo documenta la evidencia de novedad del modelo original (condición R1 de la guía): se buscó si el
mecanismo propuesto, un *gradient boosting* que en cada iteración recodifica las categóricas con la media del
residuo **contraída hacia su categoría padre en una jerarquía del dominio**, ya se ha usado para predecir el
costo unitario de producción de activos petroleros.

## 4.1 Protocolo de búsqueda

- **Bases de datos (4):** **OnePetro** (literatura técnica de la Society of Petroleum Engineers, la base de
  referencia del sector), **Semantic Scholar** (base académica multidisciplinaria), **arXiv** (preprints de
  aprendizaje automático) y un motor web general para revistas y repositorios institucionales (*Energies*, MDPI,
  repositorios universitarios).
- **Componentes de cada cadena:** (i) nombre o familia del modelo, (ii) problema de aplicación, (iii) sinónimos en
  inglés y español.
- **Revisión:** título de todos los resultados devueltos; resumen y texto completo de los más cercanos cuando fueron
  accesibles.
- **Fecha de consulta:** 23 y 27 de octubre de 2026.

**Cadenas ejecutadas** (texto exacto de la consulta):

| ID | Base | Cadena ejecutada | Componentes |
|---|---|---|---|
| O1 | OnePetro | `machine learning prediction production cost per barrel oil fields gradient boosting` | familia + problema |
| O2 | OnePetro | `machine learning operating cost OPEX lifting cost estimation upstream assets` | problema + sinónimos |
| S1 | Semantic Scholar | `hierarchical shrinkage categorical encoding gradient boosting empirical Bayes` | mecanismo |
| S2 | Semantic Scholar | `oil production cost prediction machine learning upstream breakeven` | problema + sinónimos |
| A1 | arXiv | `site:arxiv.org hierarchical target encoding gradient boosting categorical empirical Bayes shrinkage` | mecanismo |
| W1 | Web | `machine learning prediction oil production cost per barrel upstream assets gradient boosting` | familia + problema |
| W2 | Web | `"lifting cost" OR "breakeven" oil fields machine learning regression random forest XGBoost` | problema + sinónimos |
| W3 | Web | `hierarchical categorical variables gradient boosting tree shrinkage parent category regression` | mecanismo |
| W4 | Web | `"costo de producción" petróleo "aprendizaje automático" predicción por barril` | sinónimos en español |
| W5 | Web | `machine learning operating cost OPEX estimation oil and gas fields regression model study` | problema (OPEX) |

## 4.2 Matriz de novedad

En cada consulta se revisaron los resultados que devuelve el buscador de la base (la columna *Revis.* indica
cuántos se revisaron, no el total de la base).

| Base de datos | Cadena | Fecha | Revis. | Relev. | Observación |
|---|---|---|---|---|---|
| OnePetro | O1 | 2026-10-27 | 10 | 0 | Todos pronostican **producción** (pozos, fracturamiento, EOR, *gas lift*), no costo unitario. |
| OnePetro | O2 | 2026-10-27 | 10 | 2 | *Forecast Techniques for Lifting Cost in Gas and Oil Onshore Fields* (SPE-69559-MS, 2001) pronostica el costo de levantamiento, pero es anterior al uso de boosting en la industria y el resumen no fue accesible. *Automated Lease Operating Statements for Cost Optimization…* (SPE ATCE 2020) automatiza estados de costos operativos con IA; no predice el costo de activos nuevos. |
| Semantic Scholar | S1 | 2026-10-27 | 10 | 3 | Hierarchical Shrinkage {cite}`agarwal2022hierarchical` (contrae hojas hacia nodos ancestros **del árbol**), CatBoost {cite}`prokhorenkova2018catboost` y contracción en una jerarquía de clases para texto (McCallum et al.). Ninguno usa una jerarquía del dominio dentro del boosting ni aborda costos petroleros. |
| Semantic Scholar | S2 | 2026-10-27 | 10 | 0 | Pronóstico de producción y de precios del crudo, y revisiones generales de métodos predictivos; ninguno estima costo unitario por activo. |
| arXiv | A1 | 2026-10-23 | 9 | 3 | CatBoost, codificación regularizada del objetivo {cite}`pargent2022regularized` y un estudio comparativo de codificadores: codifican el **objetivo** como preprocesamiento o con estadísticos ordenados; ninguno usa una jerarquía del dominio dentro del boosting. |
| Web | W1 | 2026-10-23 | 10 | 1 | {cite}`alkhammash2022gagb`: boosting optimizado con algoritmo genético para **pronosticar producción** de crudo, no costo. |
| Web | W2 | 2026-10-23 | 10 | 0 | Productividad de pozos, desempeño SAGD y recomendación de intervenciones; ninguno predice costo unitario. |
| Web | W3 | 2026-10-23 | 9 | 2 | CatBoost y una regresión **lineal** con categóricas jerárquicas por programación entera mixta (*Expert Systems with Applications*, 2022). |
| Web | W4 | 2026-10-23 | 9 | 0 | Pronóstico de precios y de producción, no de costo unitario. |
| Web | W5 | 2026-10-23 | 9 | 1 | {cite}`kaborogo2024proxy`: modelos *proxy* de CAPEX/DRILLEX/OPEX con **regresión lineal múltiple**; sin ML ni categorías. |

**Síntesis:** en 96 resultados revisados en cuatro bases, la literatura de la industria usa aprendizaje automático
sobre todo para **pronosticar producción**; los pocos trabajos sobre costos usan regresión lineal o automatización
de reportes. La literatura de métodos sí tiene codificaciones contraídas y contracción jerárquica, pero ninguna
combina las tres piezas del HGSB.

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
Al mejor de nuestro conocimiento y según la búsqueda descrita (OnePetro, Semantic Scholar, arXiv y motor web, 23 y
27 de octubre de 2026), **no se encontraron estudios que apliquen un gradient boosting con recodificación
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
