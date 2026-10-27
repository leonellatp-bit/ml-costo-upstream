"""Contenido teórico y conclusiones que muestra la página Modelos del dashboard.

Las fórmulas se escriben en LaTeX (dcc.Markdown con mathjax=True). Los valores de los
hiperparámetros se leen de artifacts/<Modelo>/metadata.json, así que siempre coinciden con
los artefactos desplegados.
"""
from __future__ import annotations

THEORY = {
    "Ridge": {
        "family": "Lineal regularizado (L2)",
        "md": r"""
$$\hat\beta=\arg\min_\beta \lVert y-X\beta\rVert_2^2+\lambda\lVert\beta\rVert_2^2=(X^\top X+\lambda I)^{-1}X^\top y$$

**Cómo aprende:** tiene solución cerrada. La penalización L2 contrae los coeficientes hacia cero sin anularlos:
sube un poco el sesgo y baja la varianza. Además, hace invertible $X^\top X+\lambda I$ aunque haya colinealidad
(país ↔ región).

**Supuestos:** relación lineal y aditiva, errores independientes y homocedásticos. Los datos violan los tres
(interacciones región × tipo de reserva, error persistente por activo, heterocedasticidad), por eso se queda en
R² ≈ 0.79.""",
        "hp": {"alpha": "λ: fuerza de la penalización L2"},
    },
    "Lasso": {
        "family": "Lineal regularizado (L1)",
        "md": r"""
$$\hat\beta=\arg\min_\beta \tfrac{1}{2n}\lVert y-X\beta\rVert_2^2+\alpha\lVert\beta\rVert_1$$

**Cómo aprende:** descenso coordenado con umbral suave $S(z,\alpha)=\operatorname{sign}(z)\max(|z|-\alpha,0)$.
La región L1 tiene esquinas sobre los ejes, así que varios coeficientes quedan exactamente en cero: hace
**selección de variables**. Comparte los supuestos y las limitaciones de Ridge.""",
        "hp": {"alpha": "α: fuerza de la penalización L1"},
    },
    "KNN": {
        "family": "Vecindarios (no paramétrico)",
        "md": r"""
$$\hat y(x)=\frac{\sum_{i\in N_k(x)} w_i\,y_i}{\sum_{i\in N_k(x)} w_i},\qquad w_i=\frac{1}{d(x,x_i)}$$

**Cómo aprende:** no estima parámetros; guarda los datos y promedia los $k$ vecinos más cercanos. Un $k$ pequeño
da bajo sesgo y alta varianza; uno grande, lo contrario. Requiere escalar las variables y sufre la maldición de
la dimensionalidad con 173 columnas. La distancia Manhattan ($p=1$) es más robusta en alta dimensión.""",
        "hp": {"n_neighbors": "k: número de vecinos", "weights": "ponderación de los vecinos",
               "p": "métrica de Minkowski (1 = Manhattan)"},
    },
    "SVR": {
        "family": "Kernel (márgenes)",
        "md": r"""
$$\min_{w,b,\xi}\ \tfrac12\lVert w\rVert^2+C\sum_i(\xi_i+\xi_i^*)\quad\text{s.a.}\quad |y_i-f(x_i)|\le\varepsilon+\xi_i^{(*)},\ \ \xi\ge 0$$

$$K(x,x')=\exp(-\gamma\lVert x-x'\rVert^2)$$

**Cómo aprende:** resuelve el problema dual (cuadrático); solo los vectores soporte definen $f$. La pérdida es
**ε-insensible**: los errores menores que ε no cuestan. $C$ penaliza las violaciones (C alto = menos
regularización) y γ fija el alcance del kernel. El costo de entrenamiento es entre $O(n^2)$ y $O(n^3)$
(exponente medido ≈ 2.0).""",
        "hp": {"C": "penalización de los errores fuera del tubo", "gamma": "γ: alcance del kernel RBF",
               "epsilon": "ε: ancho del tubo sin costo"},
    },
    "DecisionTree": {
        "family": "Árbol (CART)",
        "md": r"""
$$\min_{j,s}\ \sum_{x_i\in R_1(j,s)}(y_i-\bar y_{R_1})^2+\sum_{x_i\in R_2(j,s)}(y_i-\bar y_{R_2})^2$$

**Cómo aprende:** elige de forma codiciosa la variable $j$ y el corte $s$ que más reducen el error cuadrático;
cada hoja predice la media. Es interpretable pero **inestable**: un cambio pequeño en los datos cambia los cortes
(alta varianza). La profundidad y el tamaño mínimo de hoja controlan la complejidad.""",
        "hp": {"max_depth": "profundidad máxima", "min_samples_leaf": "filas mínimas por hoja",
               "min_samples_split": "filas mínimas para dividir"},
    },
    "RandomForest": {
        "family": "Ensamble por bagging",
        "md": r"""
$$\hat f(x)=\frac1B\sum_{b=1}^{B}T_b(x),\qquad \operatorname{Var}[\hat f]=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$$

**Cómo aprende:** entrena $B$ árboles profundos (bajo sesgo) sobre muestras bootstrap y, en cada nodo, con un
subconjunto aleatorio de variables. Promediar reduce el segundo término y el submuestreo de variables baja la
correlación ρ entre árboles. Más árboles no sobreajustan: el sesgo no cambia y la varianza solo baja.""",
        "hp": {"n_estimators": "B: número de árboles", "max_depth": "profundidad de cada árbol",
               "max_features": "fracción de variables por nodo (decorrelaciona)", "min_samples_leaf": "filas mínimas por hoja"},
    },
    "XGBoost": {
        "family": "Ensamble por boosting",
        "md": r"""
$$\mathcal{L}=\sum_i \ell(y_i,\hat y_i)+\sum_k\Big(\gamma T_k+\tfrac12\lambda\lVert w_k\rVert^2\Big),\qquad \hat y^{(t)}=\hat y^{(t-1)}+\eta f_t(x)$$

$$w_j^*=-\frac{G_j}{H_j+\lambda},\qquad \text{Ganancia}=\tfrac12\Big[\tfrac{G_L^2}{H_L+\lambda}+\tfrac{G_R^2}{H_R+\lambda}-\tfrac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Big]-\gamma$$

**Cómo aprende:** suma árboles en secuencia; cada uno corrige los residuos del anterior usando la expansión de
Taylor de segundo orden ($g_i$, $h_i$). Con pérdida cuadrática, $g_i=\hat y_i-y_i$ y $h_i=1$. Ataca el sesgo y se
regulariza con η (pasos pequeños), λ, γ y el submuestreo de filas y columnas.""",
        "hp": {"learning_rate": "η: tamaño de cada paso", "n_estimators": "número de árboles",
               "max_depth": "profundidad de cada árbol", "colsample_bytree": "fracción de columnas por árbol",
               "subsample": "fracción de filas por árbol", "min_child_weight": "hessiano mínimo por hoja",
               "reg_lambda": "λ: regularización L2 de las hojas"},
    },
    "HGSB": {
        "family": "Modelo original: boosting con contracción jerárquica",
        "md": r"""
$$\tilde e_k=\frac{S_k+\kappa\,\pi_k}{n_k+\kappa},\qquad \pi_k=\sum_j W_{kj}\,\tilde e_j^{\,\text{padre}}$$

**Cómo aprende:** es un *gradient boosting* programado en numpy. En cada iteración, cada categoría geográfica
(país, cuenca) se codifica con la suma $S_k$ de los residuos medios de sus $n_k$ **activos**, contraída hacia la
codificación de su categoría padre (región). Sobre esas codificaciones y las variables numéricas se ajusta un
árbol con la ganancia de segundo orden y hojas $w^*=\sum r/(H+\lambda)$.

**Sustento teórico:**

1. **Bayes empírico.** Si el residuo de cada activo $\sim N(\theta_k,\sigma^2)$ y $\theta_k\sim N(\pi_k,\tau^2)$,
   la media posterior es exactamente $\tilde e_k$ con $\kappa=\sigma^2/\tau^2$ (estimador tipo James-Stein): con
   pocos activos, el país se apoya en su región.
2. **Partición óptima (Fisher, 1958).** Para pérdida cuadrática, ordenar las categorías por su media y cortar en
   un umbral encuentra la mejor partición binaria: un barrido de $K$ cortes en lugar de $2^{K-1}$.
3. **Descenso monótono.** Con `subsample = 1`, la pérdida de entrenamiento no aumenta en ninguna iteración
   (verificado con una prueba unitaria).

Un país nunca visto hereda la codificación de su región. La ablación confirma el aporte de cada pieza: sin
jerarquía +0.08, sin contracción +0.09, con One-Hot +0.18 de RMSE externo.""",
        "hp": {"n_estimators": "número de árboles", "learning_rate": "η: tamaño de cada paso",
               "max_depth": "profundidad de cada árbol", "kappa": "κ: fuerza de la contracción hacia el padre",
               "min_child_weight": "filas mínimas por hoja", "reg_lambda": "λ: regularización de las hojas",
               "subsample": "fracción de filas por árbol", "colsample": "fracción de variables por árbol"},
    },
}

METHOD_MD = r"""
**Validación anidada por activo.** El bucle interno (GroupKFold de 3) elige los hiperparámetros con 30
evaluaciones; el bucle externo (GroupKFold de 10) mide el error del modelo ya optimizado con datos que no
participaron en la elección. Todas las particiones son por activo: así se mide el error en activos **nuevos**.

**Optimizadores.** *Grid* recorre una rejilla fija; *Random* muestrea al azar (Bergstra y Bengio, 2012: pocos
hiperparámetros importan mucho); la *Bayesiana* (Optuna-TPE) modela $l(x)=p(x\mid y\ \text{bueno})$ y
$g(x)=p(x\mid y\ \text{malo})$ y maximiza la mejora esperada, equivalente a maximizar $l(x)/g(x)$; el *genético*
evoluciona una población con selección por torneo, cruce BLX-α, mutación gaussiana y elitismo. Con 30
evaluaciones los cuatro llegan a una región buena (Friedman p = 0.89); la bayesiana es la más eficiente.

**Sesgo–varianza.** $E[(y-\hat f)^2]=\text{sesgo}^2+\text{varianza}+\text{ruido}$. Los lineales fallan por sesgo
(no capturan interacciones), el árbol único por varianza y los ensambles equilibran ambos. El ruido irreducible
incluye el efecto propio de cada activo, que ninguna variable observa.
"""

FINDINGS = [
    ("Sí se puede estimar el costo de un activo nuevo",
     "XGBoost y Random Forest logran RMSE 1.65–1.72 USD/Bbl y R² ≈ 0.92 en 139 activos nunca vistos, sin usar "
     "componentes del objetivo: 41 % menos error que el modelo base del Entregable 1."),
    ("La auditoría de fuga cambió la historia",
     "El objetivo es la suma exacta del costo en cabeza de pozo y el diferencial Brent. Sin ese diferencial, el R² "
     "del modelo base cae de 0.86 a 0.78: su desempeño estaba inflado."),
    ("La familia del modelo importa más que el optimizador",
     "Los ensambles superan a los lineales con efecto grande (δ de Cliff −0.74 a −0.76); los cuatro optimizadores "
     "empatan en error externo (Friedman p = 0.89)."),
    ("El límite está en la información, no en el algoritmo",
     "El error es persistente por activo (ρ = 0.75 entre años) y todos los modelos fallan en los mismos activos "
     "(correlación de errores ≥ 0.57). Combinar modelos no mejora: ganancia máxima de 0.2 %."),
    ("El modelo original es el mejor en promedio, pero no demostrado",
     "El HGSB tiene el menor RMSE de validación de los 29 modelos (1.93 frente a 2.01) y el mejor rango medio, pero "
     "ninguna de las 5 pruebas es significativa tras Holm y en prueba queda en 1.88."),
    ("La geografía explica el costo",
     "Región (≈ 2.2 USD/Bbl de efecto medio), año y cuenca dominan; ser no convencional suma +3 a +10 USD/Bbl."),
]

USES = [
    "Filtrar activos nuevos: costo estimado con intervalo del 90 % frente al Brent esperado; si el margen es menor "
    "que 2 × RMSE (~3.4 USD/Bbl), revisión detallada (regla propuesta, no validada en campo).",
    "Benchmark de OPEX: un activo que cuesta más de lo esperado año tras año tiene un efecto propio y es candidato "
    "a iniciativas de eficiencia.",
    "Escenarios: cambiar región, tipo de reserva, ubicación o año en el predictor y leer el aporte de cada factor.",
]

LIMITATIONS = [
    "Tamaño efectivo: 695 activos independientes (no 3.834 filas); 45 de 99 países tienen un solo activo.",
    "Representatividad: 96 % de activos convencionales; los no convencionales extremos (bitumen minado) tienen los "
    "mayores errores.",
    "Faltantes estructurales: Production no está disponible para África ni Asia Central.",
    "Supuestos: los residuos violan independencia (White p ≈ 10⁻²⁶ para heterocedasticidad; ρ = 0.75 entre años), "
    "así que los intervalos por observación subestiman la incertidumbre.",
    "Extrapolación: los árboles no predicen fuera del rango visto; los años posteriores a 2023 son extrapolación.",
    "Potencia estadística: con 10 pliegues, diferencias de ~4 % entre los mejores modelos no son demostrables.",
]

FUTURE = [
    "Modelar log(costo) o pérdidas relativas por la heterocedasticidad.",
    "Validación cruzada externa repetida (5 × 5) para ganar potencia en las comparaciones.",
    "HGSB con codificación por nodo y pérdida relativa, para no perder ajuste en los activos de costo extremo.",
    "Para pronóstico de activos existentes: usar el costo del año anterior o efectos aleatorios por activo.",
    "Más variables físicas si la fuente las ofrece (profundidad, grado API, corte de agua).",
]
