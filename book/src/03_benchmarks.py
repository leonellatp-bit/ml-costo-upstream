# %% [markdown]
# # 3. Modelos benchmark
#
# Este capítulo reúne los benchmarks del Entregable 2: diseño y optimización de hiperparámetros (3.1), optimización
# computacional (3.2), evaluación (3.3), comparación estadística (3.4) y la propuesta de modelos híbridos (3.5).
# Primero se documenta el **Bloque B de la guía**: la selección de la mejor combinación de cada modelo, su
# reentrenamiento y su serialización como artefactos para el dashboard.
#
# ## Criterio de selección
#
# - Para **cada uno de los 7 modelos** se elige el optimizador con menor **RMSE medio del bucle externo** (10
#   pliegues por activo). En regresión no hay técnica de balanceo, así que las 140 combinaciones de la guía se
#   reducen a 28 (7 modelos × 4 optimizadores).
# - Se identifica la **mejor combinación global** y se verifica que pertenece al conjunto de finalistas del
#   **Model Confidence Set** (α = 0.10).
# - Se agrega el modelo original (HGSB, capítulo 5) con el mismo protocolo.

# %%
import sys, json, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")
import pandas as pd
from src import config as C

pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
lb = pd.read_csv(C.ROOT / "artifacts" / "leaderboard.csv")
mt = pd.read_csv(C.RESULTS_DIR / "master_table.csv")
mtA = mt[mt.scenario == "A_estructural"]
sel = pd.DataFrame({
    "Modelo": lb.label, "Optimizador elegido": lb.method,
    "RMSE externo (media ± d.e.)": [f"{a:.3f} ± {b:.3f}" for a, b in zip(lb.outer_rmse_mean, lb.outer_rmse_std)],
    "MAE externo": [f"{a:.3f} ± {b:.3f}" for a, b in zip(lb.outer_mae_mean, lb.outer_mae_std)],
    "R² externo": [f"{a:.3f} ± {b:.3f}" for a, b in zip(lb.outer_r2_mean, lb.outer_r2_std)],
    "RMSE prueba": lb.test_rmse, "¿En el MCS?": lb.in_mcs.map({True: "sí", False: "no"}),
}).sort_values("RMSE externo (media ± d.e.)")
sel

# %% [markdown]
# **Motivo de cada selección.** El optimizador elegido es el de menor RMSE medio externo entre los cuatro. La tabla
# siguiente muestra la alternativa de cada modelo y la diferencia, para ver que la elección no depende de un
# empate.

# %%
alt = []
for m, g in mtA.groupby("model"):
    g = g.sort_values("outer_rmse_mean")
    alt.append({"Modelo": m, "Elegido": g.method.iloc[0], "RMSE elegido": g.outer_rmse_mean.iloc[0],
                "Segundo": g.method.iloc[1], "RMSE segundo": g.outer_rmse_mean.iloc[1],
                "Diferencia": g.outer_rmse_mean.iloc[1] - g.outer_rmse_mean.iloc[0]})
pd.DataFrame(alt).sort_values("RMSE elegido")

# %% [markdown]
# ## Reentrenamiento final y serialización
#
# Con los hiperparámetros fijados por la búsqueda sobre todo el conjunto de entrenamiento (train), cada pipeline completo
# (preprocesamiento + modelo) se reentrenó sobre los 556 activos de entrenamiento (train) y se serializó con
# `joblib.dump(..., compress=3)` en `artifacts/<Modelo>/model.joblib`. Junto a cada modelo hay un `metadata.json` con
# las versiones de las librerías, la semilla, los hiperparámetros, el esquema de entrada (nombres, tipos y rangos
# válidos) y las métricas del bucle externo y de prueba (`tools/build_artifacts.py`).
#
# **Prueba de ida y vuelta.** Se carga cada artefacto y se verifica que sus predicciones sobre el conjunto de
# prueba sean **exactamente iguales** (`np.array_equal`) a las del modelo en memoria y a las registradas en la CV
# anidada.

# %%
rows = []
for m in lb.model:
    md = json.load(open(C.ROOT / "artifacts" / m / "metadata.json", encoding="utf-8"))
    rows.append({"Modelo": md["label"], "Ida y vuelta idéntica": md["round_trip_identical"], "Tamaño (MB)": md["size_mb"],
                 "Inferencia (ms/fila)": md["inference_ms_per_row"], "Cobertura IC 90 % (prueba)": md["conformal_90"]["coverage_test"],
                 "scikit-learn": md["library_versions"]["scikit-learn"], "xgboost": md["library_versions"]["xgboost"]})
pd.DataFrame(rows)

# %% [markdown]
# Los artefactos suman unos 7 MB comprimidos: caben en el límite de GitHub y en la memoria del plan gratuito de
# Render sin reducir árboles ni usar Git LFS.
