# %% [markdown]
# # 6. Validación estadística del modelo original
#
# **Pregunta (condición R3):** ¿el HGSB tiene un error menor que el de los benchmarks, con una diferencia
# estadísticamente significativa y no trivial?
#
# ## 6.1 Protocolo común
#
# - **Mismas particiones y semilla** que los 28 benchmarks: hold-out por activo (139 activos), 10 pliegues
#   externos y 3 internos GroupKFold, semilla 42. El HGSB pasó por la misma búsqueda bayesiana con el mismo
#   presupuesto de 30 evaluaciones (pareado estricto).
# - **Comparador principal:** la mejor combinación benchmark por RMSE medio del bucle externo. La definición de
#   superioridad (guía 6.1) exige además superar la mejor combinación de **cada** uno de los 7 modelos.
# - **Tipo de datos:** panel activo-año, de corte transversal entre activos. Por eso las pruebas principales son las
#   de la columna de corte transversal de la guía (Tabla 3): Wilcoxon por pliegue, t corregida de Nadeau-Bengio,
#   permutación pareada y bootstrap pareado BCa. Las dos últimas se hacen a nivel de **activo** (139 unidades
#   independientes), porque las filas de un mismo activo están correlacionadas (ρ = 0.75 en los residuos).
# - **Procedimiento global:** Model Confidence Set {cite}`hansen2011model` con el HGSB dentro del universo de 29
#   modelos, con bootstrap estacionario sobre las filas ordenadas por activo y año. Así los bloques respetan la
#   dependencia dentro del activo. Se complementa con Friedman + Nemenyi {cite}`demsar2006statistical` sobre los
#   10 pliegues.
# - **Comparaciones múltiples:** Holm sobre las cuatro pruebas dirigidas; se reportan el p crudo y el ajustado.
# - **Tamaño del efecto:** δ de Cliff (umbral de no despreciable |δ| ≥ 0.147 {cite}`romano2006appropriate`) y d de
#   Cohen pareado.

# %%
import sys, json, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src import config as C
from src.stats_tests import cd_diagram
from src.viz import set_style, SERIES

set_style()
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
ORIG = C.RESULTS_DIR / "original"
F = pd.read_csv(ORIG / "folds.csv", index_col=0)
fr = json.load(open(ORIG / "friedman.json"))
verdict = json.load(open(ORIG / "verdict.json", encoding="utf-8"))
best = fr["best_benchmark"]

# %% [markdown]
# ## 6.2 Tabla resumen

# %%
lb = pd.read_csv(C.ROOT / "artifacts" / "leaderboard.csv")
summary = pd.DataFrame({
    "Modelo": lb.label, "Optimizador": lb.method,
    "RMSE externo": [f"{a:.3f} ± {b:.3f}" for a, b in zip(lb.outer_rmse_mean, lb.outer_rmse_std)],
    "RMSE prueba": lb.test_rmse, "R² prueba": lb.test_r2,
    "Cobertura IC 90 %": lb.coverage_test, "Inferencia (ms/fila)": lb.infer_ms_per_row,
}).sort_values("RMSE prueba")
master = pd.read_csv(C.RESULTS_DIR / "master_table.csv")
times = master[master.scenario == "A_estructural"].set_index(["model", "method"])["fit_time_s"]
summary

# %% [markdown]
# ## 6.3 Supuestos
#
# - **Wilcoxon:** supone diferencias pareadas simétricas alrededor de la mediana; se revisa con la asimetría de
#   $\Delta$RMSE.
# - **t de Nadeau-Bengio** {cite}`nadeau2003inference`: supone normalidad aproximada de $\Delta$RMSE (Shapiro-Wilk)
#   y corrige la varianza por el solapamiento de los conjuntos de entrenamiento:
#   $\hat\sigma^2_{\mathrm{corr}} = \big(\tfrac1k + \tfrac{n_{\mathrm{test}}}{n_{\mathrm{train}}}\big)\hat\sigma^2$.
# - **Permutación y bootstrap por activo:** suponen activos intercambiables e independientes entre sí; la dependencia
#   dentro del activo se respeta al tomarlo como unidad.
# - **MCS:** supone estacionariedad de las diferencias de pérdida; el bootstrap estacionario por bloques
#   {cite}`politis1994stationary` respeta la dependencia en el orden activo-año.

# %%
d = F["HGSB|Bayesiana"] - F[best]
print(f"ΔRMSE por pliegue (HGSB − {best}):", np.round(d.values, 3))
print(f"media {d.mean():+.4f} · mediana {d.median():+.4f} · asimetría {stats.skew(d):.2f}")
print(f"Shapiro-Wilk p = {stats.shapiro(d).pvalue:.3f}  (H0: normalidad)")
print(f"Pliegues en que el HGSB es mejor: {(d < 0).sum()}/10")

# %% [markdown]
# ## 6.4 Pruebas de hipótesis

# %%
tests = pd.read_csv(ORIG / "tests.csv")
tests

# %% [markdown]
# ## 6.5 Procedimiento global: Friedman, Nemenyi y MCS

# %%
f8 = fr["friedman_8"]
print(f"Friedman (7 mejores combinaciones + HGSB): χ² = {f8['chi2']:.2f}, p = {f8['p_value']:.2e}, "
      f"W de Kendall = {f8['kendall_W']:.3f}; CD de Nemenyi = {fr['cd_8']:.2f}")
ranks = pd.Series(fr["ranks_8"]).sort_values()
fig, ax = plt.subplots(figsize=(9, 3.2))
cd_diagram(ranks, fr["cd_8"], ax=ax, title="Rango medio por pliegue (1 = mejor): 7 benchmarks + HGSB")
plt.show()

# %%
mcs = pd.read_csv(ORIG / "mcs.csv")
mcs[["model", "RMSE", "p_mcs", "in_mcs", "elim_order"]].head(15)

# %% [markdown]
# ## 6.6 Intervalos de confianza y robustez

# %%
seeds = pd.read_csv(ORIG / "seeds.csv")
kap = pd.read_csv(ORIG / "kappa.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].bar(seeds.seed.astype(str), seeds.test_rmse, color=SERIES[1])
axes[0].axhline(verdict["bench_test_rmse"], ls=":", color=SERIES[0], label=f"{best} (prueba)")
axes[0].set_ylim(seeds.test_rmse.min() * 0.95, max(seeds.test_rmse.max(), verdict["bench_test_rmse"]) * 1.03)
axes[0].set_title("Estabilidad ante semillas (RMSE prueba)"); axes[0].legend()
axes[1].semilogx(kap.kappa.replace(0, 0.1), kap.test_rmse, "o-", color=SERIES[1])
axes[1].set_xlabel("κ (0 graficado en 0.1)"); axes[1].set_title("Sensibilidad a κ (resto fijo)")
fig.tight_layout(); plt.show()
print(f"Semillas: {seeds.test_rmse.mean():.3f} ± {seeds.test_rmse.std(ddof=1):.3f}")
kap

# %% [markdown]
# La sensibilidad a κ es **descriptiva** (se mide en el hold-out con los demás hiperparámetros fijos): no se usó
# para elegir κ, que salió de la búsqueda anidada.
#
# ## 6.7 Veredicto

# %%
print(verdict["text"])

# %% [markdown]
# ### Discusión
#
# La superioridad se juzga con las tres condiciones de la guía (6.1): (1) menor RMSE medio externo que la mejor
# combinación benchmark y que la mejor de cada modelo; (2) al menos tres pruebas que rechacen H0 con p ajustado
# < 0.05 (< 0.10 para el MCS); y (3) |δ de Cliff| ≥ 0.147. Si alguna falla, el resultado se reporta como **no
# superior**. Las causas probables se discuten con la evidencia de este capítulo: poca potencia con 10 pliegues,
# un efecto propio del activo que ningún modelo observa (correlación de errores ≥ 0.57 entre todos los modelos) y
# la alta variación entre pliegues (d.e. ≈ 0.55 USD/Bbl, del mismo orden que las diferencias entre modelos).
