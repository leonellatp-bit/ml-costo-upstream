# %% [markdown]
# # 5b. Propuesta: modelos híbridos
#
# Random Forest y XGBoost quedaron empatados estadísticamente. La pregunta natural es si **combinarlos**
# produce un modelo mejor que cada uno. Se prueban tres mezclas, definidas antes de mirar la prueba:
#
# 1. **Híbrido RF + XGBoost** con peso óptimo de Bates y Granger (1969).
# 2. **RF + SVR**: la pareja fuerte con errores más distintos.
# 3. **Super Learner** (van der Laan, Polley y Hubbard, 2007): combinación convexa de los 7 modelos.
#
# ## Formulación
#
# Para dos modelos con errores $e_R = y - \hat y_R$ y $e_X = y - \hat y_X$:
#
# $$\hat y_H(x) = w\,\hat y_R(x) + (1-w)\,\hat y_X(x), \qquad w \in [0,1]$$
#
# $$\mathrm{MSE}(w) = w^2\sigma_R^2 + (1-w)^2\sigma_X^2 + 2w(1-w)\sigma_{RX}$$
#
# $$w^* = \frac{\sigma_X^2 - \sigma_{RX}}{\sigma_R^2 + \sigma_X^2 - 2\sigma_{RX}}, \qquad
# \mathrm{MSE}(w^*) = \frac{\sigma_R^2\sigma_X^2 - \sigma_{RX}^2}{\sigma_R^2 + \sigma_X^2 - 2\sigma_{RX}}$$
#
# La ganancia frente al mejor modelo individual depende de la correlación de errores $\rho$: si $\rho \to 1$,
# $\mathrm{MSE}(w^*) \to \min(\sigma_R^2, \sigma_X^2)$ y combinar no aporta.
#
# El Super Learner generaliza a $M$ modelos:
#
# $$\hat y_{SL}(x) = \sum_{m=1}^{M} w_m\,\hat y_m(x), \qquad
# \mathbf w = \arg\min_{\mathbf w \ge 0,\ \mathbf 1^\top \mathbf w = 1} \lVert \mathbf y - P\mathbf w \rVert^2$$
#
# donde $P$ contiene predicciones *out-of-fold* de cada modelo. Su desigualdad de oráculo garantiza que,
# asintóticamente, rinde tanto como la mejor combinación convexa.
#
# ## Protocolo sin fuga
#
# * **CV externa:** en cada pliegue externo $k$ los pesos se estiman con predicciones OOF de un
#   GroupKFold(3) interno sobre el entrenamiento externo $k$ (hiperparámetros del pliegue $k$) y se aplican al
#   pliegue externo $k$.
# * **Prueba:** los pesos se estiman con las predicciones OOF externas de todo el desarrollo y se aplican a las
#   predicciones de los modelos finales. La prueba nunca interviene en la estimación de pesos.

# %%
import sys, warnings, json, pickle
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config as C
from src.viz import set_style
import run_blend, run_blend2

C.set_global_seed()
set_style()
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
OUT = C.RESULTS_DIR / "blend"

# %% [markdown]
# ## 1. ¿Se equivocan los modelos en los mismos activos?
#
# Correlación de los errores *out-of-fold* externos (cada fila del desarrollo predicha una vez, sin haber sido
# vista) del mejor optimizador de cada modelo.

# %%
BEST = run_blend2.BEST
E = {}
for m, meth in BEST.items():
    parts = [pickle.load(open(C.RESULTS_DIR / "cache" / f"A_estructural__{m}__{meth}__{k}.pkl", "rb")) for k in range(10)]
    n = sum(len(p["y"]) for p in parts)
    e = np.zeros(n)
    for p in parts:
        e[p["index"]] = p["y"] - p["pred"]
    E[m] = e
E = pd.DataFrame(E)
corr = E.corr()
fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(corr, vmin=0.5, vmax=1, cmap="Blues"); ax.grid(False)
ax.set_xticks(range(7), corr.columns, rotation=45, ha="right"); ax.set_yticks(range(7), corr.columns)
for i in range(7):
    for j in range(7):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if corr.iloc[i, j] > 0.85 else "black")
fig.colorbar(im, ax=ax, shrink=.8, label="correlación de errores OOF")
ax.set_title("Correlación de errores entre modelos (desarrollo, OOF externo)")
fig.tight_layout(); fig.savefig(C.FIG_DIR / "hibridos_corr_errores.png", dpi=150)
plt.show()

# %% [markdown]
# Ningún par baja de 0.57: **todos los modelos fallan en los mismos activos**. RF y XGBoost tienen 0.96, la
# pareja más redundante; SVR es el compañero fuerte más distinto (0.86).
#
# ## 2. Híbrido RF + XGBoost

# %%
run_blend.main()

# %%
s1 = json.load(open(OUT / "summary.json", encoding="utf-8"))
th = s1["theory"]
sR, sX, sRX = th["var_eR"], th["var_eX"], th["cov_eRX"]
mse_min = (sR * sX - sRX ** 2) / (sR + sX - 2 * sRX)
print(f"ρ errores = {th['rho_err_dev']:.3f} · w* = {th['w_final']:.3f}")
print(f"MSE mínimo teórico = {mse_min:.3f} vs MSE RF = {sR:.3f} → ganancia máxima {100 * (1 - mse_min / sR):.2f} %")
print(f"w estimado en desarrollo = {th['w_final']:.2f}; w 'oráculo' en prueba (solo diagnóstico) = {s1['w_oracle_test_diagnostic']:.2f}")

# %% [markdown]
# Con $\rho = 0.96$ la ganancia máxima posible de la combinación es de apenas ~0.2 % del MSE. Además, el peso
# es inestable: el estimado en desarrollo (0.83) y el que habría sido óptimo en prueba (0.33) son muy distintos.
# Es el *forecast combination puzzle* (Smith y Wallis, 2009): cuando la ganancia potencial es pequeña, el error
# de estimar $w$ se la come.
#
# ## 3. RF + SVR y Super Learner

# %%
run_blend2.main()

# %% [markdown]
# ## 4. Resumen

# %%
f1 = pd.read_csv(OUT / "folds.csv"); f2 = pd.read_csv(OUT / "folds_mix2.csv")
t1 = pd.read_csv(OUT / "test_metrics.csv", index_col=0)
s2 = json.load(open(OUT / "summary_mix2.json", encoding="utf-8"))
t2 = pd.DataFrame(s2["test"]).T
res = pd.DataFrame({
    "RMSE CV externa": [f1.rmse_Hibrido.mean(), f2["rmse_RF+SVR"].mean(), f2.rmse_SuperLearner.mean(),
                        f1.rmse_RF.mean(), f1.rmse_XGB.mean()],
    "d.e. pliegues": [f1.rmse_Hibrido.std(), f2["rmse_RF+SVR"].std(), f2.rmse_SuperLearner.std(),
                      f1.rmse_RF.std(), f1.rmse_XGB.std()],
    "RMSE prueba": [t1.loc["Híbrido (w*)", "rmse"], t2.loc["RF+SVR", "rmse"], t2.loc["SuperLearner", "rmse"],
                    t1.loc["RandomForest", "rmse"], t1.loc["XGBoost", "rmse"]],
    "R² prueba": [t1.loc["Híbrido (w*)", "r2"], t2.loc["RF+SVR", "r2"], t2.loc["SuperLearner", "r2"],
                  t1.loc["RandomForest", "r2"], t1.loc["XGBoost", "r2"]],
}, index=["Híbrido RF+XGB", "RF+SVR", "Super Learner", "Random Forest", "XGBoost"])
res.to_csv(OUT / "resumen.csv")
res

# %%
tt = pd.concat([pd.read_csv(OUT / "tests.csv").assign(mezcla="Híbrido RF+XGB"), pd.read_csv(OUT / "tests_mix2.csv")])
tt[["mezcla", "vs", "dRMSE", "dm_hln", "dm_p", "dm_p_holm", "gw_p", "cw_p"]]

# %% [markdown]
# ## Conclusión
#
# * **Ninguna mezcla supera de forma significativa al mejor modelo individual.** En CV externa, RF + SVR logra
#   el RMSE medio más bajo del proyecto (1.992), pero frente a RF la diferencia no es significativa (gana en 7
#   de 10 pliegues; Wilcoxon p = 0.16; δ de Cliff despreciable). En prueba, ninguna mezcla difiere de XGBoost
#   (Diebold-Mariano p ≥ 0.39) y todas entran al MCS.
# * El único resultado significativo es que el híbrido RF + XGBoost mejora a RF en prueba (DM p = 0.014 con
#   Holm; bootstrap por activo [−0.069, −0.013]), pero no a XGBoost.
# * El Super Learner asignó 72 % a RF, 21 % a SVR y casi nada a XGBoost: para combinar importa la
#   **diversidad de errores** más que el desempeño individual.
# * **Explicación:** el error dominante es el efecto propio de cada activo, común a todos los modelos
#   (correlaciones de error ≥ 0.57). Combinar solo reduce la parte del error que los modelos no comparten. El
#   límite está en la información disponible, no en el algoritmo.
