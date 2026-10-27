"""Exploración: bosques con pesos aprendidos por árbol (prueba rápida en los 10 pliegues externos).

Cada pliegue usa los hiperparámetros de Random Forest elegidos en ese pliegue. Los árboles se
entrenan con bootstrap POR ACTIVO, así el error fuera de bolsa (OOB) de cada árbol se mide en
activos que no vio. Variantes:
  A  pesos uniformes 1/B (RF con bootstrap por activo)
  B  pesos globales  w_b ∝ exp(−β·z_b), z_b = error OOB estandarizado del árbol b
  C  pesos locales por región: error OOB del árbol en la región r contraído hacia su error global
       e_{b,r} = (n_r·MSE_{b,r} + κ·MSE_b)/(n_r + κ),  w_{b,r} ∝ exp(−β·z_{b,r})
  D  apilamiento convexo con ridge hacia el uniforme: min ‖y − P w‖² + μ‖w − 1/B‖², w ≥ 0, Σw = 1
β, κ y μ se eligen con el error OOB del ensamble dentro del entrenamiento (nunca con el pliegue externo).
"""
from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import optimize, stats
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, ".")
from src import config as C  # noqa: E402
from src.data import load_model_data, make_preprocessor, outer_cv, split_dev_test, xyg  # noqa: E402

SC = C.MAIN_SCENARIO


def fit_trees(Z, y, groups, params, seed):
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    trees, oob = [], []
    for b in range(int(params["n_estimators"])):
        pick = rng.integers(0, len(ug), len(ug))
        counts = np.bincount(pick, minlength=len(ug))
        w = counts[inv].astype(float)
        t = DecisionTreeRegressor(max_depth=params.get("max_depth"), max_features=params.get("max_features"),
                                  min_samples_leaf=params.get("min_samples_leaf", 1),
                                  random_state=int(rng.integers(1e9)))
        t.fit(Z[w > 0], y[w > 0], sample_weight=w[w > 0])
        trees.append(t)
        oob.append(w == 0)
    return trees, np.array(oob)          # oob: (B, n)


def oob_matrix(trees, Z, oob):
    P = np.full(oob.shape, np.nan)
    for b, t in enumerate(trees):
        P[b, oob[b]] = t.predict(Z[oob[b]])
    return P                              # (B, n), NaN donde el árbol vio la fila


def ens_oob_error(P, y, W):
    """Error del ensamble OOB: cada fila promedia solo los árboles que no la vieron. W: (B, n) pesos."""
    M = ~np.isnan(P)
    Wm = np.where(M, W, 0.0)
    num = np.nansum(np.where(M, P * Wm, 0.0), 0)
    den = Wm.sum(0)
    ok = den > 0
    return float(np.sqrt(np.mean((y[ok] - num[ok] / den[ok]) ** 2)))


def softmax_w(z, beta):
    a = -beta * z
    a -= a.max(axis=0, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=0, keepdims=True)


def run_fold(k, dev, test):
    X, y, g = xyg(dev, SC)
    reg = dev["RegionName"].to_numpy()
    tr, te = list(outer_cv().split(X, y, g))[k]
    params = pickle.load(open(f"results/cache/{SC}__RandomForest__Random__{k}.pkl", "rb"))["best_params"]
    prep = make_preprocessor(SC).fit(X.iloc[tr])
    Ztr, Zte = np.asarray(prep.transform(X.iloc[tr])), np.asarray(prep.transform(X.iloc[te]))
    ytr, yte, rtr, rte = y[tr], y[te], reg[tr], reg[te]
    trees, oob = fit_trees(Ztr, ytr, g[tr], params, seed=C.SEED + k)
    B = len(trees)
    Poob = oob_matrix(trees, Ztr, oob)
    Pte = np.stack([t.predict(Zte) for t in trees])          # (B, n_te)
    err2 = (Poob - ytr) ** 2
    mse_b = np.nanmean(err2, 1)
    z = (mse_b - mse_b.mean()) / (mse_b.std() + 1e-12)
    out = {"fold": k}
    rmse = lambda p: float(np.sqrt(np.mean((yte - p) ** 2)))  # noqa: E731
    # A uniforme
    out["A_uniforme"] = rmse(Pte.mean(0))
    # B global: elegir beta con OOB
    betas = [0, 0.5, 1, 2, 4, 8, 16, 32]
    bB = min(betas, key=lambda be: ens_oob_error(Poob, ytr, np.repeat(softmax_w(z[:, None], be), len(ytr), 1)))
    wB = softmax_w(z[:, None], bB)[:, 0]
    out["B_global"], out["beta_B"] = rmse(wB @ Pte), bB
    # C local por región con contracción hacia el error global
    regions = np.unique(rtr)
    best = (np.inf, None)
    for kap in [0.5, 1, 2, 5, 20]:
        Ez = {}
        for r in regions:
            m = rtr == r
            se = np.nansum(err2[:, m], 1)
            n = (~np.isnan(err2[:, m])).sum(1)
            e = (se + kap * mse_b) / (n + kap)
            Ez[r] = (e - e.mean()) / (e.std() + 1e-12)
        Zrow = np.stack([Ez[r] for r in rtr], 1)              # (B, n_tr)
        for be in betas:
            err = ens_oob_error(Poob, ytr, softmax_w(Zrow, be))
            if err < best[0]:
                best = (err, (kap, be, Ez))
    kap, be, Ez = best[1]
    Zte_w = np.stack([Ez.get(r, z) for r in rte], 1)           # región no vista -> pesos globales
    Wte = softmax_w(Zte_w, be)
    out["C_local_region"], out["kappa_C"], out["beta_C"] = rmse((Wte * Pte).sum(0)), kap, be
    # D apilamiento convexo con ridge hacia uniforme (sobre filas con predicción OOB de todos... usamos imputación por media OOB)
    Pfill = np.where(np.isnan(Poob), np.nanmean(Poob, 0, keepdims=True), Poob)
    u = np.full(B, 1 / B)

    def fitD(mu):
        f = lambda w: np.mean((ytr - w @ Pfill) ** 2) + mu * np.sum((w - u) ** 2) * B  # noqa: E731
        res = optimize.minimize(f, u, method="SLSQP", bounds=[(0, 1)] * B,
                                constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 200})
        return np.clip(res.x, 0, None) / np.clip(res.x, 0, None).sum()
    # mu por validación interna simple: mitad de los activos para ajustar, mitad para evaluar
    ug = np.unique(g[tr])
    rng = np.random.default_rng(k)
    half = np.isin(g[tr], rng.choice(ug, len(ug) // 2, replace=False))
    best_mu, best_e = None, np.inf
    for mu in [0.01, 0.1, 1.0]:
        f = lambda w: np.mean((ytr[half] - w @ Pfill[:, half]) ** 2) + mu * np.sum((w - u) ** 2) * B  # noqa: E731
        res = optimize.minimize(f, u, method="SLSQP", bounds=[(0, 1)] * B,
                                constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 200})
        w = np.clip(res.x, 0, None)
        w /= w.sum()
        e = np.mean((ytr[~half] - w @ Pfill[:, ~half]) ** 2)
        if e < best_e:
            best_mu, best_e = mu, e
    wD = fitD(best_mu)
    out["D_stacking"], out["mu_D"] = rmse(wD @ Pte), best_mu
    out["RF_sklearn"] = pickle.load(open(f"results/cache/{SC}__RandomForest__Random__{k}.pkl", "rb"))["metrics"]["rmse"]
    out["HGSB"] = pickle.load(open(f"results/cache/{SC}__HGSB__Bayesiana__{k}.pkl", "rb"))["metrics"]["rmse"]
    return out


if __name__ == "__main__":
    C.set_global_seed()
    dev, test = split_dev_test(load_model_data())
    res = Parallel(n_jobs=-1)(delayed(run_fold)(k, dev, test) for k in range(C.OUTER_FOLDS))
    df = pd.DataFrame(res).sort_values("fold")
    C.RESULTS_DIR.joinpath("weighted_forest").mkdir(exist_ok=True)
    df.to_csv(C.RESULTS_DIR / "weighted_forest" / "folds.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))
    cols = ["A_uniforme", "B_global", "C_local_region", "D_stacking", "RF_sklearn", "HGSB"]
    print(df[cols].mean().round(4))
    for c in ["B_global", "C_local_region", "D_stacking"]:
        d = df[c] - df["A_uniforme"]
        print(c, "vs A: Δ", round(d.mean(), 4), "mejor en", int((d < 0).sum()), "/10, Wilcoxon p",
              round(stats.wilcoxon(df[c], df["A_uniforme"], alternative="less").pvalue, 4))
        d2 = df[c] - df["RF_sklearn"]
        print("   vs RF sklearn: Δ", round(d2.mean(), 4), "mejor en", int((d2 < 0).sum()), "/10")
