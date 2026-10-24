"""Mezclas adicionales: (1) RF + SVR (pareja fuerte más diversa) y (2) Super Learner convexo de los 7 modelos.

    ŷ_SL(x) = Σ_m w_m ŷ_m(x),   w_m ≥ 0,  Σ w_m = 1
    w = argmin_w ‖y − P w‖²  (P = predicciones out-of-fold de cada modelo)

Igual que run_blend.py: en cada pliegue externo k, los pesos salen de predicciones OOF de un
GroupKFold(3) interno sobre el entrenamiento externo k (hiperparámetros del pliegue k);
para la prueba, los pesos salen de las predicciones OOF externas del entrenamiento (train).
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import optimize, stats

sys.path.insert(0, ".")
from src import config as C  # noqa: E402
from src.data import inner_cv, load_model_data, outer_cv, split_dev_test, xyg  # noqa: E402
from src.models import MODELS  # noqa: E402
from src.nested_cv import regression_metrics  # noqa: E402
from src.stats_tests import (bootstrap_mean_ci, clark_west, cliffs_delta, diebold_mariano,  # noqa: E402
                             giacomini_white, holm, model_confidence_set)

SC = C.MAIN_SCENARIO
BEST = {"RandomForest": "Random", "XGBoost": "Grid", "SVR": "Bayesiana", "KNN": "Bayesiana",
        "DecisionTree": "Genetica", "Ridge": "Random", "Lasso": "Bayesiana"}
NAMES = list(BEST)
MIXES = {"RF+SVR": ["RandomForest", "SVR"], "SuperLearner": NAMES}
CACHE, OUT = C.RESULTS_DIR / "cache", C.RESULTS_DIR / "blend"


def load(m, fold):
    with open(CACHE / f"{SC}__{m}__{BEST[m]}__{fold}.pkl", "rb") as fh:
        return pickle.load(fh)


def convex_weights(y, P) -> np.ndarray:
    """Pesos ≥ 0 que suman 1 y minimizan el ECM (SLSQP)."""
    M = P.shape[1]
    f = lambda w: np.mean((y - P @ w) ** 2)
    res = optimize.minimize(f, np.full(M, 1 / M), method="SLSQP", bounds=[(0, 1)] * M,
                            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                            options={"ftol": 1e-12, "maxiter": 500})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def inner_oof(k, m, X, y, g, tr_idx):
    p = load(m, k)["best_params"]
    Xk, yk, gk = X.iloc[tr_idx], y[tr_idx], g[tr_idx]
    o = np.zeros(len(yk))
    for a, b in inner_cv().split(Xk, yk, gk):
        o[b] = MODELS[m].make_pipeline(p, SC).fit(Xk.iloc[a], yk[a]).predict(Xk.iloc[b])
    return k, m, o


def main():
    C.set_global_seed()
    dev, test = split_dev_test(load_model_data())
    X, y, g = xyg(dev, SC)
    splits = list(outer_cv().split(X, y, g))
    res = Parallel(n_jobs=-1)(delayed(inner_oof)(k, m, X, y, g, splits[k][0])
                              for k in range(C.OUTER_FOLDS) for m in NAMES)
    inner = {(k, m): o for k, m, o in res}

    # ---------- CV externa ----------
    rows, wrows = [], []
    oofP = np.zeros((len(y), len(NAMES)))
    for k in range(C.OUTER_FOLDS):
        yk = y[splits[k][0]]
        outs = {m: load(m, k) for m in NAMES}
        ye, idx = outs["RandomForest"]["y"], outs["RandomForest"]["index"]
        oofP[idx] = np.column_stack([outs[m]["pred"] for m in NAMES])
        row = {"fold": k, **{f"rmse_{m}": outs[m]["metrics"]["rmse"] for m in ["RandomForest", "XGBoost", "SVR"]}}
        for mix, ms in MIXES.items():
            w = convex_weights(yk, np.column_stack([inner[(k, m)] for m in ms]))
            pe = np.column_stack([outs[m]["pred"] for m in ms]) @ w
            row[f"rmse_{mix}"] = regression_metrics(ye, pe)["rmse"]
            wrows.append({"fold": k, "mix": mix, **dict(zip(ms, w.round(3)))})
        rows.append(row)
    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / "folds_mix2.csv", index=False)
    wtab = pd.DataFrame(wrows)

    fstats = {}
    for mix in MIXES:
        for comp in ["RandomForest", "XGBoost"]:
            a, b = folds[f"rmse_{mix}"], folds[f"rmse_{comp}"]
            fstats[f"{mix} vs {comp}"] = {
                "mean_diff": float((a - b).mean()), "folds_better": int((a < b).sum()),
                "wilcoxon_p": float(stats.wilcoxon(a, b, alternative="less").pvalue),
                "ttest_p": float(stats.ttest_rel(a, b, alternative="less").pvalue),
                "cliff": cliffs_delta(a, b)}

    # ---------- Pesos finales + prueba ----------
    yt = load("RandomForest", "final")["y"]
    Pt = np.column_stack([load(m, "final")["pred"] for m in NAMES])
    groups_t = test[C.GROUP].to_numpy()
    ug = np.unique(groups_t)
    pos = {gg: np.where(groups_t == gg)[0] for gg in ug}
    test_rows, tests, w_final, preds = [], [], {}, {}
    for mix, ms in MIXES.items():
        cols = [NAMES.index(m) for m in ms]
        w = convex_weights(y, oofP[:, cols])
        w_final[mix] = dict(zip(ms, w.round(4)))
        preds[mix] = Pt[:, cols] @ w
        test_rows.append({"modelo": mix, **regression_metrics(yt, preds[mix])})
    for m in ["XGBoost", "RandomForest"]:
        preds[m] = Pt[:, NAMES.index(m)]
        test_rows.append({"modelo": m, **regression_metrics(yt, preds[m])})
    test_m = pd.DataFrame(test_rows).set_index("modelo")

    for mix in MIXES:
        e_h = yt - preds[mix]
        for comp in ["XGBoost", "RandomForest"]:
            e2 = yt - preds[comp]
            dm = diebold_mariano(e2, e_h)
            lo, hi = bootstrap_mean_ci(e_h ** 2 - e2 ** 2, B=4000)
            rng = np.random.default_rng(C.SEED)
            diffs = []
            for _ in range(4000):
                s = np.concatenate([pos[gg] for gg in rng.choice(ug, len(ug), replace=True)])
                diffs.append(np.sqrt(np.mean(e_h[s] ** 2)) - np.sqrt(np.mean(e2[s] ** 2)))
            diffs = np.array(diffs)
            tests.append({"mezcla": mix, "vs": comp,
                          "dRMSE": float(np.sqrt(np.mean(e_h ** 2)) - np.sqrt(np.mean(e2 ** 2))),
                          "dm_hln": dm["dm_hln"], "dm_p": dm["p_value"],
                          "gw_p": giacomini_white(e2, e_h)["p_value"],
                          "cw_p": clark_west(yt, preds[comp], preds[mix])["p_value"],
                          "cluster_ci": (round(float(np.quantile(diffs, .025)), 4), round(float(np.quantile(diffs, .975)), 4)),
                          "cluster_p": float((diffs >= 0).mean())})
    tests = pd.DataFrame(tests)
    tests["dm_p_holm"] = holm(tests["dm_p"].to_numpy())
    tests.to_csv(OUT / "tests_mix2.csv", index=False)

    losses = {}
    for m in MODELS:
        for meth in ["Grid", "Random", "Bayesiana", "Genetica"]:
            r = pickle.load(open(CACHE / f"{SC}__{m}__{meth}__final.pkl", "rb"))
            losses[f"{m}|{meth}"] = (r["y"] - r["pred"]) ** 2
    for mix in MIXES:
        losses[f"{mix}"] = (yt - preds[mix]) ** 2
    mcs = model_confidence_set(pd.DataFrame(losses), alpha=0.10, B=2000)
    mcs["RMSE"] = np.sqrt(mcs["mean_loss"])
    mcs.to_csv(OUT / "mcs_mix2.csv", index=False)

    with open(OUT / "summary_mix2.json", "w") as fh:
        json.dump({"fold_means": folds.mean().to_dict(), "fold_stats": fstats, "w_final": w_final,
                   "weights_by_fold": wtab.to_dict("records"), "test": test_m.to_dict("index"),
                   "tests": tests.to_dict("records")}, fh, indent=2, ensure_ascii=False, default=str)
    pd.set_option("display.width", 220)
    print(folds.round(3).to_string()); print(folds.mean().round(4))
    print(wtab.groupby("mix").mean().drop(columns="fold").round(3))
    print(json.dumps(fstats, indent=1, default=str)); print(w_final)
    print(test_m.round(4)); print(tests.round(4).T)
    print(mcs.head(10)[["model", "RMSE", "p_mcs", "in_mcs"]])


if __name__ == "__main__":
    main()
