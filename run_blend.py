"""Modelo híbrido RF + XGBoost: combinación convexa con peso óptimo (Bates-Granger).

    ŷ_H(x) = w · ŷ_RF(x) + (1 − w) · ŷ_XGB(x),   w ∈ [0, 1]

    w* = argmin_w Σ (y − ŷ_H)²  =  Σ e_X (e_X − e_R) / Σ (e_X − e_R)²   (recortado a [0, 1])

con e_R = y − ŷ_RF y e_X = y − ŷ_XGB. El peso se estima SIEMPRE con predicciones
fuera de muestra que no incluyen los datos donde luego se evalúa:

* CV externa (10 pliegues): para el pliegue k, w_k se estima con predicciones
  out-of-fold de un GroupKFold(3) interno sobre el entrenamiento externo k
  (hiperparámetros del pliegue k). Luego se aplica al pliegue externo k.
* Prueba: w_final se estima con las predicciones out-of-fold externas de todo el
  conjunto de entrenamiento (train) y se aplica a las predicciones de los modelos finales.
"""
from __future__ import annotations

import json
import pickle
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

sys.path.insert(0, ".")
from src import config as C  # noqa: E402
from src.data import inner_cv, load_model_data, outer_cv, split_dev_test, xyg  # noqa: E402
from src.models import MODELS  # noqa: E402
from src.nested_cv import regression_metrics  # noqa: E402
from src.stats_tests import (bootstrap_mean_ci, clark_west, cliffs_delta, diebold_mariano,  # noqa: E402
                             giacomini_white, holm, model_confidence_set)

SC = C.MAIN_SCENARIO
RF, XGB = ("RandomForest", "Random"), ("XGBoost", "Grid")
CACHE = C.RESULTS_DIR / "cache"
OUT = C.RESULTS_DIR / "blend"
OUT.mkdir(exist_ok=True)


def load(model, method, fold):
    with open(CACHE / f"{SC}__{model}__{method}__{fold}.pkl", "rb") as fh:
        return pickle.load(fh)


def opt_weight(y, p_rf, p_xgb) -> float:
    """Peso de RF que minimiza el ECM de la combinación convexa (forma cerrada, recortado a [0,1])."""
    e_r, e_x = y - p_rf, y - p_xgb
    diff = e_x - e_r
    return float(np.clip((e_x @ diff) / (diff @ diff), 0.0, 1.0))


def inner_oof(k, X, y, g, tr_idx):
    """Predicciones OOF (GroupKFold 3) de RF y XGB dentro del entrenamiento externo k."""
    pr, px = load(*RF, k)["best_params"], load(*XGB, k)["best_params"]
    Xk, yk, gk = X.iloc[tr_idx], y[tr_idx], g[tr_idx]
    o_r, o_x = np.zeros(len(yk)), np.zeros(len(yk))
    for a, b in inner_cv().split(Xk, yk, gk):
        o_r[b] = MODELS["RandomForest"].make_pipeline(pr, SC).fit(Xk.iloc[a], yk[a]).predict(Xk.iloc[b])
        o_x[b] = MODELS["XGBoost"].make_pipeline(px, SC).fit(Xk.iloc[a], yk[a]).predict(Xk.iloc[b])
    return k, yk, o_r, o_x


def main():
    C.set_global_seed()
    t0 = time.perf_counter()
    df = load_model_data()
    dev, test = split_dev_test(df)
    X, y, g = xyg(dev, SC)
    splits = list(outer_cv().split(X, y, g))

    # ---------- 1. CV externa anidada ----------
    inner = Parallel(n_jobs=-1)(delayed(inner_oof)(k, X, y, g, splits[k][0]) for k in range(C.OUTER_FOLDS))
    rows, oof = [], {"y": np.zeros(len(y)), "rf": np.zeros(len(y)), "xgb": np.zeros(len(y)), "hyb": np.zeros(len(y))}
    for k, yk, o_r, o_x in sorted(inner, key=lambda t: t[0]):
        w_k = opt_weight(yk, o_r, o_x)
        rr, rx = load(*RF, k), load(*XGB, k)
        assert np.array_equal(rr["index"], rx["index"])
        ye = rr["y"]
        ph = w_k * rr["pred"] + (1 - w_k) * rx["pred"]
        idx = rr["index"]
        oof["y"][idx], oof["rf"][idx], oof["xgb"][idx], oof["hyb"][idx] = ye, rr["pred"], rx["pred"], ph
        rho_in = np.corrcoef(yk - o_r, yk - o_x)[0, 1]
        rows.append({"fold": k, "w_RF": w_k, "rho_err_inner": rho_in,
                     "rmse_RF": regression_metrics(ye, rr["pred"])["rmse"],
                     "rmse_XGB": regression_metrics(ye, rx["pred"])["rmse"],
                     "rmse_Hibrido": regression_metrics(ye, ph)["rmse"],
                     "rmse_Promedio50": regression_metrics(ye, 0.5 * (rr["pred"] + rx["pred"]))["rmse"]})
    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / "folds.csv", index=False)

    fstats = {}
    for comp in ["RF", "XGB"]:
        d = folds["rmse_Hibrido"] - folds[f"rmse_{comp}"]
        w = stats.wilcoxon(folds["rmse_Hibrido"], folds[f"rmse_{comp}"], alternative="less")
        t = stats.ttest_rel(folds["rmse_Hibrido"], folds[f"rmse_{comp}"], alternative="less")
        fstats[comp] = {"mean_diff": float(d.mean()), "folds_better": int((d < 0).sum()),
                        "wilcoxon_p": float(w.pvalue), "ttest_p": float(t.pvalue),
                        "cliff": cliffs_delta(folds["rmse_Hibrido"], folds[f"rmse_{comp}"])}
    fr = stats.friedmanchisquare(folds["rmse_RF"], folds["rmse_XGB"], folds["rmse_Hibrido"])
    fstats["friedman"] = {"chi2": float(fr.statistic), "p": float(fr.pvalue)}

    # ---------- 2. Peso final con OOF externa del entrenamiento (train) ----------
    w_final = opt_weight(oof["y"], oof["rf"], oof["xgb"])
    er, ex = oof["y"] - oof["rf"], oof["y"] - oof["xgb"]
    s_r, s_x, s_rx = er.var(), ex.var(), np.cov(er, ex)[0, 1]
    theory = {"w_final": w_final, "rho_err_dev": float(np.corrcoef(er, ex)[0, 1]),
              "var_eR": float(s_r), "var_eX": float(s_x), "cov_eRX": float(s_rx),
              "w_bates_granger_unclipped": float((s_x - s_rx) / (s_r + s_x - 2 * s_rx)),
              "bias_RF": float(er.mean()), "bias_XGB": float(ex.mean()),
              "rmse_oof": {m: regression_metrics(oof["y"], oof[m])["rmse"] for m in ["rf", "xgb", "hyb"]}}

    # ---------- 3. Prueba (hold-out, 139 activos) ----------
    fr_, fx_ = load(*RF, "final"), load(*XGB, "final")
    yt = fr_["y"]
    p_rf, p_x = fr_["pred"], fx_["pred"]
    p_h = w_final * p_rf + (1 - w_final) * p_x
    p_50 = 0.5 * (p_rf + p_x)
    test_m = pd.DataFrame({n: regression_metrics(yt, p) for n, p in
                           [("Híbrido (w*)", p_h), ("Promedio 50/50", p_50), ("XGBoost", p_x), ("RandomForest", p_rf)]}).T
    test_m.to_csv(OUT / "test_metrics.csv")
    w_oracle = opt_weight(yt, p_rf, p_x)  # solo diagnóstico (usa la prueba): NO se usa para elegir

    e_h, e_r, e_x = yt - p_h, yt - p_rf, yt - p_x
    groups_t = test[C.GROUP].to_numpy()
    tests = []
    for name, e2, pb in [("XGBoost", e_x, p_x), ("RandomForest", e_r, p_rf)]:
        dm = diebold_mariano(e2, e_h)              # d>0 => híbrido mejor
        lo, hi = bootstrap_mean_ci(e_h ** 2 - e2 ** 2, B=4000)
        gw = giacomini_white(e2, e_h, conditional=True)
        cw = clark_west(yt, pb, p_h)               # el híbrido anida a cada componente (w=0 o w=1)
        # bootstrap por activo (clúster) de la diferencia de RMSE
        rng = np.random.default_rng(C.SEED)
        ug = np.unique(groups_t)
        pos = {gg: np.where(groups_t == gg)[0] for gg in ug}
        diffs = []
        for _ in range(4000):
            s = np.concatenate([pos[gg] for gg in rng.choice(ug, len(ug), replace=True)])
            diffs.append(np.sqrt(np.mean(e_h[s] ** 2)) - np.sqrt(np.mean(e2[s] ** 2)))
        diffs = np.array(diffs)
        tests.append({"vs": name, "dRMSE": float(np.sqrt(np.mean(e_h ** 2)) - np.sqrt(np.mean(e2 ** 2))),
                      "dm_hln": dm["dm_hln"], "dm_p": dm["p_value"],
                      "dMSE_ci_low": lo, "dMSE_ci_high": hi,
                      "gw": gw["gw"], "gw_p": gw["p_value"], "cw": cw["cw"], "cw_p": cw["p_value"],
                      "cluster_ci_low": float(np.quantile(diffs, .025)), "cluster_ci_high": float(np.quantile(diffs, .975)),
                      "cluster_p_one_sided": float((diffs >= 0).mean()),
                      "asset_better_share": float(np.mean([np.mean(e_h[pos[gg]] ** 2) < np.mean(e2[pos[gg]] ** 2) for gg in ug]))})
    tests = pd.DataFrame(tests)
    tests["dm_p_holm"] = holm(tests["dm_p"].to_numpy())
    tests.to_csv(OUT / "tests.csv", index=False)

    # MCS: las 28 combinaciones + híbrido
    losses = {}
    for m in MODELS:
        for meth in ["Grid", "Random", "Bayesiana", "Genetica"]:
            r = load(m, meth, "final")
            losses[f"{m}|{meth}"] = (r["y"] - r["pred"]) ** 2
    losses["Híbrido RF+XGB|w*"] = e_h ** 2
    mcs = model_confidence_set(pd.DataFrame(losses), alpha=0.10, B=2000)
    mcs["RMSE"] = np.sqrt(mcs["mean_loss"])
    mcs.to_csv(OUT / "mcs.csv", index=False)

    summary = {"folds": folds.to_dict("records"), "fold_stats": fstats, "theory": theory,
               "test": test_m.to_dict("index"), "w_oracle_test_diagnostic": w_oracle,
               "tests": tests.to_dict("records"),
               "mcs_top": mcs.head(12)[["model", "RMSE", "p_mcs", "in_mcs"]].to_dict("records"),
               "elapsed_s": time.perf_counter() - t0}
    with open(OUT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    pd.set_option("display.width", 200)
    print(folds.round(4)); print(json.dumps(fstats, indent=1, default=str)); print(json.dumps(theory, indent=1))
    print(test_m.round(4)); print("w oráculo (diagnóstico):", round(w_oracle, 3))
    print(tests.round(4).T); print(mcs.head(12)[["model", "RMSE", "p_mcs", "in_mcs"]])
    print("tiempo", round(time.perf_counter() - t0, 1))


if __name__ == "__main__":
    main()
