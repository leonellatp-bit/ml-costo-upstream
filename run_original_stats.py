"""Validación estadística del modelo original (HGSB) frente a los benchmarks (guía, sección 6).

Requiere: run_nested.py (benchmarks) y run_original.py (HGSB) ya ejecutados.
Produce results/original/: folds.csv, tests.csv, mcs.csv, ablation.csv, convergence.csv,
seeds.csv, kappa.csv, friedman.json, verdict.json.

Pruebas (datos de corte transversal agrupados por activo; guía, Tabla 3, columna izquierda):
  1. Wilcoxon de rangos con signo sobre las diferencias de RMSE por pliegue externo (10 pliegues).
  2. t corregida de Nadeau y Bengio (2003) sobre las mismas diferencias (corrige la dependencia
     entre pliegues: Var·(1/k + n_test/n_train)).
  3. Permutación pareada (cambio de signo) de la diferencia de pérdida cuadrática media por
     activo en el hold-out (139 activos independientes).
  4. Bootstrap BCa por activo (conglomerado) de la diferencia de RMSE en el hold-out.
  Procedimiento global: Model Confidence Set (Hansen et al., 2011) con el HGSB incluido en el
  universo (28 combinaciones + HGSB), y Friedman + Nemenyi sobre los 10 pliegues.
Corrección por comparaciones múltiples: Holm (FWER) sobre las pruebas 1-4.
Tamaños del efecto: δ de Cliff (pliegues), d de Cohen pareado (pliegues y activos).
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
from src.data import load_model_data, outer_cv, split_dev_test  # noqa: E402
from src.models import MODELS  # noqa: E402
from src.original_model import HierarchicalShrinkageBoosting  # noqa: E402
from src.stats_tests import cliffs_delta, cohens_d_paired, friedman, holm, model_confidence_set, nemenyi  # noqa: E402

SC = C.MAIN_SCENARIO
CACHE = C.RESULTS_DIR / "cache"
OUT = C.RESULTS_DIR / "original"
OUT.mkdir(exist_ok=True)
METHODS = ["Grid", "Random", "Bayesiana", "Genetica"]
ORIG = "HGSB|Bayesiana"
B = 10_000


def load(model, method, fold):
    return pickle.load(open(CACHE / f"{SC}__{model}__{method}__{fold}.pkl", "rb"))


def fold_matrix():
    cols = {}
    for m in MODELS:
        for me in METHODS:
            cols[f"{m}|{me}"] = [load(m, me, k)["metrics"]["rmse"] for k in range(C.OUTER_FOLDS)]
    cols[ORIG] = [load("HGSB", "Bayesiana", k)["metrics"]["rmse"] for k in range(C.OUTER_FOLDS)]
    return pd.DataFrame(cols)


def nadeau_bengio(d, n_train, n_test):
    d = np.asarray(d, float)
    k = len(d)
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return float(t), float(stats.t.cdf(t, k - 1))  # unilateral H1: HGSB menor RMSE (d < 0)


def bca_cluster(e1, e2, groups, B=B, seed=C.SEED):
    """IC BCa 95 % de RMSE(e1) − RMSE(e2) remuestreando activos."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    G = len(ug)
    s1 = np.bincount(inv, weights=e1 ** 2, minlength=G)
    s2 = np.bincount(inv, weights=e2 ** 2, minlength=G)
    n = np.bincount(inv, minlength=G).astype(float)

    def stat(w):
        return np.sqrt((w @ s1) / (w @ n)) - np.sqrt((w @ s2) / (w @ n))

    theta = stat(np.ones(G))
    W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(B)]).astype(float)
    boot = np.sqrt((W @ s1) / (W @ n)) - np.sqrt((W @ s2) / (W @ n))
    z0 = stats.norm.ppf(np.clip((boot < theta).mean(), 1e-6, 1 - 1e-6))
    jack = np.array([stat(np.where(np.arange(G) == i, 0.0, 1.0)) for i in range(G)])
    jm = jack.mean()
    a = ((jm - jack) ** 3).sum() / (6 * (((jm - jack) ** 2).sum()) ** 1.5 + 1e-12)
    out = []
    for al in (0.025, 0.975):
        z = stats.norm.ppf(al)
        q = stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        out.append(float(np.quantile(boot, q)))
    p_one = float((boot >= 0).mean())  # unilateral: proporción de réplicas sin mejora
    return float(theta), out[0], out[1], p_one


def sign_flip(e1, e2, groups, B=B, seed=C.SEED):
    """Permutación pareada por activo: H0 E[L1 − L2] = 0; H1 unilateral L1 < L2."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    d = (np.bincount(inv, weights=e1 ** 2) - np.bincount(inv, weights=e2 ** 2)) / np.bincount(inv)
    obs = d.mean()
    flips = rng.choice([-1.0, 1.0], size=(B, len(d)))
    null = (flips * d).mean(1)
    return float(obs), float(((null <= obs).sum() + 1) / (B + 1)), d


def ablation_task(dev, k, name, kw):
    from src.data import outer_cv as _ocv
    feats = C.SCENARIOS[SC]["numeric"] + C.SCENARIOS[SC]["categorical"] + [C.GROUP]
    X, y, g = dev[feats], dev[C.TARGET].to_numpy(), dev[C.GROUP].to_numpy()
    tr, te = list(_ocv().split(X, y, g))[k]
    params = dict(load("HGSB", "Bayesiana", k)["best_params"])
    params.update({"oof_folds": 1, "n_bins": 32})
    params.update(kw)
    m = HierarchicalShrinkageBoosting(random_state=C.SEED, **params).fit(X.iloc[tr], y[tr])
    return name, k, float(np.sqrt(np.mean((y[te] - m.predict(X.iloc[te])) ** 2)))


def main():
    t0 = time.perf_counter()
    C.set_global_seed()
    dev, test = split_dev_test(load_model_data())
    feats = C.SCENARIOS[SC]["numeric"] + C.SCENARIOS[SC]["categorical"] + [C.GROUP]

    # ---------------------------------------------------------------- pliegues
    F = fold_matrix()
    F.to_csv(OUT / "folds.csv", index_label="fold")
    means = F.mean()
    bench = means.drop(ORIG)
    best_bench = bench.idxmin()
    best_per_model = {m: means[[f"{m}|{me}" for me in METHODS]].idxmin() for m in MODELS}
    fr_all = friedman(F)
    cd_all, _ = nemenyi(F.rank(axis=1).mean(), N=len(F))
    F8 = F[[best_per_model[m] for m in MODELS] + [ORIG]]
    fr8 = friedman(F8)
    ranks8 = F8.rank(axis=1).mean()
    cd8, nem8 = nemenyi(ranks8, N=len(F8))

    d = (F[ORIG] - F[best_bench]).to_numpy()
    n_dev = len(dev)
    n_te = n_dev / C.OUTER_FOLDS
    w = stats.wilcoxon(F[ORIG], F[best_bench], alternative="less")
    t_nb, p_nb = nadeau_bengio(d, n_dev - n_te, n_te)
    cliff, cliff_lbl = cliffs_delta(F[ORIG], F[best_bench])
    dz, dz_lbl = cohens_d_paired(d)

    # ---------------------------------------------------------------- hold-out
    rb = load(*best_bench.split("|"), "final")
    ro = load("HGSB", "Bayesiana", "final")
    yt = ro["y"]
    e_o, e_b = yt - ro["pred"], yt - rb["pred"]
    gt = test[C.GROUP].to_numpy()
    diff_rmse, lo, hi, p_bca = bca_cluster(e_o, e_b, gt)
    obs, p_perm, dasset = sign_flip(e_o, e_b, gt)
    d_asset, d_asset_lbl = cohens_d_paired(dasset)

    losses = {f"{m}|{me}": (load(m, me, "final")["y"] - load(m, me, "final")["pred"]) ** 2
              for m in MODELS for me in METHODS}
    losses[ORIG] = e_o ** 2
    mcs = model_confidence_set(pd.DataFrame(losses), alpha=0.10, B=2000)
    mcs["RMSE"] = np.sqrt(mcs["mean_loss"])
    mcs.to_csv(OUT / "mcs.csv", index=False)
    p_mcs_bench = float(mcs.set_index("model").loc[best_bench, "p_mcs"])
    p_mcs_orig = float(mcs.set_index("model").loc[ORIG, "p_mcs"])

    raw_p = np.array([w.pvalue, p_nb, p_perm, p_bca])
    adj = holm(raw_p)
    rows = [
        ["Wilcoxon (pliegues)", "Mediana de ΔRMSE por pliegue = 0", "Pares por pliegue; diferencias simétricas",
         f"W = {w.statistic:.1f}", raw_p[0], adj[0], f"δ de Cliff = {cliff:.3f} ({cliff_lbl})"],
        ["t corregida de Nadeau-Bengio", "E[ΔRMSE] = 0", "Normalidad aprox. de ΔRMSE; corrección por solapamiento",
         f"t = {t_nb:.3f}", raw_p[1], adj[1], f"d de Cohen = {dz:.3f} ({dz_lbl})"],
        ["Permutación pareada por activo (prueba)", "E[L_HGSB − L_bench] = 0 por activo",
         "Activos independientes; intercambiabilidad del signo", f"Δ pérdida = {obs:.3f}", raw_p[2], adj[2],
         f"d de Cohen (activos) = {d_asset:.3f} ({d_asset_lbl})"],
        ["Bootstrap BCa por activo (prueba)", "RMSE_HGSB − RMSE_bench = 0", "Remuestreo de activos (conglomerados)",
         f"ΔRMSE = {diff_rmse:.3f} [{lo:.3f}, {hi:.3f}]", raw_p[3], adj[3], f"ΔRMSE = {diff_rmse:+.3f} USD/Bbl"],
        ["MCS (global, α = 0.10)", "El benchmark pertenece al conjunto de los mejores", "Bootstrap estacionario (orden activo-año)",
         f"p_MCS bench = {p_mcs_bench:.3f}; p_MCS HGSB = {p_mcs_orig:.3f}", p_mcs_bench, np.nan,
         "—"],
    ]
    tests = pd.DataFrame(rows, columns=["Prueba", "H0", "Supuestos", "Estadístico", "p crudo", "p ajustado (Holm)",
                                        "Tamaño del efecto"])
    rejects = [adj[i] < 0.05 for i in range(4)] + [p_mcs_bench < 0.10]
    tests["Conclusión"] = ["Rechaza H0: HGSB mejor" if r else "No rechaza H0" for r in rejects]
    tests["p crudo"] = tests["p crudo"].map(lambda v: f"{v:.4f}")
    tests["p ajustado (Holm)"] = tests["p ajustado (Holm)"].map(lambda v: "—" if pd.isna(v) else f"{v:.4f}")
    tests.to_csv(OUT / "tests.csv", index=False)

    # ---------------------------------------------------------------- ablación
    variants = {"HGSB completo": {}, "Sin jerarquía (contracción hacia la media global)": {"use_hierarchy": False},
                "Sin contracción (κ = 0)": {"kappa": 0.0},
                "Codificación fuera de pliegue (K = 5)": {"oof_folds": 5},
                "Sin codificación (One-Hot)": {"encoding": "onehot"}}
    res = Parallel(n_jobs=-1)(delayed(ablation_task)(dev, k, n, kw) for n, kw in variants.items()
                              for k in range(C.OUTER_FOLDS))
    A = pd.DataFrame(res, columns=["variante", "fold", "rmse"]).pivot(index="fold", columns="variante", values="rmse")
    full = A["HGSB completo"]
    abl = []
    for v in variants:
        dlt = A[v] - full
        p = stats.wilcoxon(A[v], full).pvalue if v != "HGSB completo" else np.nan
        abl.append({"Variante": v, "RMSE externo": f"{A[v].mean():.3f} ± {A[v].std(ddof=1):.3f}",
                    "Δ vs completo": "—" if v == "HGSB completo" else f"{dlt.mean():+.3f}",
                    "Pliegues peores": "—" if v == "HGSB completo" else f"{int((dlt > 0).sum())}/10",
                    "p Wilcoxon": "—" if np.isnan(p) else f"{p:.3f}"})
    pd.DataFrame(abl).to_csv(OUT / "ablation.csv", index=False)
    A.to_csv(OUT / "ablation_folds.csv")

    # ---------------------------------------------------------------- convergencia
    final = pickle.load(open(C.MODELS_DIR / f"{SC}__HGSB__Bayesiana.pkl", "rb"))
    est = final.named_steps["model"]
    Xd, yd = dev[feats], dev[C.TARGET].to_numpy()
    Xt = test[feats]
    tr_curve = [float(np.sqrt(np.mean((yd - p) ** 2))) for p in est.staged_predict(Xd)]
    te_curve = [float(np.sqrt(np.mean((yt - p) ** 2))) for p in est.staged_predict(Xt)]
    conv = pd.DataFrame({"iteration": np.arange(1, len(tr_curve) + 1), "train_rmse": tr_curve, "test_rmse": te_curve})
    conv["rf_test_rmse"] = load("RandomForest", "Random", "final")["metrics"]["rmse"]
    conv["xgb_test_rmse"] = load("XGBoost", "Grid", "final")["metrics"]["rmse"]
    conv.to_csv(OUT / "convergence.csv", index=False)

    # ---------------------------------------------------------------- semillas y κ (robustez descriptiva)
    bp = dict(ro["best_params"])
    bp.update({"oof_folds": 1, "n_bins": 32})

    def fit_eval(seed=C.SEED, **kw):
        p = dict(bp)
        p.update(kw)
        m = HierarchicalShrinkageBoosting(random_state=seed, **p).fit(Xd, yd)
        return float(np.sqrt(np.mean((yt - m.predict(Xt)) ** 2)))

    seeds = Parallel(n_jobs=-1)(delayed(fit_eval)(s) for s in [1, 7, 42, 123, 2024])
    pd.DataFrame({"seed": [1, 7, 42, 123, 2024], "test_rmse": seeds}).to_csv(OUT / "seeds.csv", index=False)
    kgrid = [0.0, 0.5, 2.0, bp["kappa"], 10.0, 50.0]
    kap = Parallel(n_jobs=-1)(delayed(fit_eval)(C.SEED, kappa=k) for k in kgrid)
    pd.DataFrame({"kappa": kgrid, "test_rmse": kap}).to_csv(OUT / "kappa.csv", index=False)

    # ---------------------------------------------------------------- veredicto
    better_all = bool(means[ORIG] < bench.min())
    n_sig = int(sum(rejects))
    effect_ok = abs(cliff) >= 0.147
    superior = better_all and n_sig >= 3 and effect_ok
    js = lambda d: {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in d.items()}  # noqa: E731
    json.dump({"friedman_all": js(fr_all), "cd_all": cd_all, "friedman_8": js(fr8), "cd_8": cd8, "ranks_8": ranks8.to_dict(),
               "best_benchmark": best_bench, "best_per_model": best_per_model}, open(OUT / "friedman.json", "w"),
              indent=1, default=float)
    txt = (f"HGSB: RMSE externo {means[ORIG]:.3f} ± {F[ORIG].std(ddof=1):.3f} frente a {means[best_bench]:.3f} ± "
           f"{F[best_bench].std(ddof=1):.3f} del mejor benchmark ({best_bench}); en prueba {ro['metrics']['rmse']:.3f} vs "
           f"{rb['metrics']['rmse']:.3f}. {n_sig} de 5 pruebas rechazan H0 (Holm / MCS); δ de Cliff = {cliff:.3f}. ")
    txt += ("Cumple R3: es superior con significancia estadística." if superior else
            "No cumple R3: la diferencia no es estadísticamente significativa con las correcciones exigidas.")
    verdict = {"status": "superior" if superior else "no superior",
               "label": "R3 cumplido" if superior else "R3 no cumplido", "text": txt,
               "summary": f"Modelo original HGSB · veredicto: {'superior' if superior else 'no superior'} a los benchmarks.",
               "orig_outer_mean": float(means[ORIG]), "bench_outer_mean": float(means[best_bench]),
               "orig_test_rmse": float(ro["metrics"]["rmse"]), "bench_test_rmse": float(rb["metrics"]["rmse"]),
               "n_significant": n_sig, "cliff": float(cliff), "better_than_all_means": better_all,
               "seeds_test_rmse_mean": float(np.mean(seeds)), "seeds_test_rmse_sd": float(np.std(seeds, ddof=1))}
    json.dump(verdict, open(OUT / "verdict.json", "w"), indent=1, ensure_ascii=False)
    pd.set_option("display.width", 220)
    print(F.mean().sort_values().head(8).round(4))
    print(tests.to_string())
    print(pd.DataFrame(abl).to_string())
    print(mcs.head(10)[["model", "RMSE", "p_mcs", "in_mcs"]])
    print("semillas", np.round(seeds, 3), "kappa", dict(zip(kgrid, np.round(kap, 3))))
    print(txt)
    print(f"{(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
