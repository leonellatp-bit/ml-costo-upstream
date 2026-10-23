"""Construye artifacts/ y las tablas del dashboard (Bloque B de la guía).

Para cada uno de los 7 modelos base se elige la mejor combinación según el RMSE medio del
bucle externo; se añade el modelo original (HGSB). Cada pipeline final (reentrenado sobre
todo el conjunto de entrenamiento/desarrollo durante la CV anidada) se serializa con
joblib (compress=3) junto a un metadata.json, y se verifica con una prueba de ida y vuelta
que el artefacto produce exactamente las mismas predicciones que el modelo en memoria.

Ejecutar desde la raíz, después de run_nested.py, run_original.py y run_original_stats.py:
    python tools/build_artifacts.py
"""
from __future__ import annotations

import json
import pickle
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402
from src.data import load_model_data, split_dev_test  # noqa: E402
from src.evaluation import acf  # noqa: E402
from src.explain import conformal_interval, conformal_quantile, grouped_shapley  # noqa: E402
from src.nested_cv import regression_metrics  # noqa: E402

SC = C.MAIN_SCENARIO
ART = ROOT / "artifacts"
DASH = C.RESULTS_DIR / "dashboard"
ORIG = C.RESULTS_DIR / "original"
CACHE = C.RESULTS_DIR / "cache"
BASE = ["KNN", "Ridge", "Lasso", "DecisionTree", "RandomForest", "XGBoost", "SVR"]
LABELS = {"KNN": "KNN", "Ridge": "Ridge", "Lasso": "Lasso", "DecisionTree": "Árbol de decisión",
          "RandomForest": "Random Forest", "XGBoost": "XGBoost", "SVR": "SVR (RBF)", "HGSB": "HGSB · modelo original"}
FEATS = C.SCENARIOS[SC]["numeric"] + C.SCENARIOS[SC]["categorical"]


def versions():
    import sklearn
    import xgboost
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "joblib": joblib.__version__}


def main():
    ART.mkdir(exist_ok=True)
    DASH.mkdir(exist_ok=True)
    df = load_model_data()
    dev, test = split_dev_test(df)
    Xd, yd = dev[FEATS], dev[C.TARGET].to_numpy()
    Xt, yt = test[FEATS], test[C.TARGET].to_numpy()
    mt = pd.read_csv(C.RESULTS_DIR / "master_table.csv")
    mtA = mt[mt.scenario == SC]
    best = mtA.loc[mtA.groupby("model")["outer_rmse_mean"].idxmin()].set_index("model")
    mcs = pd.read_csv(ORIG / "mcs.csv").set_index("model")

    choice = {m: best.loc[m, "method"] for m in BASE}
    choice["HGSB"] = "Bayesiana"
    schema = {"Year": {"type": "int", "min": int(dev.Year.min()), "max": int(dev.Year.max())},
              "Production": {"type": "float", "min": 0.0, "max": float(dev.Production.max()), "nullable": True}}
    for c in C.SCENARIOS[SC]["categorical"]:
        schema[c] = {"type": "category", "values": sorted(dev[c].astype(str).unique())}

    bg = Xd.sample(15, random_state=C.SEED)
    bg.to_csv(DASH / "shap_background.csv", index=False)
    shap_rows = Xt.sample(min(200, len(Xt)), random_state=C.SEED)

    lb, conf, pred_rows, acf_rows, shap_g = [], {}, [], [], []
    for m, me in choice.items():
        res = [pickle.load(open(CACHE / f"{SC}__{m}__{me}__{k}.pkl", "rb")) for k in range(C.OUTER_FOLDS)]
        fin = pickle.load(open(CACHE / f"{SC}__{m}__{me}__final.pkl", "rb"))
        pipe = pickle.load(open(C.MODELS_DIR / f"{SC}__{m}__{me}.pkl", "rb"))
        # --- serialización + prueba de ida y vuelta
        d = ART / m
        d.mkdir(exist_ok=True)
        path = d / "model.joblib"
        joblib.dump(pipe, path, compress=3)
        mem = pipe.predict(Xt)
        back = joblib.load(path).predict(Xt)
        round_trip = bool(np.array_equal(mem, back))
        assert round_trip, f"La prueba de ida y vuelta falló para {m}"
        # Diferencia frente a las predicciones registradas en la CV anidada (KNN puede diferir levemente por
        # empates entre vecinos equidistantes con distancia Manhattan sobre columnas One-Hot)
        max_diff_cv = float(np.abs(mem - fin["pred"]).max())
        t = time.perf_counter()
        for _ in range(3):
            pipe.predict(Xt)
        infer = (time.perf_counter() - t) / 3 / len(Xt) * 1000
        # --- conformal (predicciones fuera de pliegue del desarrollo)
        oof = np.full(len(dev), np.nan)
        for r in res:
            oof[r["index"]] = r["pred"]
        q = conformal_quantile(yd, oof, alpha=0.10)
        lo, hi = conformal_interval(mem, q)
        cov = float(np.mean((yt >= lo) & (yt <= hi)))
        conf[m] = {"q": q, "alpha": 0.10, "coverage_test": cov, "method": "split conformal relativo |y−ŷ|/ŷ"}
        rm = [r["metrics"] for r in res]
        meta = {"model": m, "label": LABELS[m], "method": me, "scenario": SC, "seed": C.SEED,
                "hyperparameters": fin["best_params"], "library_versions": versions(),
                "input_schema": schema, "target": C.TARGET, "units": "USD/Bbl",
                "outer_cv": {k: {"mean": float(np.mean([x[k] for x in rm])), "sd": float(np.std([x[k] for x in rm], ddof=1))}
                             for k in ["rmse", "mae", "mape", "r2"]},
                "test": fin["metrics"], "conformal_90": conf[m], "round_trip_identical": round_trip, "max_abs_diff_vs_nested_cv": max_diff_cv,
                "size_mb": path.stat().st_size / 1e6, "inference_ms_per_row": infer,
                "trained_on": "conjunto de desarrollo completo (556 activos)", "created": time.strftime("%Y-%m-%d %H:%M")}
        json.dump(meta, open(d / "metadata.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
        key = f"{m}|{me}"
        lb.append({"model": m, "label": LABELS[m], "method": me,
                   **{f"outer_{k}_mean": meta["outer_cv"][k]["mean"] for k in ["rmse", "mae", "r2"]},
                   **{f"outer_{k}_std": meta["outer_cv"][k]["sd"] for k in ["rmse", "mae", "r2"]},
                   "test_rmse": fin["metrics"]["rmse"], "test_r2": fin["metrics"]["r2"], "coverage_test": cov,
                   "infer_ms_per_row": infer, "size_mb": meta["size_mb"],
                   "in_mcs": bool(mcs.loc[key, "in_mcs"]) if key in mcs.index else False})
        # --- predicciones para diagnóstico
        for split, frame, yy, pp in [("Entrenamiento", dev, yd, pipe.predict(Xd)), ("Validación", dev, yd, oof),
                                     ("Prueba", test, yt, mem)]:
            pred_rows.append(pd.DataFrame({"model": m, "split": split, "y": yy, "pred": pp, "Year": frame.Year.values,
                                           "RegionName": frame.RegionName.values}))
        for k, v in enumerate(acf(yt - mem, 20)):
            acf_rows.append({"model": m, "lag": k, "acf": v, "band": 1.96 / np.sqrt(len(yt))})
        # --- SHAP global por grupos
        phi, base = grouped_shapley(pipe.predict, shap_rows, bg)
        for pl in phi.columns:
            shap_g.append({"model": m, "player": pl, "mean_abs": float(phi[pl].abs().mean()), "base": base})
        print(f"{m:13s} {me:9s} round-trip OK  test RMSE {fin['metrics']['rmse']:.3f}  "
              f"IC90 cobertura {cov:.2%}  {meta['size_mb']:.1f} MB", flush=True)

    pd.DataFrame(lb).to_csv(ART / "leaderboard.csv", index=False)
    json.dump(conf, open(ART / "conformal.json", "w"), indent=1)
    pd.concat(pred_rows).to_csv(DASH / "predictions.csv", index=False)
    pd.DataFrame(acf_rows).to_csv(DASH / "residual_acf.csv", index=False)
    pd.DataFrame(shap_g).to_csv(DASH / "shap_global.csv", index=False)

    raw = pd.read_csv(C.DATA_PATH, sep=";")
    hier = raw.drop_duplicates(["RegionName", "CountryName", "ProductionElementName", "MemberOpec"])[
        ["RegionName", "CountryName", "ProductionElementName", "MemberOpec"]]
    json.dump({"hierarchy": hier.to_dict("records"), "schema": schema,
               "assets_per_country": raw.groupby("CountryName")[C.GROUP].nunique().to_dict()},
              open(DASH / "predictor_options.json", "w"), indent=1)
    print("Artefactos en", ART)


if __name__ == "__main__":
    main()
