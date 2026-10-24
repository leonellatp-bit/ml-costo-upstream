"""Validación cruzada anidada combinatoria (modelo × optimizador) con registro estructurado.

Esquema
-------
- Hold-out por activo (20 %) reservado para la evaluación final.
- Sobre el 80 % de entrenamiento (train):
    * bucle externo: GroupKFold(10) por activo -> estimación de generalización;
    * bucle interno: GroupKFold(3) por activo dentro de cada entrenamiento
      externo -> búsqueda de hiperparámetros (Grid, Random, Bayes, GA).
- Las métricas reportadas salen del bucle externo, nunca del óptimo interno.
- Tras la CV anidada se repite la búsqueda sobre todo el conjunto de entrenamiento (train)
  ("final") para obtener el modelo que se evalúa en el hold-out.

Cada tarea (escenario, modelo, método, pliegue) se guarda en disco al terminar,
de modo que el experimento puede reanudarse si se interrumpe.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

from . import config as C
from .data import outer_cv, xyg
from .models import MODELS
from .optimizers import SEARCHERS

CACHE_DIR = C.RESULTS_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def regression_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE, MAPE (%) y R²."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(100 * mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _task_path(scenario, model, method, fold) -> Path:
    return CACHE_DIR / f"{scenario}__{model}__{method}__{fold}.pkl"


def run_task(dev: pd.DataFrame, test: pd.DataFrame, scenario: str, model: str, method: str,
             fold, budget: int = C.BUDGET, seed: int = C.SEED) -> dict:
    """Ejecuta una búsqueda y evalúa el mejor modelo.

    Parameters
    ----------
    fold : int or "final"
        Índice del pliegue externo, o "final" para la búsqueda sobre todo el
        conjunto de entrenamiento (train) (evaluada en el hold-out).

    Returns
    -------
    dict
        Mejores hiperparámetros, métricas externas, tiempos e historial.
    """
    path = _task_path(scenario, model, method, fold)
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)

    spec = MODELS[model]
    X, y, g = xyg(dev, scenario)
    if fold == "final":
        tr_idx, X_ev, y_ev = np.arange(len(dev)), *xyg(test, scenario)[:2]
        ev_index = test.index.to_numpy()
    else:
        splits = list(outer_cv().split(X, y, g))
        tr_idx, te_idx = splits[fold]
        X_ev, y_ev = X.iloc[te_idx], y[te_idx]
        ev_index = te_idx

    t0 = time.perf_counter()
    res = SEARCHERS[method](spec, X.iloc[tr_idx], y[tr_idx], g[tr_idx], scenario, budget, seed)
    t_search = time.perf_counter() - t0

    pipe = spec.make_pipeline(res["best_params"], scenario, seed)
    t1 = time.perf_counter()
    pipe.fit(X.iloc[tr_idx], y[tr_idx])
    t_fit = time.perf_counter() - t1
    t2 = time.perf_counter()
    pred = pipe.predict(X_ev)
    t_pred = time.perf_counter() - t2

    out = {
        "scenario": scenario, "model": model, "method": method, "fold": fold, "seed": seed,
        "best_params": res["best_params"], "best_inner_rmse": res["best_inner_rmse"],
        "n_evals": res["n_evals"], "search_time": t_search, "fit_time": t_fit,
        "predict_time": t_pred, "history": res["history"], "extra": res["extra"],
        "metrics": regression_metrics(y_ev, pred),
        "pred": pred.astype(float), "y": np.asarray(y_ev, float), "index": np.asarray(ev_index),
    }
    if fold == "final":
        with open(C.MODELS_DIR / f"{scenario}__{model}__{method}.pkl", "wb") as fh:
            pickle.dump(pipe, fh)
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return out


def run_experiment(dev, test, scenario: str, models=None, methods=None, n_jobs=C.N_JOBS,
                   budget=C.BUDGET, seed=C.SEED, verbose=10) -> list[dict]:
    """Corre todas las combinaciones modelo × método × pliegue en paralelo (joblib/loky).

    El paralelismo es entre tareas (grano grueso); dentro de cada tarea todo
    corre con ``n_jobs=1`` para evitar sobresuscripción de núcleos.
    """
    models = models or list(MODELS)
    methods = methods or list(SEARCHERS)
    folds = list(range(C.OUTER_FOLDS)) + ["final"]
    # tareas costosas primero para balancear la carga
    cost = {"RandomForest": 6, "SVR": 4, "XGBoost": 3}
    tasks = [(m, me, f) for m in models for me in methods for f in folds]
    tasks.sort(key=lambda t: -cost.get(t[0], 1))
    return Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(
        delayed(run_task)(dev, test, scenario, m, me, f, budget, seed) for m, me, f in tasks
    )


def collect(scenario: str | None = None) -> list[dict]:
    """Lee todas las tareas guardadas en caché."""
    out = []
    for p in sorted(CACHE_DIR.glob("*.pkl")):
        with open(p, "rb") as fh:
            r = pickle.load(fh)
        if scenario is None or r["scenario"] == scenario:
            out.append(r)
    return out


def _jsonable(o):
    """Convierte escalares numpy a tipos nativos para JSON."""
    return o.item() if hasattr(o, "item") else str(o)


def master_table(results: list[dict]) -> pd.DataFrame:
    """Tabla maestra: una fila por combinación (escenario, modelo, método).

    Métricas del bucle externo como media ± desviación estándar entre pliegues,
    métricas del hold-out, tiempos, presupuesto e hiperparámetros finales.
    """
    rows = []
    df = pd.DataFrame(results)
    for (sc, mo, me), grp in df.groupby(["scenario", "model", "method"]):
        outer = grp[grp["fold"] != "final"]
        final = grp[grp["fold"] == "final"]
        row = {"scenario": sc, "model": mo, "method": me, "seed": int(outer["seed"].iloc[0]),
               "n_outer_folds": len(outer)}
        for k in ["rmse", "mae", "mape", "r2"]:
            vals = np.array([m[k] for m in outer["metrics"]])
            row[f"outer_{k}_mean"] = vals.mean()
            row[f"outer_{k}_std"] = vals.std(ddof=1)
        row["inner_rmse_mean"] = outer["best_inner_rmse"].mean()
        row["evals_per_search"] = outer["n_evals"].mean()
        row["total_evals"] = int(outer["n_evals"].sum() + (final["n_evals"].sum() if len(final) else 0))
        row["search_time_total_s"] = float(outer["search_time"].sum() + (final["search_time"].sum() if len(final) else 0))
        row["time_per_eval_s"] = float(outer["search_time"].sum() / outer["n_evals"].sum())
        if len(final):
            f = final.iloc[0]
            for k in ["rmse", "mae", "mape", "r2"]:
                row[f"test_{k}"] = f["metrics"][k]
            row["final_params"] = json.dumps(f["best_params"], default=_jsonable)
            row["fit_time_s"] = f["fit_time"]
            row["predict_time_s"] = f["predict_time"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["scenario", "outer_rmse_mean"]).reset_index(drop=True)


def fold_matrix(results: list[dict], scenario: str, metric: str = "rmse") -> pd.DataFrame:
    """Matriz pliegues externos × combinaciones (para Friedman/Nemenyi)."""
    rows = {}
    for r in results:
        if r["scenario"] != scenario or r["fold"] == "final":
            continue
        rows.setdefault(f"{r['model']}|{r['method']}", {})[r["fold"]] = r["metrics"][metric]
    return pd.DataFrame(rows).sort_index()


def anytime_table(results: list[dict]) -> pd.DataFrame:
    """Curvas anytime: mejor RMSE interno encontrado hasta la evaluación t (por pliegue)."""
    rows = []
    for r in results:
        if r["fold"] == "final":
            continue
        best = np.inf
        for h in r["history"]:
            best = min(best, h["rmse"])
            rows.append({"scenario": r["scenario"], "model": r["model"], "method": r["method"],
                         "fold": r["fold"], "eval": h["eval"], "rmse": h["rmse"],
                         "best_so_far": best, "elapsed": h["elapsed"]})
    return pd.DataFrame(rows)
