"""Optimización computacional: complejidad empírica, variantes eficientes,
multi-fidelidad, paralelización y perfilamiento (Sección 4 de la guía)."""
from __future__ import annotations

import cProfile
import io
import pstats
import time
import tracemalloc

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.decomposition import PCA
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Lasso, Ridge, SGDRegressor
from sklearn.model_selection import HalvingRandomSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR, LinearSVR
from xgboost import XGBRegressor

from . import config as C
from .data import inner_cv, make_preprocessor
from .models import MODELS
from .optimizers import _distribution


def _timeit(fn, repeats=3):
    ts = []
    out = None
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), out


def measure(estimator, Xtr, ytr, Xte, yte, repeats=3) -> dict:
    """Tiempo de entrenamiento e inferencia (mediana de ``repeats``), memoria pico y RMSE."""
    t_fit, _ = _timeit(lambda: estimator.fit(Xtr, ytr), repeats)
    t_pred, pred = _timeit(lambda: estimator.predict(Xte), repeats)
    tracemalloc.start()
    estimator.fit(Xtr, ytr)
    estimator.predict(Xte)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rmse = float(np.sqrt(np.mean((yte - pred) ** 2)))
    return {"fit_s": t_fit, "predict_s": t_pred, "peak_mem_MB": peak / 1e6, "rmse": rmse}


# ------------------------------------------------------------------
# 1) Escalamiento empírico con n
# ------------------------------------------------------------------
def scaling_experiment(X, y, params_by_model: dict, sizes=(250, 500, 1000, 2000, 3000),
                       repeats=3, seed=C.SEED) -> pd.DataFrame:
    """Tiempo de ajuste y de predicción frente a n (p fijo) para cada modelo.

    El preprocesamiento se aplica antes (fuera del cronómetro) para medir solo el
    algoritmo. Se predice siempre sobre las mismas 500 filas.
    """
    prep = make_preprocessor().fit(X)
    Z = prep.transform(X)
    rng = np.random.default_rng(seed)
    q_idx = rng.choice(len(Z), 500, replace=False)
    rows = []
    for name, params in params_by_model.items():
        spec = MODELS[name]
        for n in sizes:
            n = min(n, len(Z))
            idx = rng.choice(len(Z), n, replace=False)
            est = spec.builder(seed=seed, **{**spec.fixed, **params})
            t_fit, _ = _timeit(lambda: est.fit(Z[idx], y[idx]), repeats)
            t_pred, _ = _timeit(lambda: est.predict(Z[q_idx]), repeats)
            rows.append({"model": name, "n": n, "p": Z.shape[1], "fit_s": t_fit, "predict_s": t_pred})
    return pd.DataFrame(rows)


def scaling_exponents(df: pd.DataFrame) -> pd.DataFrame:
    """Pendiente de log(tiempo) vs log(n): exponente empírico b en t ∝ n^b."""
    out = []
    for name, g in df.groupby("model"):
        bf = np.polyfit(np.log(g["n"]), np.log(g["fit_s"]), 1)[0]
        bp = np.polyfit(np.log(g["n"]), np.log(g["predict_s"]), 1)[0]
        out.append({"model": name, "exp_fit": bf, "exp_predict": bp})
    return pd.DataFrame(out)


# ------------------------------------------------------------------
# 2) Estándar vs optimizado
# ------------------------------------------------------------------
def standard_vs_optimized(Xtr, ytr, Xte, yte, best_params: dict, seed=C.SEED) -> pd.DataFrame:
    """Tabla comparativa de variantes algorítmicas (mismos datos preprocesados)."""
    prep = make_preprocessor().fit(Xtr)
    Ztr, Zte = prep.transform(Xtr), prep.transform(Xte)
    k = int(best_params.get("KNN", {}).get("n_neighbors", 10))
    w = best_params.get("KNN", {}).get("weights", "distance")
    pk = int(best_params.get("KNN", {}).get("p", 2))
    xgb_p = {kk: v for kk, v in best_params.get("XGBoost", {}).items() if kk != "n_estimators"}
    svr_p = best_params.get("SVR", {"C": 10, "gamma": 0.01, "epsilon": 0.1})
    alpha_r = best_params.get("Ridge", {}).get("alpha", 1.0)
    alpha_l = best_params.get("Lasso", {}).get("alpha", 0.01)

    variants = [
        ("KNN", "Estándar: fuerza bruta", "O(1) / O(n·p) por consulta",
         KNeighborsRegressor(k, weights=w, p=pk, algorithm="brute")),
        ("KNN", "KD-Tree", "O(n·p·log n) / ~O(p·log n) en baja dim.",
         KNeighborsRegressor(k, weights=w, p=pk, algorithm="kd_tree")),
        ("KNN", "Ball-Tree", "O(n·p·log n) / ~O(p·log n)",
         KNeighborsRegressor(k, weights=w, p=pk, algorithm="ball_tree")),
        ("KNN", "Aproximado: PCA(20) + KD-Tree", "O(n·p·d) + O(n·d·log n) / ~O(d·log n)",
         Pipeline([("pca", PCA(20, random_state=seed)),
                   ("knn", KNeighborsRegressor(k, weights=w, p=pk, algorithm="kd_tree"))])),
        ("Ridge", "Estándar: Cholesky", "O(n·p² + p³)", Ridge(alpha_r, solver="cholesky")),
        ("Ridge", "SAGA", "O(n·p) por época", Ridge(alpha_r, solver="saga", max_iter=5000,
                                                   random_state=seed, tol=1e-4)),
        ("Lasso", "Estándar: descenso coordenado", "O(n·p) por iteración",
         Lasso(alpha_l, max_iter=20000, random_state=seed)),
        ("Lasso", "SGD (penalización L1)", "O(n·p) por época, online",
         SGDRegressor(penalty="l1", alpha=alpha_l / 10, max_iter=2000, random_state=seed, tol=1e-5)),
        ("XGBoost", "Estándar: tree_method='exact', 600 árboles", "O(T·n·p·log n)",
         XGBRegressor(**{**xgb_p, "n_estimators": 600}, tree_method="exact", random_state=seed, n_jobs=1)),
        ("XGBoost", "tree_method='hist', 600 árboles", "O(T·n·p) (histogramas)",
         XGBRegressor(**{**xgb_p, "n_estimators": 600}, tree_method="hist", random_state=seed, n_jobs=1)),
        ("XGBoost", "'hist' + early stopping (val. 15 %), lr ajustado", "O(T*·n·p), T* < T",
         ("early_stopping", None)),
        ("XGBoost", "'hist' + early stopping (val. 15 %), lr = 0.05", "O(T*·n·p), T* < T",
         ("early_stopping", 0.05)),
        ("SVR", "Estándar: SVR kernel RBF (libsvm)", "O(n²·p)–O(n³)",
         SVR(kernel="rbf", **svr_p)),
        ("SVR", "LinearSVR (liblinear)", "O(n·p) por iteración",
         LinearSVR(C=1.0, epsilon=0.0, max_iter=20000, random_state=seed, dual="auto")),
        ("SVR", "SGDRegressor (pérdida ε-insensible)", "O(n·p) por época",
         SGDRegressor(loss="epsilon_insensitive", epsilon=0.1, alpha=1e-4, max_iter=2000,
                      random_state=seed)),
        ("SVR", "Nyström (300 comp.) + LinearSVR", "O(n·m·p + n·m), m=300",
         Pipeline([("nys", Nystroem(gamma=svr_p.get("gamma", 0.01), n_components=300, random_state=seed)),
                   ("svr", LinearSVR(C=svr_p.get("C", 10), epsilon=svr_p.get("epsilon", 0.1),
                                     max_iter=20000, random_state=seed, dual="auto"))])),
    ]
    rows = []
    for model, variant, cplx, est in variants:
        if isinstance(est, tuple):  # early stopping necesita un conjunto de validación
            rng = np.random.default_rng(seed)
            val = rng.random(len(Ztr)) < 0.15
            lr = est[1] or xgb_p.get("learning_rate", 0.05)
            est = XGBRegressor(**{**xgb_p, "learning_rate": lr, "n_estimators": 2000}, tree_method="hist",
                               early_stopping_rounds=50, random_state=seed, n_jobs=1)

            class _ES:
                def fit(self_, A, b):
                    est.fit(A[~val], b[~val], eval_set=[(A[val], b[val])], verbose=False)
                    return self_

                def predict(self_, A):
                    return est.predict(A)

            r = measure(_ES(), Ztr, ytr, Zte, yte)
            r["n_trees"] = int(est.best_iteration + 1)
        else:
            r = measure(est, Ztr, ytr, Zte, yte)
        rows.append({"model": model, "variant": variant, "complexity": cplx, **r})
    out = pd.DataFrame(rows)
    base = out.groupby("model")["fit_s"].transform("first")
    base_p = out.groupby("model")["predict_s"].transform("first")
    base_r = out.groupby("model")["rmse"].transform("first")
    out["speedup_fit"] = base / out["fit_s"]
    out["speedup_predict"] = base_p / out["predict_s"]
    out["delta_rmse"] = out["rmse"] - base_r
    return out


# ------------------------------------------------------------------
# 3) Multi-fidelidad: Successive Halving
# ------------------------------------------------------------------
def successive_halving_vs_random(X, y, groups, model: str, n_candidates: int = 27,
                                 min_resources: int = 20, max_resources: int = 540,
                                 factor: int = 3, seed=C.SEED) -> pd.DataFrame:
    """Compara RandomizedSearchCV (fidelidad completa) con HalvingRandomSearchCV.

    Fidelidad = número de árboles (``n_estimators``). Con η = 3 cada ronda
    conserva el tercio superior de configuraciones y triplica su presupuesto
    de árboles: 27 → 9 → 3 → 1 candidatos con 20 → 60 → 180 → 540 árboles.
    """
    spec = MODELS[model]
    dists = {f"model__{p.name}": _distribution(p) for p in spec.space if p.name != "n_estimators"}
    pipe = spec.make_pipeline({"n_estimators": max_resources}, C.MAIN_SCENARIO, seed)
    rows = []
    rs = RandomizedSearchCV(pipe, dists, n_iter=n_candidates, scoring=C.SCORING, cv=inner_cv(),
                            random_state=seed, n_jobs=1, refit=False)
    t = time.perf_counter()
    rs.fit(X, y, groups=groups)
    rows.append({"model": model, "method": "Random (fidelidad completa)", "time_s": time.perf_counter() - t,
                 "n_candidates": n_candidates, "best_rmse": -rs.best_score_,
                 "total_trees_fitted": n_candidates * max_resources * C.INNER_FOLDS})
    hs = HalvingRandomSearchCV(pipe, dists, n_candidates=n_candidates, factor=factor,
                               resource="model__n_estimators", min_resources=min_resources,
                               max_resources=max_resources, scoring=C.SCORING, cv=inner_cv(),
                               random_state=seed, n_jobs=1, refit=False)
    t = time.perf_counter()
    hs.fit(X, y, groups=groups)
    cvr = pd.DataFrame(hs.cv_results_)
    trees = int((cvr["n_resources"] * C.INNER_FOLDS).sum())
    last = cvr[cvr["iter"] == cvr["iter"].max()]
    rows.append({"model": model, "method": f"Successive Halving (η={factor})",
                 "time_s": time.perf_counter() - t, "n_candidates": n_candidates,
                 "best_rmse": float(-last["mean_test_score"].max()), "total_trees_fitted": trees,
                 "rounds": int(hs.n_iterations_),
                 "candidates_per_round": list(map(int, hs.n_candidates_)),
                 "resources_per_round": list(map(int, hs.n_resources_))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 4) Paralelización
# ------------------------------------------------------------------
def parallel_speedup(X, y, groups, model: str = "RandomForest", params=None, seed=C.SEED,
                     n_folds: int = 6) -> pd.DataFrame:
    """Speedup de la validación cruzada con joblib: secuencial vs loky vs threading."""
    from sklearn.model_selection import GroupKFold
    import os

    pipe = MODELS[model].make_pipeline(params or {}, C.MAIN_SCENARIO, seed)
    cv = GroupKFold(n_folds)
    rows = []
    ncpu = os.cpu_count() or 1
    configs = [("secuencial", 1, "loky"), ("loky", ncpu, "loky"), ("threading", ncpu, "threading")]
    base = None
    for name, nj, backend in configs:
        with parallel_backend(backend, n_jobs=nj):
            t, _ = _timeit(lambda: cross_val_score(pipe, X, y, groups=groups, cv=cv,
                                                   scoring=C.SCORING, n_jobs=nj), 2)
        base = base or t
        rows.append({"model": model, "backend": name, "n_jobs": nj, "time_s": t, "speedup": base / t})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 5) Perfilamiento
# ------------------------------------------------------------------
def profile_search(X, y, groups, model: str = "RandomForest", method: str = "Bayesiana",
                   budget: int = 5, top: int = 15) -> tuple[pd.DataFrame, float]:
    """cProfile de una búsqueda: funciones con mayor tiempo acumulado + memoria pico."""
    from .optimizers import SEARCHERS

    pr = cProfile.Profile()
    tracemalloc.start()
    pr.enable()
    SEARCHERS[method](MODELS[model], X, y, groups, C.MAIN_SCENARIO, budget, C.SEED)
    pr.disable()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    s = io.StringIO()
    st = pstats.Stats(pr, stream=s)
    rows = []
    for (file, line, func), (cc, nc, tt, ct, _) in st.stats.items():
        rows.append({"function": f"{func} ({file.split('/')[-1]}:{line})", "ncalls": nc,
                     "tottime_s": tt, "cumtime_s": ct})
    df = pd.DataFrame(rows).sort_values("tottime_s", ascending=False).head(top).reset_index(drop=True)
    return df, peak / 1e6
