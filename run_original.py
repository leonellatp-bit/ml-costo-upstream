"""Validación anidada del modelo original (HGSB) con el MISMO protocolo de los benchmarks.

* Mismas particiones: hold-out por activo (GroupShuffleSplit, semilla 42), bucle externo
  GroupKFold(10), bucle interno GroupKFold(3).
* Mismo presupuesto: 30 evaluaciones de Optimización Bayesiana (Optuna-TPE, 8 ensayos
  iniciales, semilla 42), igual que los benchmarks.
* Misma semilla y mismas métricas. La tabla de resultados se guarda en el mismo formato
  de caché que ``src/nested_cv.run_task`` para integrarse a la tabla maestra.

La única diferencia de entrada: el modelo recibe además la columna del identificador del
activo, que usa solo para contar activos (no filas) al contraer las categorías; nunca es
predictora (ver prueba unitaria ``test_group_column_not_used_at_predict``).

Uso:  python run_original.py            (≈ 1 h con 2 núcleos)
"""
from __future__ import annotations

import pickle
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline

sys.path.insert(0, ".")
from src import config as C  # noqa: E402
from src.data import load_model_data, outer_cv, split_dev_test  # noqa: E402
from src.models import ModelSpec, Param  # noqa: E402
from src.nested_cv import regression_metrics  # noqa: E402
from src.optimizers import bayesian_search  # noqa: E402
from src.original_model import HierarchicalShrinkageBoosting  # noqa: E402

SC = C.MAIN_SCENARIO
NAME, METHOD = "HGSB", "Bayesiana"
CACHE = C.RESULTS_DIR / "cache"


class OriginalSpec(ModelSpec):
    """El HGSB hace su propio preprocesamiento: el pipeline solo contiene el modelo."""

    def make_pipeline(self, params=None, scenario=SC, seed=C.SEED):
        kw = dict(self.fixed)
        kw.update(params or {})
        return Pipeline([("model", self.builder(seed=seed, **kw))])


def _hgsb(seed=None, **kw):
    return HierarchicalShrinkageBoosting(random_state=seed, **kw)


SPEC = OriginalSpec(
    name=NAME,
    builder=_hgsb,
    space=[
        Param("n_estimators", "int", 100, 800, priority=0),
        Param("learning_rate", "float", 0.01, 0.3, log=True, priority=0),
        Param("max_depth", "int", 3, 10, priority=1),
        Param("kappa", "float", 0.1, 50.0, log=True, priority=1),
        Param("min_child_weight", "float", 1.0, 20.0, log=True, priority=2),
        Param("reg_lambda", "float", 0.01, 10.0, log=True, priority=2),
        Param("subsample", "float", 0.5, 1.0, priority=3),
        Param("colsample", "float", 0.5, 1.0, priority=3),
    ],
    fixed={"oof_folds": 1, "n_bins": 32},
    complexity_train="O(M · (n·p + 2^d·p·B + Σ_c K_c))",
    complexity_predict="O(M · m · (d + p))",
    justification="Boosting propio con codificación jerárquica contraída de las categóricas.",
)

FEATURES = C.SCENARIOS[SC]["numeric"] + C.SCENARIOS[SC]["categorical"]


def xyg_aug(df):
    """X con las variables del escenario A + identificador del activo (solo para contar)."""
    return df[FEATURES + [C.GROUP]], df[C.TARGET].to_numpy(), df[C.GROUP].to_numpy()


def run_fold(dev, test, fold, budget=C.BUDGET, seed=C.SEED):
    path = CACHE / f"{SC}__{NAME}__{METHOD}__{fold}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    X, y, g = xyg_aug(dev)
    if fold == "final":
        tr = np.arange(len(dev))
        X_ev, y_ev, _ = xyg_aug(test)
        ev_index = test.index.to_numpy()
    else:
        tr, te = list(outer_cv().split(X, y, g))[fold]
        X_ev, y_ev, ev_index = X.iloc[te], y[te], te
    t0 = time.perf_counter()
    res = bayesian_search(SPEC, X.iloc[tr], y[tr], g[tr], SC, budget, seed)
    t_search = time.perf_counter() - t0
    pipe = SPEC.make_pipeline(res["best_params"], SC, seed)
    t1 = time.perf_counter()
    pipe.fit(X.iloc[tr], y[tr])
    t_fit = time.perf_counter() - t1
    t2 = time.perf_counter()
    pred = pipe.predict(X_ev)
    t_pred = time.perf_counter() - t2
    out = {"scenario": SC, "model": NAME, "method": METHOD, "fold": fold, "seed": seed,
           "best_params": res["best_params"], "best_inner_rmse": res["best_inner_rmse"],
           "n_evals": res["n_evals"], "search_time": t_search, "fit_time": t_fit, "predict_time": t_pred,
           "history": res["history"], "extra": res["extra"], "metrics": regression_metrics(y_ev, pred),
           "pred": pred.astype(float), "y": np.asarray(y_ev, float), "index": np.asarray(ev_index)}
    if fold == "final":
        with open(C.MODELS_DIR / f"{SC}__{NAME}__{METHOD}.pkl", "wb") as fh:
            pickle.dump(pipe, fh)
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    print(f"[{NAME}] pliegue {fold}: RMSE {out['metrics']['rmse']:.4f}  ({t_search / 60:.1f} min)", flush=True)
    return out


if __name__ == "__main__":
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    C.set_global_seed()
    dev, test = split_dev_test(load_model_data())
    folds = list(range(C.OUTER_FOLDS)) + ["final"]
    t = time.perf_counter()
    res = Parallel(n_jobs=-1, verbose=0)(delayed(run_fold)(dev, test, f) for f in folds)
    rm = [r["metrics"]["rmse"] for r in res if r["fold"] != "final"]
    print(f"RMSE externo {np.mean(rm):.4f} ± {np.std(rm, ddof=1):.4f} | prueba {res[-1]['metrics']['rmse']:.4f}"
          f" | {(time.perf_counter() - t) / 60:.1f} min")
