"""Cuatro métodos de optimización de hiperparámetros con un mismo presupuesto.

Todos devuelven el mismo diccionario de resultados (mejores parámetros, mejor
RMSE interno e historial ordenado de evaluaciones) para poder construir curvas
de desempeño *anytime* y comparar la calidad alcanzada por unidad de presupuesto.

En los cuatro casos cada evaluación ajusta el **Pipeline completo**
(preprocesamiento + modelo) dentro de cada pliegue interno, de modo que la
imputación, el escalado y el One-Hot se aprenden solo con el entrenamiento del
pliegue (sin fuga de información).
"""
from __future__ import annotations

import random
import time
from itertools import product

import numpy as np
import optuna
from deap import base, creator, tools
from scipy import stats
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, cross_val_score

from . import config as C
from .data import inner_cv
from .models import ModelSpec, Param

optuna.logging.set_verbosity(optuna.logging.WARNING)
METHODS = ["Grid", "Random", "Bayesiana", "Genetica"]


# ------------------------------------------------------------------
# Utilidades comunes
# ------------------------------------------------------------------
def _cv_rmse(spec: ModelSpec, params: dict, X, y, groups, scenario: str, seed: int) -> float:
    """RMSE medio de la CV interna por grupos para una configuración."""
    pipe = spec.make_pipeline(params, scenario, seed)
    s = cross_val_score(pipe, X, y, groups=groups, cv=inner_cv(), scoring=C.SCORING, n_jobs=1)
    return float(-s.mean())


def _history_from_cv_results(cvr: dict) -> list[dict]:
    hist = []
    t = 0.0
    n = len(cvr["params"])
    for i in range(n):
        t += (cvr["mean_fit_time"][i] + cvr["mean_score_time"][i]) * C.INNER_FOLDS
        hist.append({"eval": i + 1, "rmse": float(-cvr["mean_test_score"][i]),
                     "elapsed": float(t), "params": cvr["params"][i]})
    return hist


def _strip(params: dict) -> dict:
    return {k.replace("model__", ""): v for k, v in params.items()}


# ------------------------------------------------------------------
# 1) Grid Search
# ------------------------------------------------------------------
def build_grid(space: list[Param], budget: int) -> dict[str, list]:
    """Rejilla con a lo sumo ``budget`` puntos.

    Se parte de 1 nivel por hiperparámetro (punto medio) y se agregan niveles
    por orden de prioridad mientras el producto no supere el presupuesto. Así
    la rejilla deja de ser exhaustiva en espacios de alta dimensión, que es
    precisamente la limitación de Grid Search frente a Random/Bayes/GA.
    """
    levels = {p.name: 1 for p in space}
    ordered = sorted(space, key=lambda p: p.priority)
    changed = True
    while changed:
        changed = False
        for p in ordered:
            cur = int(np.prod(list(levels.values())))
            new = cur // levels[p.name] * (levels[p.name] + 1)
            if new <= budget and levels[p.name] < p.n_levels_max():
                levels[p.name] += 1
                changed = True
    return {p.name: p.grid(levels[p.name]) for p in space}


def grid_search(spec, X, y, groups, scenario, budget=C.BUDGET, seed=C.SEED):
    grid = build_grid(spec.space, budget)
    gs = GridSearchCV(
        spec.make_pipeline(None, scenario, seed),
        {f"model__{k}": v for k, v in grid.items()},
        scoring=C.SCORING, cv=inner_cv(), n_jobs=1, refit=False,
    )
    gs.fit(X, y, groups=groups)
    hist = _history_from_cv_results(gs.cv_results_)
    best = int(np.argmax(gs.cv_results_["mean_test_score"]))
    return {"best_params": _strip(gs.cv_results_["params"][best]),
            "best_inner_rmse": float(-gs.cv_results_["mean_test_score"][best]),
            "history": hist, "n_evals": len(hist), "extra": {"grid": grid}}


# ------------------------------------------------------------------
# 2) Random Search
# ------------------------------------------------------------------
class _LogIntUniform:
    """Distribución entera log-uniforme compatible con RandomizedSearchCV."""

    def __init__(self, low, high):
        self.d = stats.loguniform(low, high + 0.999)

    def rvs(self, random_state=None, size=None):
        return np.floor(self.d.rvs(random_state=random_state, size=size)).astype(int)


def _distribution(p: Param):
    if p.kind == "cat":
        return list(p.choices)
    if p.kind == "int":
        return _LogIntUniform(p.low, p.high) if p.log else stats.randint(p.low, p.high + 1)
    return stats.loguniform(p.low, p.high) if p.log else stats.uniform(p.low, p.high - p.low)


def random_search(spec, X, y, groups, scenario, budget=C.BUDGET, seed=C.SEED):
    rs = RandomizedSearchCV(
        spec.make_pipeline(None, scenario, seed),
        {f"model__{p.name}": _distribution(p) for p in spec.space},
        n_iter=budget, scoring=C.SCORING, cv=inner_cv(), n_jobs=1, refit=False,
        random_state=seed,
    )
    rs.fit(X, y, groups=groups)
    hist = _history_from_cv_results(rs.cv_results_)
    best = int(np.argmax(rs.cv_results_["mean_test_score"]))
    return {"best_params": _strip(rs.cv_results_["params"][best]),
            "best_inner_rmse": float(-rs.cv_results_["mean_test_score"][best]),
            "history": hist, "n_evals": len(hist), "extra": {}}


# ------------------------------------------------------------------
# 3) Optimización Bayesiana (Optuna, TPE)
# ------------------------------------------------------------------
def _suggest(trial, p: Param):
    if p.kind == "cat":
        return trial.suggest_categorical(p.name, p.choices)
    if p.kind == "int":
        return trial.suggest_int(p.name, int(p.low), int(p.high), log=p.log)
    return trial.suggest_float(p.name, p.low, p.high, log=p.log)


def bayesian_search(spec, X, y, groups, scenario, budget=C.BUDGET, seed=C.SEED,
                    n_startup: int = 8):
    """Optuna con Tree-structured Parzen Estimator (TPE).

    - Modelo sustituto: TPE modela p(x | y < y*) = l(x) y p(x | y >= y*) = g(x)
      con estimadores de Parzen (densidades kernel) sobre el cuantil γ de las
      mejores evaluaciones.
    - Adquisición: Mejora Esperada (EI). Bergstra et al. (2011) muestran que
      maximizar EI en TPE equivale a maximizar l(x)/g(x); Optuna muestrea 24
      candidatos de l(x) y elige el de mayor cociente (explotación), mientras
      que el ancho de banda de los kernels y los ``n_startup`` ensayos
      aleatorios iniciales garantizan exploración.
    - ``multivariate=True`` modela dependencias entre hiperparámetros
      (p. ej. learning_rate × n_estimators en XGBoost).
    """
    hist = []
    t0 = time.perf_counter()

    def objective(trial):
        params = {p.name: _suggest(trial, p) for p in spec.space}
        rmse = _cv_rmse(spec, params, X, y, groups, scenario, seed)
        hist.append({"eval": len(hist) + 1, "rmse": rmse,
                     "elapsed": time.perf_counter() - t0, "params": params})
        return rmse

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=budget)
    return {"best_params": dict(study.best_params), "best_inner_rmse": float(study.best_value),
            "history": hist, "n_evals": len(hist),
            "extra": {"sampler": "TPE", "acquisition": "Expected Improvement (l(x)/g(x))",
                      "n_startup_trials": n_startup}}


# ------------------------------------------------------------------
# 4) Algoritmo genético (DEAP)
# ------------------------------------------------------------------
GA_CONFIG = {
    "pop_size": 10,          # tamaño de población
    "tournsize": 3,          # selección por torneo
    "cxpb": 0.7,             # prob. de cruce (cxBlend, alpha=0.5)
    "blend_alpha": 0.5,
    "mutpb": 0.3,            # prob. de mutar un individuo
    "mut_sigma": 0.15,       # mutación gaussiana sobre el genoma en [0,1]
    "elite": 1,              # elitismo: el mejor pasa intacto
}

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


def _decode(space, genome):
    return {p.name: p.from_unit(g) for p, g in zip(space, genome)}


def _diversity(pop) -> float:
    """Diversidad genética: distancia euclídea media entre pares de genomas (en [0,1]^d)."""
    G = np.array([list(ind) for ind in pop])
    if len(G) < 2:
        return 0.0
    d = np.sqrt(((G[:, None, :] - G[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(len(G), 1)
    return float(d[iu].mean() / np.sqrt(G.shape[1]))  # normalizada por sqrt(d)


def genetic_search(spec, X, y, groups, scenario, budget=C.BUDGET, seed=C.SEED, cfg=None):
    """Algoritmo genético con genoma real en [0,1]^d (decodificado a cada hiperparámetro).

    - Selección: torneo de tamaño 3 (presión selectiva moderada, robusta a la
      escala del RMSE, a diferencia de la ruleta).
    - Cruce: ``cxBlend`` (BLX-α, α=0.5) con prob. 0.7: genera hijos dentro y
      ligeramente fuera del segmento entre padres (exploración continua).
    - Mutación: gaussiana σ=0.15 por gen (prob. por gen 1/d) con prob. 0.3 por individuo.
    - Elitismo: el mejor individuo pasa sin cambios a la siguiente generación.
    - Población: 10. El presupuesto total de evaluaciones únicas es ``budget``
      (las configuraciones repetidas se recuperan de caché y no se cuentan).
    - Se registra la diversidad genética por generación como indicador de
      convergencia prematura.
    """
    cfg = {**GA_CONFIG, **(cfg or {})}
    rng = random.Random(seed)
    d = len(spec.space)
    tb = base.Toolbox()
    tb.register("mate", tools.cxBlend, alpha=cfg["blend_alpha"])
    tb.register("mutate", tools.mutGaussian, mu=0.0, sigma=cfg["mut_sigma"], indpb=max(1.0 / d, 0.34))
    tb.register("select", tools.selTournament, tournsize=cfg["tournsize"])

    cache: dict[tuple, float] = {}
    hist: list[dict] = []
    t0 = time.perf_counter()

    def evaluate(ind):
        genome = [float(np.clip(g, 0, 1)) for g in ind]
        ind[:] = genome
        params = _decode(spec.space, genome)
        key = tuple(sorted((k, str(v)) for k, v in params.items()))
        if key not in cache:
            if len(hist) >= budget:
                return (np.inf,)
            rmse = _cv_rmse(spec, params, X, y, groups, scenario, seed)
            cache[key] = rmse
            hist.append({"eval": len(hist) + 1, "rmse": rmse,
                         "elapsed": time.perf_counter() - t0, "params": params})
        return (cache[key],)

    pop = [creator.Individual([rng.random() for _ in range(d)]) for _ in range(cfg["pop_size"])]
    for ind in pop:
        ind.fitness.values = evaluate(ind)
    gens = [{"gen": 0, "diversity": _diversity(pop),
             "best": min(i.fitness.values[0] for i in pop), "n_evals": len(hist)}]

    # semilla del módulo random global usada por los operadores de DEAP
    state = random.getstate()
    random.seed(seed)
    g = 0
    while len(hist) < budget and g < 200:
        g += 1
        elite = tools.selBest(pop, cfg["elite"])
        offspring = [creator.Individual(list(i)) for i in tb.select(pop, len(pop) - cfg["elite"])]
        for a, b in zip(offspring[::2], offspring[1::2]):
            if random.random() < cfg["cxpb"]:
                tb.mate(a, b)
                del a.fitness.values, b.fitness.values
        for m in offspring:
            if random.random() < cfg["mutpb"]:
                tb.mutate(m)
                del m.fitness.values
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = evaluate(ind)
        pop = [creator.Individual(list(e)) for e in elite] + offspring
        for e, src in zip(pop[:cfg["elite"]], elite):
            e.fitness.values = src.fitness.values
        finite = [i.fitness.values[0] for i in pop if np.isfinite(i.fitness.values[0])]
        gens.append({"gen": g, "diversity": _diversity(pop), "best": min(finite),
                     "n_evals": len(hist)})
    random.setstate(state)

    best = min(hist, key=lambda h: h["rmse"])
    return {"best_params": best["params"], "best_inner_rmse": best["rmse"],
            "history": hist, "n_evals": len(hist),
            "extra": {"generations": gens, "config": cfg}}


SEARCHERS = {
    "Grid": grid_search,
    "Random": random_search,
    "Bayesiana": bayesian_search,
    "Genetica": genetic_search,
}
