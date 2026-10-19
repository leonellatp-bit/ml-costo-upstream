"""Registro de los 7 modelos de regresión, sus espacios de búsqueda y su complejidad teórica.

Cada espacio se define una sola vez con la clase :class:`Param`; los cuatro
optimizadores (Grid, Random, Bayesiano y Genético) lo traducen a su propio
formato, de modo que todos exploran exactamente el mismo dominio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from . import config as C
from .data import make_preprocessor


@dataclass
class Param:
    """Hiperparámetro de un espacio de búsqueda.

    Parameters
    ----------
    name : str
        Nombre del argumento del estimador.
    kind : {"float", "int", "cat"}
        Tipo del hiperparámetro.
    low, high : float, optional
        Límites (inclusive) para ``float`` e ``int``.
    log : bool
        Si True, se explora en escala logarítmica (C, alpha, learning rate...).
    choices : list, optional
        Valores posibles para ``cat``.
    priority : int
        Orden de importancia (0 = más importante). Grid Search asigna más
        niveles a los hiperparámetros prioritarios cuando el presupuesto no
        alcanza para la rejilla completa.
    """

    name: str
    kind: str
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: list | None = None
    priority: int = 0

    # --- transformaciones entre el cubo unitario y el dominio real ---
    def from_unit(self, u: float) -> Any:
        """Mapea u en [0, 1] al dominio del parámetro (usado por Random y GA)."""
        u = float(np.clip(u, 0.0, 1.0))
        if self.kind == "cat":
            idx = min(int(u * len(self.choices)), len(self.choices) - 1)
            return self.choices[idx]
        if self.log:
            v = np.exp(np.log(self.low) + u * (np.log(self.high) - np.log(self.low)))
        else:
            v = self.low + u * (self.high - self.low)
        return int(round(v)) if self.kind == "int" else float(v)

    def grid(self, k: int) -> list:
        """k niveles equiespaciados (en escala log si aplica)."""
        if self.kind == "cat":
            return list(self.choices[:k]) if k < len(self.choices) else list(self.choices)
        if k == 1:
            us = [0.5]
        else:
            us = np.linspace(0, 1, k)
        vals = [self.from_unit(u) for u in us]
        # quitar duplicados en enteros conservando el orden
        out = []
        for v in vals:
            if v not in out:
                out.append(v)
        return out

    def n_levels_max(self) -> int:
        if self.kind == "cat":
            return len(self.choices)
        if self.kind == "int":
            return int(self.high - self.low + 1)
        return 10_000


@dataclass
class ModelSpec:
    """Especificación de un modelo: constructor, espacio y complejidad teórica."""

    name: str
    builder: Callable[..., Any]
    space: list[Param]
    fixed: dict = field(default_factory=dict)
    complexity_train: str = ""
    complexity_predict: str = ""
    justification: str = ""

    def make_pipeline(self, params: dict | None = None, scenario: str = C.MAIN_SCENARIO,
                      seed: int = C.SEED) -> Pipeline:
        """Pipeline completo (preprocesamiento + modelo) con los hiperparámetros dados."""
        kw = dict(self.fixed)
        kw.update(params or {})
        est = self.builder(seed=seed, **kw)
        return Pipeline([("prep", make_preprocessor(scenario)), ("model", est)])


# ------------------------------------------------------------------
# Constructores (reciben la semilla explícitamente)
# ------------------------------------------------------------------
def _knn(seed=None, **kw):
    return KNeighborsRegressor(**kw)


def _ridge(seed=None, **kw):
    return Ridge(random_state=seed, **kw)


def _lasso(seed=None, **kw):
    return Lasso(random_state=seed, max_iter=20_000, **kw)


def _tree(seed=None, **kw):
    return DecisionTreeRegressor(random_state=seed, **kw)


def _rf(seed=None, **kw):
    return RandomForestRegressor(random_state=seed, n_jobs=1, **kw)


def _xgb(seed=None, **kw):
    return XGBRegressor(random_state=seed, tree_method="hist", n_jobs=1, verbosity=0, **kw)


def _svr(seed=None, **kw):
    return SVR(kernel="rbf", cache_size=500, **kw)


MODELS: dict[str, ModelSpec] = {
    "KNN": ModelSpec(
        "KNN", _knn,
        [Param("n_neighbors", "int", 2, 60, log=True, priority=0),
         Param("weights", "cat", choices=["uniform", "distance"], priority=1),
         Param("p", "cat", choices=[1, 2], priority=2)],
        complexity_train="O(1) (fuerza bruta) | O(n·p·log n) (KD/Ball tree)",
        complexity_predict="O(n·p) por consulta (fuerza bruta) | ~O(p·log n) (árbol, baja dimensión)",
        justification="k en escala log (2–60): con ~2.4k filas de entrenamiento, k>60 suaviza en exceso; "
                      "distance vs uniform y métrica Manhattan/Euclídea.",
    ),
    "Ridge": ModelSpec(
        "Ridge", _ridge,
        [Param("alpha", "float", 1e-3, 1e3, log=True)],
        complexity_train="O(n·p² + p³) (Cholesky) | O(n·p) por iteración (SAGA)",
        complexity_predict="O(p)",
        justification="λ en escala log de 1e-3 a 1e3: cubre desde OLS casi sin penalización hasta "
                      "coeficientes muy contraídos.",
    ),
    "Lasso": ModelSpec(
        "Lasso", _lasso,
        [Param("alpha", "float", 1e-4, 1e1, log=True)],
        complexity_train="O(n·p) por iteración (descenso coordenado) | O(n·p) por época (SAGA)",
        complexity_predict="O(p)",
        justification="λ log 1e-4–10: con target en USD/Bbl (σ≈5.5) λ>10 anula todos los coeficientes.",
    ),
    "DecisionTree": ModelSpec(
        "DecisionTree", _tree,
        [Param("max_depth", "int", 2, 25, priority=0),
         Param("min_samples_leaf", "int", 1, 50, log=True, priority=1),
         Param("min_samples_split", "int", 2, 40, log=True, priority=2)],
        complexity_train="O(n·p·log n)",
        complexity_predict="O(profundidad) ≈ O(log n)",
        justification="Profundidad 2–25 y hojas mínimas 1–50 (log): controlan el sobreajuste.",
    ),
    "RandomForest": ModelSpec(
        "RandomForest", _rf,
        [Param("max_features", "float", 0.1, 1.0, priority=0),
         Param("min_samples_leaf", "int", 1, 20, log=True, priority=1),
         Param("max_depth", "int", 4, 30, priority=2),
         Param("n_estimators", "int", 50, 300, priority=3)],
        complexity_train="O(T·n·m·log n), m = max_features·p",
        complexity_predict="O(T·log n)",
        justification="max_features (decorrelación de árboles) es el más influyente; n_estimators "
                      "acotado a 300 porque el error se estabiliza antes y el costo es lineal en T.",
    ),
    "XGBoost": ModelSpec(
        "XGBoost", _xgb,
        [Param("learning_rate", "float", 0.01, 0.3, log=True, priority=0),
         Param("max_depth", "int", 2, 10, priority=1),
         Param("n_estimators", "int", 100, 600, priority=2),
         Param("subsample", "float", 0.5, 1.0, priority=3),
         Param("colsample_bytree", "float", 0.3, 1.0, priority=4),
         Param("min_child_weight", "float", 1.0, 20.0, log=True, priority=5),
         Param("reg_lambda", "float", 1e-3, 10.0, log=True, priority=6)],
        complexity_train="O(T·(n·p) ) con 'hist' (O(T·n·p·log n) con 'exact')",
        complexity_predict="O(T·profundidad)",
        justification="learning_rate y n_estimators interactúan (lr bajo requiere más árboles); "
                      "subsample/colsample introducen aleatoriedad; reg_lambda log.",
    ),
    "SVR": ModelSpec(
        "SVR", _svr,
        [Param("C", "float", 0.1, 1000.0, log=True, priority=0),
         Param("gamma", "float", 1e-4, 1.0, log=True, priority=1),
         Param("epsilon", "float", 0.01, 2.0, log=True, priority=2)],
        complexity_train="O(n²·p) a O(n³) (kernel RBF, libsvm)",
        complexity_predict="O(n_SV·p)",
        justification="Kernel RBF (el SVR lineal fue el modelo base del entregable 1). C, γ y ε en "
                      "escala log; ε hasta 2 USD/Bbl (≈ RMSE del modelo base).",
    ),
}

MODEL_ORDER = list(MODELS)
