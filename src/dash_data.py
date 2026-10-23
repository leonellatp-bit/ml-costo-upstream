"""Carga y caché de datos y artefactos para el dashboard (se leen una sola vez por proceso).

El dashboard nunca entrena ni optimiza: solo carga artefactos ya entrenados
(``artifacts/<Modelo>/model.joblib`` + ``metadata.json``) y tablas precalculadas
(``results/``). Los datos crudos (licencia S&P Global) no están en el repositorio:
se buscan en DATA_PATH, en el Secret File de Render o en ``data/``.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from . import config as C

ROOT = C.ROOT
ART = ROOT / "artifacts"
RES = C.RESULTS_DIR
DASH = RES / "dashboard"
ORIG = RES / "original"

MODEL_ORDER = ["KNN", "Ridge", "Lasso", "DecisionTree", "RandomForest", "XGBoost", "SVR", "HGSB"]
LABELS = {"KNN": "KNN", "Ridge": "Ridge", "Lasso": "Lasso", "DecisionTree": "Árbol de decisión",
          "RandomForest": "Random Forest", "XGBoost": "XGBoost", "SVR": "SVR (RBF)",
          "HGSB": "HGSB · modelo original"}
FEATURES = C.SCENARIOS[C.MAIN_SCENARIO]["numeric"] + C.SCENARIOS[C.MAIN_SCENARIO]["categorical"]


def data_path() -> Path:
    """DATA_PATH -> Secret File de Render (/etc/secrets/) -> data/ local."""
    for cand in [os.environ.get("DATA_PATH"), f"/etc/secrets/{C.DATA_PATH.name}", str(C.DATA_PATH)]:
        if cand and Path(cand).exists():
            return Path(cand)
    raise FileNotFoundError(
        f"No se encontró {C.DATA_PATH.name}. En local: cópialo en data/. En Render: súbelo como Secret File.")


@lru_cache(maxsize=1)
def raw() -> pd.DataFrame:
    return pd.read_csv(data_path(), sep=";")


@lru_cache(maxsize=1)
def data() -> pd.DataFrame:
    """Filas con objetivo observado."""
    return raw().dropna(subset=[C.TARGET]).reset_index(drop=True)


def _csv(path, **kw) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kw)
    except FileNotFoundError:
        return pd.DataFrame()


def _json(path) -> dict:
    try:
        return json.load(open(path, encoding="utf-8"))
    except FileNotFoundError:
        return {}


@lru_cache(maxsize=None)
def table(name: str) -> pd.DataFrame:
    """Tablas precalculadas por nombre lógico."""
    paths = {
        "leaderboard": ART / "leaderboard.csv",
        "master": RES / "master_table.csv",
        "predictions": DASH / "predictions.csv",
        "acf": DASH / "residual_acf.csv",
        "shap_global": DASH / "shap_global.csv",
        "shap_rows": DASH / "shap_rows.csv",
        "mcs": ORIG / "mcs.csv",
        "tests": ORIG / "tests.csv",
        "ablation": ORIG / "ablation.csv",
        "convergence": ORIG / "convergence.csv",
        "folds": ORIG / "folds.csv",
        "background": DASH / "shap_background.csv",
    }
    return _csv(paths[name])


@lru_cache(maxsize=None)
def jfile(name: str) -> dict:
    paths = {"options": DASH / "predictor_options.json", "verdict": ORIG / "verdict.json",
             "conformal": ART / "conformal.json"}
    return _json(paths[name])


@lru_cache(maxsize=None)
def metadata(model: str) -> dict:
    return _json(ART / model / "metadata.json")


@lru_cache(maxsize=None)
def model(name: str):
    """Carga perezosa del artefacto (una vez por proceso)."""
    path = ART / name / "model.joblib"
    return joblib.load(path) if path.exists() else None


def available_models() -> list[str]:
    return [m for m in MODEL_ORDER if (ART / m / "model.joblib").exists()]
