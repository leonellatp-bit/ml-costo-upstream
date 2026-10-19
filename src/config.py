"""Configuración global del proyecto: rutas, semilla, variables y esquema de validación.

Todas las demás piezas del pipeline leen sus constantes de este módulo para que
exista una única fuente de verdad (reproducibilidad, Sección 7.3 de la guía).
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------
# Rutas
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "DatasetOilCostProjectWorldwide.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
for _d in (RESULTS_DIR, FIG_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Semilla global única (se propaga a numpy, random, scikit-learn,
# XGBoost, Optuna y DEAP)
# ------------------------------------------------------------------
SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    """Fija la semilla de todas las fuentes de aleatoriedad del proyecto.

    Parameters
    ----------
    seed : int
        Semilla global. scikit-learn, XGBoost, Optuna y DEAP reciben esta
        misma semilla de forma explícita en cada constructor.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ------------------------------------------------------------------
# Variables
# ------------------------------------------------------------------
TARGET = "BrentEquivalentProductionCost"
GROUP = "CostProductionElementName"  # entidad (activo) -> particiones por grupo

CATEGORICAL = [
    "RegionName",
    "CountryName",
    "ReserveTypeName",
    "SupplySubTypeName",
    "ReserveLocationName",
    "ProductionElementName",
    "MemberOpec",
]
NUMERIC_BASE = ["Year", "Production"]

# Identidad contable verificada en el EDA mejorado:
#   BrentEquivalentProductionCost = WellheadProductionCost + BrentDifferential
#   BrentEquivalentDevelopmentCost = WellheadDevelopmentCost + BrentDifferential
# Por eso se definen dos escenarios de variables.
SCENARIOS = {
    # A: estructural (principal). Solo características del activo conocidas
    #    antes de observar el costo. Excluye todo componente del target.
    "A_estructural": {"numeric": NUMERIC_BASE, "categorical": CATEGORICAL},
    # B: con BrentDifferential (alternativo). BD es un sumando exacto del target;
    #    se mantiene solo como análisis de sensibilidad frente al entregable 1.
    "B_con_BD": {"numeric": NUMERIC_BASE + ["BrentDifferential"], "categorical": CATEGORICAL},
}
MAIN_SCENARIO = "A_estructural"

# Variables descartadas y motivo (auditoría de fuga)
DROPPED = {
    "Commodity": "constante (Crude oil)",
    "SupplyTypeName": "constante (Crude)",
    "WellheadProductionCost": "componente exacto del target (target = WPC + BD)",
    "WellheadDevelopmentCost": "costo de desarrollo; redundante con BEDC (VIF ~1e9)",
    "BrentEquivalentDevelopmentCost": "= WDC + BD (identidad exacta)",
    "CostProductionElementName": "identificador de entidad: solo se usa para agrupar",
}

# ------------------------------------------------------------------
# Validación
# ------------------------------------------------------------------
TEST_SIZE = 0.20      # hold-out por grupos (igual que el entregable 1)
OUTER_FOLDS = 10      # bucle externo de la CV anidada (556 activos -> ~56 por pliegue)
INNER_FOLDS = 3       # bucle interno (búsqueda de hiperparámetros)
BUDGET = 30           # evaluaciones por búsqueda (igual para los 4 métodos)
SCORING = "neg_root_mean_squared_error"
N_JOBS = int(os.environ.get("N_JOBS", "-1"))
