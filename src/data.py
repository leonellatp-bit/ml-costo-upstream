"""Carga de datos, partición por grupos y preprocesamiento encapsulado en Pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config as C


def load_raw(path=C.DATA_PATH) -> pd.DataFrame:
    """Lee el CSV original (separador ';').

    Returns
    -------
    pandas.DataFrame
        Datos crudos (4.595 filas x 17 columnas).
    """
    return pd.read_csv(path, sep=";")


def load_model_data(path=C.DATA_PATH) -> pd.DataFrame:
    """Datos listos para modelar: filas con target observado, ordenadas por activo y año.

    El orden activo-año se usa después para las pruebas de independencia de
    residuos (BDS y Ljung-Box), que requieren una secuencia definida.
    """
    df = load_raw(path)
    df = df.dropna(subset=[C.TARGET]).copy()
    df = df.sort_values([C.GROUP, "Year"]).reset_index(drop=True)
    return df


def split_dev_test(df: pd.DataFrame, seed: int = C.SEED):
    """Partición hold-out por activo (GroupShuffleSplit, 80/20).

    Ningún activo aparece a la vez en entrenamiento (train) y prueba (test). El conjunto de prueba
    no se toca durante la optimización: solo se usa para la evaluación final.

    Returns
    -------
    dev : DataFrame
        Conjunto de entrenamiento (train) (sobre él corre la CV anidada).
    test : DataFrame
        Conjunto de prueba reservado.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE, random_state=seed)
    dev_idx, test_idx = next(gss.split(df, groups=df[C.GROUP]))
    return df.iloc[dev_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def outer_cv(n_splits: int = C.OUTER_FOLDS) -> GroupKFold:
    """Bucle externo: GroupKFold por activo."""
    return GroupKFold(n_splits=n_splits)


def inner_cv(n_splits: int = C.INNER_FOLDS) -> GroupKFold:
    """Bucle interno: GroupKFold por activo dentro de cada partición de entrenamiento externa."""
    return GroupKFold(n_splits=n_splits)


def _log1p_nonneg(x):
    return np.log1p(np.clip(x, 0, None))


def make_preprocessor(scenario: str = C.MAIN_SCENARIO) -> ColumnTransformer:
    """Preprocesamiento ajustado solo con datos de entrenamiento (dentro de Pipeline).

    - Numéricas: imputación por mediana + indicador de faltante (Production falta
      por diseño en 2014: MNAR estructural), log1p para Production (asimetría 6.8),
      estandarización.
    - Categóricas: imputación por moda + One-Hot (categorías con < 5 casos se
      agrupan en 'infrequent'; categorías nuevas en prueba se ignoran).

    Parameters
    ----------
    scenario : str
        Clave de ``config.SCENARIOS``.
    """
    spec = C.SCENARIOS[scenario]
    num = spec["numeric"]
    cat = spec["categorical"]
    transformers = []
    if "Production" in num:
        transformers.append((
            "prod",
            Pipeline([
                ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                ("log", FunctionTransformer(_log1p_nonneg, feature_names_out="one-to-one")),
                ("sc", StandardScaler()),
            ]),
            ["Production"],
        ))
    other_num = [c for c in num if c != "Production"]
    if other_num:
        transformers.append((
            "num",
            Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
            other_num,
        ))
    transformers.append((
        "cat",
        Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5,
                                  sparse_output=False)),
        ]),
        cat,
    ))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def features_of(scenario: str = C.MAIN_SCENARIO) -> list[str]:
    """Lista de columnas de entrada para un escenario."""
    spec = C.SCENARIOS[scenario]
    return spec["numeric"] + spec["categorical"]


def xyg(df: pd.DataFrame, scenario: str = C.MAIN_SCENARIO):
    """Devuelve (X, y, grupos) para un escenario."""
    return df[features_of(scenario)], df[C.TARGET].to_numpy(), df[C.GROUP].to_numpy()
