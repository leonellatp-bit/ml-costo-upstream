"""Interpretabilidad global y local: SHAP (TreeExplainer) y LIME."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from . import config as C
from .data import features_of


def _original_var(feat: str, scenario: str) -> str:
    """Mapea una columna transformada (p. ej. 'CountryName_Colombia') a su variable original."""
    if feat.startswith("missingindicator_"):
        return feat.replace("missingindicator_", "") + " (faltante)"
    for v in sorted(features_of(scenario), key=len, reverse=True):
        if feat == v or feat.startswith(v + "_"):
            return v
    return feat


def shap_values(pipe, X: pd.DataFrame, scenario: str = C.MAIN_SCENARIO, background: int = 200,
                seed: int = C.SEED):
    """Valores SHAP exactos de árbol (TreeExplainer) sobre el espacio transformado.

    Returns
    -------
    sv : shap.Explanation
        Explicación a nivel de columna transformada (One-Hot).
    sv_agg : shap.Explanation
        Explicación agregada por variable original (suma de las columnas
        One-Hot de cada variable: por aditividad de Shapley la suma conserva la
        contribución total de la variable).
    """
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Z = prep.transform(X)
    names = list(prep.get_feature_names_out())
    explainer = shap.TreeExplainer(model)
    sv = explainer(Z)
    sv.feature_names = names
    groups = pd.Series([_original_var(n, scenario) for n in names])
    uniq = list(dict.fromkeys(groups))
    vals = np.column_stack([sv.values[:, (groups == u).to_numpy()].sum(1) for u in uniq])
    data = []
    for u in uniq:
        base = u.replace(" (faltante)", "")
        if base in X.columns and u == base:
            col = X[base]
            data.append(pd.to_numeric(col, errors="coerce").to_numpy()
                        if col.dtype.kind in "if" else pd.factorize(col)[0].astype(float))
        else:
            data.append(X["Production"].isna().astype(float).to_numpy())
    disp = []
    for u in uniq:
        base = u.replace(" (faltante)", "")
        if u != base:
            disp.append(np.where(X["Production"].isna(), "sí", "no"))
        elif X[base].dtype.kind in "if":
            disp.append(X[base].map(lambda v: "NaN" if pd.isna(v) else f"{v:g}").to_numpy())
        else:
            disp.append(X[base].astype(str).to_numpy())
    sv_agg = shap.Explanation(values=vals, base_values=sv.base_values,
                              data=np.column_stack(data), display_data=np.column_stack(disp),
                              feature_names=uniq)
    return sv, sv_agg


class _Codec:
    """Codifica categóricas como enteros para LIME y decodifica para el Pipeline."""

    def __init__(self, X: pd.DataFrame, cat_cols: list[str]):
        self.cols = list(X.columns)
        self.cat = cat_cols
        self.maps = {c: list(pd.Series(X[c].astype(str).unique()).sort_values()) for c in cat_cols}
        self.med = X.median(numeric_only=True)

    def encode(self, X: pd.DataFrame) -> np.ndarray:
        Z = X.copy()
        for c in self.cols:
            if c in self.cat:
                m = {v: i for i, v in enumerate(self.maps[c])}
                Z[c] = X[c].astype(str).map(m).fillna(0)
            else:
                Z[c] = X[c].fillna(self.med[c])
        return Z.to_numpy(float)

    def decode(self, A: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(A, columns=self.cols)
        for c in self.cat:
            idx = np.clip(np.round(df[c]).astype(int), 0, len(self.maps[c]) - 1)
            df[c] = [self.maps[c][i] for i in idx]
        return df


def lime_explain(pipe, X_train: pd.DataFrame, x_row: pd.DataFrame, scenario: str = C.MAIN_SCENARIO,
                 num_features: int = 8, num_samples: int = 3000, seed: int = C.SEED) -> pd.DataFrame:
    """Explicación LIME (modelo lineal local ponderado) de una observación.

    LIME perturba la observación, predice con el Pipeline completo y ajusta una
    regresión ridge ponderada por proximidad; los coeficientes son la
    explicación local (aproximación, no atribución exacta como SHAP).
    """
    cat = [c for c in C.SCENARIOS[scenario]["categorical"] if c in X_train.columns]
    codec = _Codec(X_train, cat)
    A = codec.encode(X_train)
    cat_idx = [codec.cols.index(c) for c in cat]
    expl = LimeTabularExplainer(A, mode="regression", feature_names=codec.cols,
                                categorical_features=cat_idx,
                                categorical_names={i: codec.maps[c] for i, c in zip(cat_idx, cat)},
                                discretize_continuous=True, random_state=seed)
    exp = expl.explain_instance(codec.encode(x_row)[0], lambda M: pipe.predict(codec.decode(M)),
                                num_features=num_features, num_samples=num_samples)
    rows = [{"regla": r, "peso": w} for r, w in exp.as_list()]
    out = pd.DataFrame(rows)
    first = lambda v: float(np.ravel(list(v.values()) if isinstance(v, dict) else v)[0])  # noqa: E731
    out.attrs["intercept"] = first(exp.intercept)
    out.attrs["local_pred"] = first(exp.local_pred)
    out.attrs["score_r2_local"] = first(exp.score)
    return out
