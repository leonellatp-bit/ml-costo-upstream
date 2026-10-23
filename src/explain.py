"""Explicaciones e intervalos para el dashboard (independientes del modelo).

* ``grouped_shapley``: valores de Shapley exactos sobre grupos de variables (jugadores),
  con la función de valor intervencional v(S) = E_b[f(x_S, b_{-S})] sobre un fondo b.
  Con 5 jugadores hay 2^5 = 32 coaliciones, así que el cálculo es exacto y sirve para
  cualquier modelo (benchmarks y modelo original). Agrupar la geografía evita mezclar un
  país con la región de otro (combinaciones imposibles).
* ``conformal_*``: intervalo de predicción conformal partido (split conformal) relativo:
  ŷ · (1 ± q), con q el cuantil (1 − α)(1 + 1/n) de |y − ŷ| / ŷ en predicciones fuera de
  pliegue. Es relativo porque el error crece con el nivel de costo (heterocedasticidad).
"""
from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

PLAYERS = {
    "Geografía": ["RegionName", "CountryName", "ProductionElementName", "MemberOpec"],
    "Tipo de reserva": ["ReserveTypeName", "SupplySubTypeName"],
    "Ubicación": ["ReserveLocationName"],
    "Año": ["Year"],
    "Producción": ["Production"],
}


def grouped_shapley(predict, X: pd.DataFrame, background: pd.DataFrame, players: dict = PLAYERS):
    """Shapley exacto por grupos para cada fila de X.

    Returns
    -------
    phi : DataFrame (len(X), n_jugadores) con las contribuciones en unidades del objetivo.
    base : float, valor esperado E_b[f(b)] (predicción con ningún jugador presente).
    Se cumple la eficiencia: base + phi.sum(1) = f(x).
    """
    names = list(players)
    p = len(names)
    coalitions = [frozenset(c) for r in range(p + 1) for c in combinations(range(p), r)]
    B = len(background)
    rows = []
    for _, x in X.iterrows():
        for S in coalitions:
            z = background.copy()
            for i in S:
                for col in players[names[i]]:
                    z[col] = x[col]
            rows.append(z)
    Z = pd.concat(rows, ignore_index=True)
    f = np.asarray(predict(Z), float).reshape(len(X), len(coalitions), B).mean(2)
    idx = {S: k for k, S in enumerate(coalitions)}
    w = {s: factorial(s) * factorial(p - s - 1) / factorial(p) for s in range(p)}
    phi = np.zeros((len(X), p))
    for i in range(p):
        for S in coalitions:
            if i in S:
                continue
            phi[:, i] += w[len(S)] * (f[:, idx[S | {i}]] - f[:, idx[S]])
    base = float(f[:, idx[frozenset()]].mean())
    return pd.DataFrame(phi, columns=names, index=X.index), base


def conformal_quantile(y, yhat, alpha: float = 0.10) -> float:
    """Cuantil conformal del error relativo |y − ŷ| / ŷ (predicciones fuera de pliegue)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    s = np.abs(y - yhat) / np.maximum(yhat, 1e-6)
    n = len(s)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, level, method="higher"))


def conformal_interval(yhat, q: float):
    """Intervalo [ŷ(1 − q), ŷ(1 + q)]."""
    yhat = np.asarray(yhat, float)
    return yhat * (1 - q), yhat * (1 + q)
