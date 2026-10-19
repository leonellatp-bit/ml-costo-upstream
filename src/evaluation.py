"""Diagnóstico de residuos e intervalos de confianza.

Implementaciones con numpy/scipy (sin dependencias adicionales), siguiendo las
definiciones originales de cada prueba:

- White (1980), forma especial con valores ajustados.
- Ljung & Box (1978).
- Brock, Dechert, Scheinkman & LeBaron (BDS, 1996), misma construcción que
  ``statsmodels.tsa.stattools.bds``.
- Jarque-Bera y Shapiro-Wilk (normalidad), bootstrap BCa (scipy).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------
# Heterocedasticidad
# ------------------------------------------------------------------
def white_test(resid, fitted) -> dict:
    """Test de White (forma especial).

    Regresión auxiliar  e_i² = a + b·ŷ_i + c·ŷ_i² + u_i.  Bajo H0
    (homocedasticidad) LM = n·R²_aux ~ χ²(2). Se usa la forma especial porque
    el modelo tiene ~170 columnas One-Hot: la regresión auxiliar completa con
    cuadrados y productos cruzados tendría más regresores que observaciones.

    Returns
    -------
    dict con ``lm``, ``df`` y ``p_value``.
    """
    e2 = np.asarray(resid, float) ** 2
    f = np.asarray(fitted, float)
    Z = np.column_stack([np.ones_like(f), f, f ** 2])
    beta, *_ = np.linalg.lstsq(Z, e2, rcond=None)
    fit = Z @ beta
    r2 = 1 - ((e2 - fit) ** 2).sum() / ((e2 - e2.mean()) ** 2).sum()
    lm = len(e2) * r2
    return {"lm": float(lm), "df": 2, "p_value": float(stats.chi2.sf(lm, 2))}


# ------------------------------------------------------------------
# Autocorrelación
# ------------------------------------------------------------------
def acf(x, nlags: int = 20) -> np.ndarray:
    """Función de autocorrelación muestral (rezagos 0..nlags)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    denom = (x ** 2).sum()
    return np.array([1.0] + [(x[k:] * x[:-k]).sum() / denom for k in range(1, nlags + 1)])


def ljung_box(x, lags=(5, 10, 20)) -> pd.DataFrame:
    """Q de Ljung-Box: Q(h) = n(n+2) Σ_{k=1..h} ρ_k² / (n-k) ~ χ²(h) bajo independencia."""
    x = np.asarray(x, float)
    n = len(x)
    r = acf(x, max(lags))
    rows = []
    for h in lags:
        q = n * (n + 2) * np.sum(r[1:h + 1] ** 2 / (n - np.arange(1, h + 1)))
        rows.append({"lag": h, "Q": float(q), "p_value": float(stats.chi2.sf(q, h))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# BDS
# ------------------------------------------------------------------
def _corr_sum(ind, dim):
    if dim == 1:
        joint = ind
    else:
        _, ind = _corr_sum(ind, dim - 1)
        joint = ind[1:, 1:] * ind[:-1, :-1]
    iu = np.triu_indices(len(joint), 1)
    return joint[iu].mean(), joint


def bds_test(x, max_dim: int = 3, distance: float = 1.5) -> pd.DataFrame:
    """Prueba BDS de independencia i.i.d. (Brock et al., 1996).

    Compara la integral de correlación en dimensión m, C_m(ε), con C_1(ε)^m.
    Bajo i.i.d. W_m = √n (C_m − C_1^m) / σ_m → N(0, 1). ε = distance · sd(x).
    Detecta dependencia lineal y no lineal remanente en los residuos.
    """
    x = np.asarray(x, float).ravel()
    n_full = len(x)
    eps = distance * x.std(ddof=1)
    ind = (np.abs(x[:, None] - x[None, :]) < eps).astype(float)

    # integrales de correlación por dimensión
    cs = np.zeros(max_dim)
    for m in range(1, max_dim + 1):
        cs[m - 1], _ = _corr_sum(ind, m)

    # varianza asintótica
    c1 = cs[0]
    nobs = n_full
    k = ((ind.sum(1) ** 2).sum() - 3 * ind.sum() + 2 * nobs) / (nobs * (nobs - 1) * (nobs - 2))
    rows = []
    for m in range(2, max_dim + 1):
        tmp = sum((k ** (m - j)) * (c1 ** (2 * j)) for j in range(1, m))
        var = 4 * (k ** m + 2 * tmp + (m - 1) ** 2 * c1 ** (2 * m) - m ** 2 * k * c1 ** (2 * m - 2))
        n0 = m - 1
        c1_m, _ = _corr_sum(ind[n0:, n0:], 1)
        effect = cs[m - 1] - c1_m ** m
        w = np.sqrt(n_full - n0) * effect / np.sqrt(var)
        rows.append({"dim": m, "bds": float(w), "p_value": float(2 * stats.norm.sf(abs(w)))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Normalidad
# ------------------------------------------------------------------
def normality(resid) -> dict:
    r = np.asarray(resid, float)
    jb = stats.jarque_bera(r)
    sw = stats.shapiro(r[:5000])
    return {"jarque_bera": float(jb.statistic), "jb_p": float(jb.pvalue),
            "shapiro_w": float(sw.statistic), "shapiro_p": float(sw.pvalue),
            "skew": float(stats.skew(r)), "kurtosis_excess": float(stats.kurtosis(r))}


def residual_diagnostics(y, yhat, name: str = "") -> dict:
    """Resumen de pruebas de supuestos sobre los residuos (orden activo-año)."""
    e = np.asarray(y, float) - np.asarray(yhat, float)
    w = white_test(e, yhat)
    lb = ljung_box(e, (10,)).iloc[0]
    b = bds_test(e, 3)
    nm = normality(e)
    return {"model": name, "white_lm": w["lm"], "white_p": w["p_value"],
            "ljungbox_Q10": lb["Q"], "ljungbox_p10": lb["p_value"],
            "bds_m2": b.loc[0, "bds"], "bds_p_m2": b.loc[0, "p_value"],
            "bds_m3": b.loc[1, "bds"], "bds_p_m3": b.loc[1, "p_value"],
            "jb_p": nm["jb_p"], "skew": nm["skew"], "kurtosis_excess": nm["kurtosis_excess"]}


# ------------------------------------------------------------------
# Bootstrap BCa
# ------------------------------------------------------------------
def bca_ci(y, yhat, metric, n_resamples: int = 2000, seed: int = 42, level: float = 0.95):
    """IC bootstrap BCa (bias-corrected and accelerated) para una métrica (y, ŷ).

    Se usa BCa porque métricas como el RMSE tienen distribución asimétrica.
    """
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    res = stats.bootstrap((y, yhat), lambda a, b: metric(a, b), paired=True, vectorized=False,
                          n_resamples=n_resamples, method="BCa", confidence_level=level,
                          random_state=np.random.default_rng(seed))
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def r2(a, b):
    a = np.asarray(a)
    return float(1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))
