"""Comparación estadística jerárquica de modelos.

Clasificación no aplica (el proyecto es de regresión), así que se implementa:

1. Ómnibus sobre pliegues externos: Friedman.
2. Post-hoc: Nemenyi (diferencia crítica) + diagrama CD.
3. Regresión (pronósticos sobre el hold-out):
   MCS (Hansen, Lunde & Nason, 2011) -> Giacomini-White (2006) / Clark-West
   (2007) -> Diebold-Mariano con corrección HLN + bootstrap estacionario
   (Politis & Romano, 1994). SPA de Hansen (2005) para data snooping.
4. Corrección por comparaciones múltiples (Holm, Benjamini-Hochberg).
5. Tamaños de efecto: δ de Cliff y d de Cohen.

Todas las funciones dependen solo de numpy, pandas, scipy y matplotlib.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------
# Comparaciones múltiples
# ------------------------------------------------------------------
def holm(pvals) -> np.ndarray:
    """p-valores ajustados por Holm-Bonferroni (control FWER)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def benjamini_hochberg(pvals) -> np.ndarray:
    """p-valores ajustados por Benjamini-Hochberg (control FDR)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    run = 1.0
    for rank, i in enumerate(order):
        run = min(run, p[i] * m / (m - rank))
        adj[i] = run
    return adj


# ------------------------------------------------------------------
# Tamaños de efecto
# ------------------------------------------------------------------
def cliffs_delta(a, b) -> tuple[float, str]:
    """δ de Cliff = P(a > b) − P(a < b). Magnitud según Romano et al. (2006)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    d = (gt - lt) / (len(a) * len(b))
    ad = abs(d)
    mag = "despreciable" if ad < 0.147 else "pequeño" if ad < 0.33 else "mediano" if ad < 0.474 else "grande"
    return float(d), mag


def cohens_d_paired(diff) -> tuple[float, str]:
    """d de Cohen para diferencias pareadas de pérdida (media / desv. estándar)."""
    diff = np.asarray(diff, float)
    d = diff.mean() / diff.std(ddof=1)
    ad = abs(d)
    mag = "despreciable" if ad < 0.2 else "pequeño" if ad < 0.5 else "mediano" if ad < 0.8 else "grande"
    return float(d), mag


# ------------------------------------------------------------------
# Friedman + Nemenyi
# ------------------------------------------------------------------
def friedman(fold_matrix: pd.DataFrame) -> dict:
    """Test de Friedman sobre una matriz pliegues × modelos (métrica de pérdida).

    Rango 1 = mejor (menor pérdida). H0: todos los modelos tienen el mismo rango medio.
    También se reporta la corrección F de Iman-Davenport (menos conservadora).
    """
    M = fold_matrix.to_numpy(float)
    N, k = M.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, M)
    avg = pd.Series(ranks.mean(0), index=fold_matrix.columns).sort_values()
    chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    ff = (N - 1) * chi2 / (N * (k - 1) - chi2)
    p_ff = stats.f.sf(ff, k - 1, (k - 1) * (N - 1))
    # W de Kendall como tamaño de efecto del ómnibus
    w = chi2 / (N * (k - 1))
    return {"chi2": float(chi2), "df": k - 1, "p_value": float(p), "iman_davenport_F": float(ff),
            "p_value_F": float(p_ff), "kendall_W": float(w), "N": N, "k": k, "avg_ranks": avg}


def nemenyi(avg_ranks: pd.Series, N: int, alpha: float = 0.05) -> tuple[float, pd.DataFrame]:
    """Diferencia crítica de Nemenyi y p-valores por pares.

    CD = q_α · sqrt(k(k+1) / (6N)), con q_α = rango studentizado(1−α, k, ∞)/√2.
    """
    k = len(avg_ranks)
    se = np.sqrt(k * (k + 1) / (6 * N))
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * se
    names = avg_ranks.index
    P = pd.DataFrame(np.ones((k, k)), index=names, columns=names)
    for i in range(k):
        for j in range(i + 1, k):
            z = abs(avg_ranks.iloc[i] - avg_ranks.iloc[j]) / se
            p = stats.studentized_range.sf(z * np.sqrt(2), k, np.inf)
            P.iloc[i, j] = P.iloc[j, i] = p
    return float(cd), P


def cd_diagram(avg_ranks: pd.Series, cd: float, ax=None, title: str = "", max_label: int = 40):
    """Diagrama de diferencia crítica (Demšar, 2006).

    Ubica cada modelo según su rango medio (1 = mejor, a la izquierda) y une con
    barras horizontales los grupos de modelos cuyas diferencias son menores que
    la CD (no significativas según Nemenyi).
    """
    import matplotlib.pyplot as plt

    r = avg_ranks.sort_values()
    k = len(r)
    names = [n[:max_label] for n in r.index]
    vals = r.values
    lo, hi = 1, k
    # grupos no significativos: cliques máximas de modelos consecutivos
    groups = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in groups):
            groups.append((i, j))
    half = int(np.ceil(k / 2))
    row_h, grp_h = 0.42, 0.22                     # alturas en "unidades de fila"
    top_cd, axis_y = 1.35, 0.0                    # coordenadas verticales (y crece hacia arriba)
    first_lbl = -(0.35 + grp_h * len(groups) + 0.35)
    bottom = first_lbl - row_h * (half - 1) - 0.4
    if ax is None:
        h = max(2.2, 0.55 * (top_cd + 0.6 - bottom))
        _, ax = plt.subplots(figsize=(10, h))
    ax.set_ylim(bottom, top_cd + 0.6)
    ax.set_xlim(lo - 0.25 * k - 1.5, hi + 0.25 * k + 1.5)
    ax.axis("off")
    ax.grid(False)
    ink = "#333333"
    ax.hlines(axis_y, lo, hi, color=ink, lw=1)
    step = 1 if k <= 12 else 2 if k <= 24 else 3
    for t in range(lo, hi + 1):
        ax.vlines(t, axis_y, axis_y + 0.12, color=ink, lw=1)
        if (t - lo) % step == 0 or t == hi:
            ax.text(t, axis_y + 0.2, str(t), ha="center", va="bottom", fontsize=8, color="#555")
    # barra de la diferencia crítica
    ax.hlines(top_cd, lo, lo + cd, color="#c0392b", lw=2.2)
    ax.vlines([lo, lo + cd], top_cd - 0.08, top_cd + 0.08, color="#c0392b", lw=1.5)
    ax.text(lo + cd / 2, top_cd + 0.14, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8.5, color="#c0392b")
    # barras de grupos no significativos, cada una en su propia fila
    for g, (a, b) in enumerate(groups):
        yy = axis_y - 0.35 - grp_h * g
        ax.hlines(yy, vals[a] - 0.08, vals[b] + 0.08, color="#2c3e50", lw=3.2)
    # etiquetas: mejores a la izquierda, peores a la derecha
    for i, (n, v) in enumerate(zip(names, vals)):
        left = i < half
        row = i if left else k - 1 - i
        yy = first_lbl - row_h * row
        ax.vlines(v, yy, axis_y, color="#8a8a8a", lw=0.8)
        if left:
            ax.hlines(yy, lo - 0.25 * k - 0.2, v, color="#8a8a8a", lw=0.8)
            ax.text(lo - 0.25 * k - 0.3, yy, f"{n} ({v:.2f})", ha="right", va="center", fontsize=8)
        else:
            ax.hlines(yy, v, hi + 0.25 * k + 0.2, color="#8a8a8a", lw=0.8)
            ax.text(hi + 0.25 * k + 0.3, yy, f"({v:.2f}) {n}", ha="left", va="center", fontsize=8)
    if title:
        ax.set_title(title, fontsize=10, pad=6)
    return ax


# ------------------------------------------------------------------
# Bootstrap estacionario (Politis & Romano, 1994)
# ------------------------------------------------------------------
def stationary_bootstrap_indices(n: int, B: int, mean_block: float, rng) -> np.ndarray:
    """Índices de B réplicas del bootstrap estacionario (bloques de longitud geométrica)."""
    p = 1.0 / mean_block
    idx = np.empty((B, n), dtype=int)
    idx[:, 0] = rng.integers(0, n, B)
    new_block = rng.random((B, n)) < p
    starts = rng.integers(0, n, (B, n))
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def _block_len(n):
    return max(2.0, n ** (1 / 3))


# ------------------------------------------------------------------
# Diebold-Mariano con corrección HLN
# ------------------------------------------------------------------
def diebold_mariano(e1, e2, h: int = 1, power: int = 2) -> dict:
    """DM con corrección de Harvey-Leybourne-Newbold (1997).

    d_t = |e1_t|^p − |e2_t|^p. DM = d̄ / sqrt(LRV/n), LRV con autocovarianzas
    hasta h−1. HLN: DM* = DM · sqrt((n + 1 − 2h + h(h−1)/n) / n), comparado con
    t_{n−1}. d̄ > 0 => el modelo 2 tiene menor pérdida.
    """
    e1 = np.asarray(e1, float)
    e2 = np.asarray(e2, float)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    lrv = (dc @ dc) / n
    for lag in range(1, h):
        lrv += 2 * (dc[lag:] @ dc[:-lag]) / n
    dm = dbar / np.sqrt(lrv / n)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * corr
    p = 2 * stats.t.sf(abs(stat), n - 1)
    return {"mean_loss_diff": float(dbar), "dm_hln": float(stat), "p_value": float(p), "n": n}


def bootstrap_mean_ci(d, B: int = 2000, level: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """IC percentil de la media de una serie dependiente vía bootstrap estacionario."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(len(d), B, _block_len(len(d)), rng)
    means = d[idx].mean(1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


# ------------------------------------------------------------------
# Giacomini-White
# ------------------------------------------------------------------
def giacomini_white(e1, e2, conditional: bool = True, power: int = 2) -> dict:
    """Test de habilidad predictiva condicional de Giacomini & White (2006), h = 1.

    d_t = L1_t − L2_t; instrumentos h_{t−1} = (1, d_{t−1}) (condicional) o (1)
    (incondicional). Z_t = h_{t−1} d_t; GW = n · Z̄' Ω̂⁻¹ Z̄ ~ χ²(q).
    """
    d = np.abs(np.asarray(e1, float)) ** power - np.abs(np.asarray(e2, float)) ** power
    if conditional:
        H = np.column_stack([np.ones(len(d) - 1), d[:-1]])
        dd = d[1:]
    else:
        H = np.ones((len(d), 1))
        dd = d
    Z = H * dd[:, None]
    n, q = Z.shape
    zbar = Z.mean(0)
    omega = (Z.T @ Z) / n
    gw = n * zbar @ np.linalg.solve(omega, zbar)
    return {"gw": float(gw), "df": q, "p_value": float(stats.chi2.sf(gw, q)),
            "mean_loss_diff": float(d.mean())}


# ------------------------------------------------------------------
# Clark-West (modelos anidados)
# ------------------------------------------------------------------
def clark_west(y, yhat_small, yhat_big) -> dict:
    """Test de Clark & West (2007) para modelos anidados.

    f_t = (y − ŷ₁)² − [(y − ŷ₂)² − (ŷ₁ − ŷ₂)²]; H0: igual MSPE poblacional;
    H1 (unilateral): el modelo grande (2) predice mejor. Estadístico t ~ N(0,1).
    """
    y = np.asarray(y, float)
    a = np.asarray(yhat_small, float)
    b = np.asarray(yhat_big, float)
    f = (y - a) ** 2 - ((y - b) ** 2 - (a - b) ** 2)
    t = f.mean() / (f.std(ddof=1) / np.sqrt(len(f)))
    return {"cw": float(t), "p_value": float(stats.norm.sf(t)), "mean_adj_diff": float(f.mean())}


# ------------------------------------------------------------------
# Model Confidence Set (Hansen, Lunde & Nason, 2011) - estadístico T_max
# ------------------------------------------------------------------
def model_confidence_set(losses: pd.DataFrame, alpha: float = 0.10, B: int = 2000,
                         seed: int = 42, block: float | None = None) -> pd.DataFrame:
    """MCS con estadístico T_max y bootstrap estacionario.

    En cada paso: d_i· = media de L_i − L̄ (pérdida relativa al promedio del
    conjunto), t_i = d̄_i· / sd_boot(d̄_i·), T_max = max t_i. Si el p-valor
    bootstrap de T_max < α se elimina el modelo con mayor t_i. El p-valor MCS
    de cada modelo es el máximo acumulado de los p-valores de eliminación.
    """
    L = losses.to_numpy(float)
    n, M = L.shape
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, B, block or _block_len(n), rng)
    Lb = L[idx].mean(1)            # B x M medias bootstrap
    Lbar = L.mean(0)
    alive = list(range(M))
    pvals = np.zeros(M)
    order = []
    running = 0.0
    while len(alive) > 1:
        a = np.array(alive)
        dbar = Lbar[a] - Lbar[a].mean()
        db = Lb[:, a] - Lb[:, a].mean(1, keepdims=True)
        sd = np.sqrt(((db - dbar) ** 2).mean(0))
        t = dbar / sd
        tmax = t.max()
        tb = ((db - dbar) / sd).max(1)
        p = float((tb >= tmax).mean())
        running = max(running, p)
        worst = a[int(np.argmax(t))]
        pvals[worst] = running
        order.append(worst)
        alive.remove(worst)
    pvals[alive[0]] = 1.0
    order.append(alive[0])
    out = pd.DataFrame({"model": losses.columns, "mean_loss": Lbar, "p_mcs": pvals})
    out["in_mcs"] = out["p_mcs"] >= alpha
    out["elim_order"] = [order.index(i) + 1 for i in range(M)]
    return out.sort_values("mean_loss").reset_index(drop=True)


# ------------------------------------------------------------------
# SPA de Hansen (2005)
# ------------------------------------------------------------------
def spa_test(bench_loss, model_losses: pd.DataFrame, B: int = 2000, seed: int = 42) -> dict:
    """Superior Predictive Ability: ¿algún modelo supera genuinamente al benchmark?

    d_kt = L_bench,t − L_k,t (positivo = el modelo k es mejor). H0: max_k E[d_k] ≤ 0.
    Devuelve p-valores consistente (c), inferior (l) y superior (u, = Reality Check).
    """
    Lb = np.asarray(bench_loss, float)
    D = Lb[:, None] - model_losses.to_numpy(float)
    n, M = D.shape
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, B, _block_len(n), rng)
    dbar = D.mean(0)
    Db = D[idx].mean(1)  # B x M
    omega = np.sqrt(n * ((Db - dbar) ** 2).mean(0))
    t_spa = max(0.0, np.max(np.sqrt(n) * dbar / omega))
    # Recentrado de Hansen: Z* = d̄* − d̄ + μ̂, con
    #   μ̂_u = 0 (configuración menos favorable, equivale al Reality Check de White),
    #   μ̂_c = d̄ si √n d̄/ω < −sqrt(2 log log n) (modelos claramente peores), 0 en otro caso,
    #   μ̂_l = min(d̄, 0).
    thresh = -np.sqrt(2 * np.log(np.log(n)))
    mu_c = np.where(np.sqrt(n) * dbar / omega < thresh, dbar, 0.0)
    out = {}
    for name, mu in {"l": np.minimum(dbar, 0.0), "c": mu_c, "u": np.zeros(M)}.items():
        z = np.sqrt(n) * (Db - dbar + mu) / omega
        tb = np.maximum(0, z.max(1))
        out[f"p_{name}"] = float((tb >= t_spa).mean())
    out["t_spa"] = float(t_spa)
    out["best_model"] = str(model_losses.columns[int(np.argmax(dbar))])
    return out
