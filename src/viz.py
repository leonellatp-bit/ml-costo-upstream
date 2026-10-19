"""Estilo gráfico común (notebooks y Dash): paleta categórica validada y tinta neutra.

El color sigue a la entidad (cada modelo y cada optimizador tiene un color
fijo en todos los gráficos), nunca a su posición en un ranking.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]

MODEL_COLORS = {
    "KNN": SERIES[0], "Ridge": SERIES[1], "Lasso": SERIES[2], "DecisionTree": SERIES[3],
    "RandomForest": SERIES[4], "XGBoost": SERIES[5], "SVR": SERIES[6],
}
METHOD_COLORS = {"Grid": SERIES[0], "Random": SERIES[1], "Bayesiana": SERIES[2], "Genetica": SERIES[6]}
SPLIT_COLORS = {"Entrenamiento": SERIES[0], "Validación": SERIES[1], "Prueba": SERIES[2]}


def set_style() -> None:
    """Estilo matplotlib: líneas finas, grilla recesiva, sin bordes superiores/derechos."""
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 140, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 2,
        "lines.markersize": 5, "axes.prop_cycle": mpl.cycler(color=SERIES),
        "font.family": "sans-serif",
    })
    plt.rcParams["image.cmap"] = "Blues"
