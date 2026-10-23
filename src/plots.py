"""Funciones reutilizables de graficación (Plotly) para el dashboard.

Paleta limitada a cinco colores (guía 7.4): azul marino (principal), azul (secundario),
ámbar (acento: modelo original y alertas), gris (contexto) y azul claro (relleno).
El modelo seleccionado se resalta y el resto queda en gris, en vez de un color por modelo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

NAVY, BLUE, AMBER, GRAY, LIGHT = "#0D2B4F", "#2A78D6", "#D9892B", "#8A8A84", "#B7D3F6"
PALETTE = [NAVY, BLUE, AMBER, GRAY, LIGHT]
INK, INK2, GRID = "#16202C", "#4A5563", "#E3E5E8"
SPLIT_COLORS = {"Entrenamiento": GRAY, "Validación": BLUE, "Prueba": AMBER}

pio.templates["costo"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif", size=13, color=INK2),
    title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
    paper_bgcolor="white", plot_bgcolor="white", colorway=PALETTE,
    margin=dict(l=60, r=20, t=50, b=50),
    xaxis=dict(gridcolor=GRID, linecolor="#C7CBD1", zeroline=False, ticks="outside", tickcolor="#C7CBD1",
               title=dict(standoff=8)),
    yaxis=dict(gridcolor=GRID, linecolor="#C7CBD1", zeroline=False, title=dict(standoff=8)),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.22, x=0),
    hoverlabel=dict(bgcolor="white", font_size=13, bordercolor=GRID),
))
pio.templates.default = "costo"


def empty(msg: str, h: int = 320) -> go.Figure:
    f = go.Figure()
    f.add_annotation(text=msg, showarrow=False, font=dict(size=14, color=GRAY), x=0.5, y=0.5, xref="paper", yref="paper")
    f.update_layout(height=h, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return f


def target_hist(y: pd.Series, title="Distribución del costo") -> go.Figure:
    f = go.Figure(go.Histogram(x=y, nbinsx=50, marker_color=BLUE, opacity=0.9,
                               hovertemplate="%{x} USD/Bbl<br>%{y} filas<extra></extra>"))
    f.add_vline(x=float(y.median()), line=dict(color=AMBER, dash="dash", width=2),
                annotation_text=f"mediana {y.median():.1f}", annotation_font_color=AMBER)
    f.update_layout(title=title, xaxis_title="Costo equivalente Brent (USD/Bbl)", yaxis_title="Filas (activo-año)",
                    height=340, bargap=0.03, showlegend=False)
    return f


def box_by(d: pd.DataFrame, cat: str, y: str, label: str) -> go.Figure:
    order = d.groupby(cat)[y].median().sort_values().index.tolist()
    f = go.Figure()
    for c in order:
        f.add_trace(go.Box(x=d.loc[d[cat] == c, y], name=str(c), marker_color=NAVY, line=dict(width=1.3),
                           fillcolor=LIGHT, boxmean=True, orientation="h",
                           hovertemplate=f"{c}<br>%{{x:.2f}} USD/Bbl<extra></extra>"))
    f.update_layout(title=f"Costo por {label}", xaxis_title="USD/Bbl", height=max(320, 34 * len(order) + 110),
                    showlegend=False, margin=dict(l=170))
    return f


def explorer(d: pd.DataFrame, x: str, y: str, color: str | None, labels: dict) -> go.Figure:
    f = go.Figure()
    if color is not None and not pd.api.types.is_numeric_dtype(d[color]):
        cats = d[color].value_counts().index.tolist()
        cols = PALETTE[:4]
        top = cats[:4]
        for i, c in enumerate(top):
            s = d[d[color] == c]
            f.add_trace(go.Scattergl(x=s[x], y=s[y], mode="markers", name=str(c),
                                     marker=dict(size=6, color=cols[i], opacity=0.6)))
        rest = d[~d[color].isin(top)]
        if len(rest):
            f.add_trace(go.Scattergl(x=rest[x], y=rest[y], mode="markers", name="otras",
                                     marker=dict(size=5, color=LIGHT, opacity=0.6)))
    else:
        f.add_trace(go.Scattergl(x=d[x], y=d[y], mode="markers", marker=dict(size=6, color=BLUE, opacity=0.55),
                                 showlegend=False))
    f.update_layout(title=f"{labels.get(y, y)} vs {labels.get(x, x)}", xaxis_title=labels.get(x, x),
                    yaxis_title=labels.get(y, y), height=420)
    return f


def corr_heatmap(c: pd.DataFrame, labels: dict) -> go.Figure:
    names = [labels.get(k, k) for k in c.columns]
    f = go.Figure(go.Heatmap(z=c.values, x=names, y=names, zmin=-1, zmax=1,
                             colorscale=[[0, AMBER], [0.5, "#F4F4F2"], [1, NAVY]],
                             text=np.round(c.values, 2), texttemplate="%{text}",
                             hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>",
                             colorbar=dict(title="ρ", thickness=12)))
    f.update_layout(title="Correlación de Spearman", height=420, yaxis=dict(autorange="reversed"),
                    margin=dict(l=170, b=120))
    return f


def missing_bar(pct: pd.Series, title: str) -> go.Figure:
    pct = pct.sort_values()
    f = go.Figure(go.Bar(x=pct.values, y=pct.index, orientation="h", marker_color=BLUE,
                         text=[f"{v:.0f} %" for v in pct.values], textposition="outside",
                         hovertemplate="%{y}: %{x:.1f} %<extra></extra>"))
    f.update_layout(title=title, xaxis_title="% de filas sin dato", xaxis=dict(range=[0, 115]),
                    height=max(300, 26 * len(pct) + 100), margin=dict(l=190), showlegend=False)
    return f


def year_trend(d: pd.DataFrame, y: str) -> go.Figure:
    g = d.groupby("Year")[y].agg(["median", lambda s: s.quantile(.25), lambda s: s.quantile(.75), "size"])
    g.columns = ["med", "q1", "q3", "n"]
    f = go.Figure()
    f.add_trace(go.Scatter(x=list(g.index) + list(g.index[::-1]), y=list(g.q3) + list(g.q1[::-1]), fill="toself",
                           fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), name="rango intercuartil",
                           hoverinfo="skip"))
    f.add_trace(go.Scatter(x=g.index, y=g.med, mode="lines+markers", line=dict(color=NAVY, width=2.5),
                           name="mediana", customdata=g.n, hovertemplate="%{x}: %{y:.2f} USD/Bbl (n=%{customdata})<extra></extra>"))
    f.update_layout(title="Costo por año", xaxis_title="Año", yaxis_title="USD/Bbl", height=340)
    return f


def real_vs_pred(p: pd.DataFrame) -> go.Figure:
    f = go.Figure()
    for split in ["Entrenamiento", "Validación", "Prueba"]:
        s = p[p.split == split]
        if len(s):
            f.add_trace(go.Scattergl(x=s.y, y=s.pred, mode="markers", name=split,
                                     marker=dict(size=5, color=SPLIT_COLORS[split], opacity=0.5)))
    mx = float(p.y.max()) * 1.03 if len(p) else 1
    f.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=INK2, dash="dash", width=1),
                           name="y = ŷ"))
    f.update_layout(title="Real vs predicho", xaxis_title="Real (USD/Bbl)", yaxis_title="Predicho (USD/Bbl)", height=400)
    return f


def resid_vs_pred(te: pd.DataFrame) -> go.Figure:
    e = te.y - te.pred
    f = go.Figure(go.Scattergl(x=te.pred, y=e, mode="markers", marker=dict(size=5, color=BLUE, opacity=0.55),
                               hovertemplate="ŷ %{x:.2f} · e %{y:.2f}<extra></extra>"))
    f.add_hline(y=0, line=dict(color=INK2, dash="dash", width=1))
    f.update_layout(title="Residuo vs predicción (prueba)", xaxis_title="Predicho (USD/Bbl)",
                    yaxis_title="Residuo y − ŷ (USD/Bbl)", height=340, showlegend=False)
    return f


def resid_hist(te: pd.DataFrame) -> go.Figure:
    e = te.y - te.pred
    f = go.Figure(go.Histogram(x=e, nbinsx=50, marker_color=BLUE))
    f.update_layout(title=f"Histograma de residuos (prueba): media {e.mean():.2f}, d.e. {e.std():.2f}",
                    xaxis_title="Residuo (USD/Bbl)", yaxis_title="Filas", height=340, bargap=0.03, showlegend=False)
    return f


def acf_bar(a: pd.DataFrame) -> go.Figure:
    a = a[a.lag > 0]
    f = go.Figure(go.Bar(x=a.lag, y=a.acf, marker_color=NAVY, width=0.4,
                         hovertemplate="rezago %{x}: %{y:.2f}<extra></extra>"))
    band = float(a.band.iloc[0]) if len(a) else 0.07
    f.add_hrect(y0=-band, y1=band, fillcolor=BLUE, opacity=0.12, line_width=0)
    f.update_layout(title="ACF de residuos (prueba, orden activo-año)", xaxis_title="Rezago", yaxis_title="Autocorrelación",
                    height=340, showlegend=False)
    return f


def shap_global_bar(s: pd.Series, title: str) -> go.Figure:
    s = s.sort_values()
    f = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=NAVY,
                         text=[f"{v:.2f}" for v in s.values], textposition="outside",
                         hovertemplate="%{y}: %{x:.2f} USD/Bbl<extra></extra>"))
    f.update_layout(title=title, xaxis_title="Media |SHAP| (USD/Bbl)", height=320, margin=dict(l=130),
                    xaxis=dict(range=[0, float(s.max()) * 1.25 if len(s) else 1]), showlegend=False)
    return f


def shap_waterfall(phi: pd.Series, base: float, pred: float) -> go.Figure:
    phi = phi.reindex(phi.abs().sort_values(ascending=False).index)
    f = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * len(phi) + ["total"],
        y=["Valor base"] + list(phi.index) + ["Predicción"], x=[base] + list(phi.values) + [pred],
        text=[f"{base:.2f}"] + [f"{v:+.2f}" for v in phi.values] + [f"{pred:.2f}"], textposition="outside",
        increasing=dict(marker=dict(color=AMBER)), decreasing=dict(marker=dict(color=BLUE)),
        totals=dict(marker=dict(color=NAVY)), connector=dict(line=dict(color=GRID))))
    f.update_layout(title="Explicación local (Shapley por grupos de variables)", xaxis_title="USD/Bbl",
                    height=360, margin=dict(l=130, r=40), yaxis=dict(autorange="reversed"), showlegend=False)
    return f


def models_bar(s: pd.Series, selected: str, labels: dict) -> go.Figure:
    s = s.sort_values()
    f = go.Figure(go.Bar(x=s.values, y=[labels.get(k, k) for k in s.index], orientation="h",
                         marker_color=[AMBER if k == selected else LIGHT for k in s.index],
                         text=[f"{v:.2f}" for v in s.values], textposition="outside",
                         hovertemplate="%{y}: %{x:.2f} USD/Bbl<extra></extra>"))
    f.update_layout(title="Predicción de cada modelo para el mismo activo", xaxis_title="USD/Bbl", height=330,
                    margin=dict(l=170, r=40), xaxis=dict(range=[0, float(s.max()) * 1.2]), showlegend=False)
    return f


def convergence(cv: pd.DataFrame) -> go.Figure:
    f = go.Figure()
    for col, name, color in [("train_rmse", "Entrenamiento", GRAY), ("test_rmse", "Prueba (139 activos)", AMBER)]:
        if col in cv:
            f.add_trace(go.Scatter(x=cv.iteration, y=cv[col], mode="lines", name=name, line=dict(color=color, width=2.5)))
    for col, name, color in [("rf_test_rmse", "Random Forest (prueba)", BLUE), ("xgb_test_rmse", "XGBoost (prueba)", NAVY)]:
        if col in cv:
            f.add_hline(y=float(cv[col].iloc[0]), line=dict(color=color, dash="dot", width=1.5),
                        annotation_text=name, annotation_font_color=color, annotation_position="top right")
    f.update_layout(title="Curva de convergencia del HGSB (RMSE vs número de árboles)", xaxis_title="Árboles",
                    yaxis_title="RMSE (USD/Bbl)", height=380)
    return f


def mcs_bar(m: pd.DataFrame) -> go.Figure:
    m = m.sort_values("RMSE").head(15)
    colors = [AMBER if "HGSB" in k else (NAVY if i else BLUE) for k, i in zip(m.model, m.in_mcs)]
    f = go.Figure(go.Bar(x=m.RMSE, y=m.model, orientation="h", marker_color=colors,
                         customdata=np.c_[m.p_mcs], text=[f"p={p:.2f}" for p in m.p_mcs], textposition="outside",
                         hovertemplate="%{y}<br>RMSE %{x:.3f} · p_MCS %{customdata[0]:.3f}<extra></extra>"))
    f.update_layout(title="Model Confidence Set (α = 0.10) en el hold-out", xaxis_title="RMSE de prueba (USD/Bbl)",
                    height=460, margin=dict(l=210, r=60), yaxis=dict(autorange="reversed"), showlegend=False)
    return f
