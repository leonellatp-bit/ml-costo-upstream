"""Dashboard final del proyecto: costo de producción equivalente Brent en activos upstream.

Ejecutar desde la raíz del proyecto (después de run_nested.py, los notebooks y build_data.py):
    python dashboard/app.py
y abrir http://127.0.0.1:8050
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from dash import Dash, Input, Output, dash_table, dcc, html  # noqa: E402

from src import config as C  # noqa: E402
from src.viz import INK, INK2, MUTED, GRID, AXIS, SURFACE, SERIES, MODEL_COLORS, METHOD_COLORS, SPLIT_COLORS  # noqa: E402

R = C.RESULTS_DIR
D = R / "dashboard"
MODEL_ORDER = ["KNN", "Ridge", "Lasso", "DecisionTree", "RandomForest", "XGBoost", "SVR"]
METHODS = ["Grid", "Random", "Bayesiana", "Genetica"]


# ------------------------------------------------------------------
# Carga de datos
# ------------------------------------------------------------------
def _csv(path, **kw):
    try:
        return pd.read_csv(path, **kw)
    except FileNotFoundError:
        return pd.DataFrame()


def _json(path):
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return {}


def _data_path() -> Path:
    """Ubicación del CSV crudo (licencia S&P Global, nunca se sube a GitHub).

    Orden de búsqueda: variable de entorno DATA_PATH -> Secret File de Render
    (/etc/secrets/<archivo>) -> carpeta data/ local del proyecto.
    """
    for cand in [os.environ.get("DATA_PATH"), f"/etc/secrets/{C.DATA_PATH.name}", str(C.DATA_PATH)]:
        if cand and Path(cand).exists():
            return Path(cand)
    raise FileNotFoundError(
        f"No se encontró {C.DATA_PATH.name}. En local: cópialo en data/. En Render: súbelo como Secret File.")


raw = pd.read_csv(_data_path(), sep=";")
data = raw.dropna(subset=[C.TARGET])
mt = _csv(R / "master_table.csv")
mtA = mt[mt.scenario == "A_estructural"] if len(mt) else mt
mtB = mt[mt.scenario == "B_con_BD"] if len(mt) else mt
anytime = _csv(D / "anytime_median.csv")
ga = _csv(D / "ga_diversity.csv")
searches = _csv(D / "searches.csv")
preds = _csv(D / "predictions.csv")
racf = _csv(D / "residual_acf.csv")
test_metrics = _csv(R / "test_metrics.csv")
scaling = _csv(R / "scaling.csv")
svo = _csv(R / "standard_vs_optimized.csv")
sh = _csv(R / "successive_halving.csv")
par = _csv(R / "parallel.csv")
shap_imp = _csv(R / "shap_importance.csv", index_col=0)
mcs = _csv(R / "mcs.csv")
directed = _csv(R / "directed_tests.csv")
dm = _csv(R / "dm_tests.csv")
bca = _csv(R / "bca_ci.csv")
fn = _json(R / "friedman_nemenyi.json")
spa = _json(R / "spa.json")
opts = _json(D / "predictor_options.json")

best = mtA.loc[mtA.groupby("model")["outer_rmse_mean"].idxmin()].set_index("model") if len(mtA) else pd.DataFrame()
BEST = opts.get("best_model", best["outer_rmse_mean"].idxmin() if len(best) else "RandomForest")
models = {}
for m in MODEL_ORDER:
    p = D / f"model_{m}.pkl"
    if p.exists():
        models[m] = pickle.load(open(p, "rb"))


# ------------------------------------------------------------------
# Estilo de figuras
# ------------------------------------------------------------------
def style(fig, h=360, legend=True):
    fig.update_layout(
        template="none", height=h, margin=dict(l=55, r=20, t=40, b=45),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12, color=INK2),
        title_font=dict(size=13.5, color=INK), showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=GRID),
        colorway=SERIES,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def card(title, children, sub=None):
    return html.Div([html.H3(title), html.P(sub, className="sub") if sub else None, *children], className="card")


def kpi(label, value, note=""):
    return html.Div([html.Div(label, className="label"), html.Div(value, className="value"),
                     html.Div(note, className="note")], className="kpi")


def table(df, page_size=12, fmt=3):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].round(fmt)
    return dash_table.DataTable(
        d.to_dict("records"), [{"name": c, "id": c} for c in d.columns], page_size=page_size,
        sort_action="native", style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "system-ui, sans-serif", "fontSize": 12.5, "padding": "6px 8px",
                    "border": "none", "borderBottom": f"1px solid {GRID}", "textAlign": "left",
                    "backgroundColor": SURFACE, "color": INK, "maxWidth": 340, "whiteSpace": "normal"},
        style_header={"fontWeight": 600, "color": MUTED, "backgroundColor": SURFACE,
                      "borderBottom": f"1px solid {AXIS}"},
    )


def missing(msg="Ejecute el pipeline (run_nested.py, notebooks y build_data.py) para ver esta sección."):
    return html.Div(msg, className="callout warn")


# ------------------------------------------------------------------
# Pestaña 1: resumen
# ------------------------------------------------------------------
def tab_resumen():
    k = []
    if len(test_metrics):
        tm = test_metrics.set_index("Modelo")
        bt = tm.loc[BEST]
        base = tm.loc["SVR lineal (E1)"] if "SVR lineal (E1)" in tm.index else None
        dummy = tm.loc["Dummy (media)"] if "Dummy (media)" in tm.index else None
        k = [
            kpi("Mejor modelo", BEST, f"optimizado con {best.loc[BEST, 'method']}"),
            kpi("RMSE en prueba", f"{bt['RMSE']:.2f} USD/Bbl",
                f"vs {base['RMSE']:.2f} del SVR lineal sin BD" if base is not None else ""),
            kpi("MAE en prueba", f"{bt['MAE']:.2f} USD/Bbl", f"MAPE {bt['MAPE %']:.1f} %"),
            kpi("R² en prueba", f"{bt['R²']:.3f}", f"Dummy: {dummy['R²']:.3f}" if dummy is not None else ""),
        ]
    findings = html.Ul([
        html.Li(["El target cumple una ", html.B("identidad contable exacta"),
                 ": BrentEquivalentProductionCost = WellheadProductionCost + BrentDifferential. "
                 "Por eso el modelo principal excluye BrentDifferential (escenario A) y el escenario B lo usa solo como sensibilidad."]),
        html.Li("La varianza del costo es sobre todo entre activos (σ total 5.5 vs σ dentro de un activo 1.4): "
                "todas las particiones son por activo (GroupKFold) para medir generalización a activos nuevos."),
        html.Li("Se entrenaron y evaluaron 28 combinaciones (7 modelos × 4 optimizadores) con validación cruzada anidada "
                "10×3 por grupos, más 7 modelos del escenario B: 35 combinaciones y 385 búsquedas."),
        html.Li("La comparación estadística sigue la secuencia Friedman → Nemenyi (CD) → MCS → Giacomini-White / Clark-West "
                "→ Diebold-Mariano (HLN) con bootstrap estacionario, tamaños de efecto e IC BCa."),
    ], className="findings")
    data_k = [kpi("Observaciones", f"{len(raw):,}", f"{len(data):,} con target"),
              kpi("Activos", f"{raw[C.GROUP].nunique()}", f"{data[C.GROUP].nunique()} con costo observado"),
              kpi("Países · Regiones", f"{raw.CountryName.nunique()} · {raw.RegionName.nunique()}", "cobertura global"),
              kpi("Periodo", f"{raw.Year.min()}–{raw.Year.max()}", "panel activo-año")]
    # Imagen de portada: si existe una foto propia en assets/ (hero_oilfield.jpg/.png) se usa; si no, la ilustración SVG
    assets = Path(__file__).parent / "assets"
    hero = next((f for f in ["hero_oilfield.jpg", "hero_oilfield.jpeg", "hero_oilfield.png", "hero_oilfield.svg"]
                 if (assets / f).exists()), None)
    hero_block = html.Div([
        html.Img(src=app.get_asset_url(hero), alt="Campo petrolero: balancines, tanques y plataforma offshore",
                 className="hero-img") if hero else None,
        html.Div([
            html.Div("Contexto del problema", className="hero-kicker"),
            html.H2("¿Cuánto cuesta producir un barril en un activo que aún no conocemos?"),
            html.P("El costo de producción por barril define qué activos son rentables a un precio dado del Brent, "
                   "cuáles se desarrollan y cómo se asignan presupuestos de OPEX en un portafolio upstream."),
        ], className="hero-text"),
    ], className="hero")
    why = card("Por qué es importante", [
        html.Div([
            html.Div([html.H4("Rentabilidad y punto de equilibrio"),
                      html.P("El costo equivalente Brent permite comparar activos de distintas calidades de crudo contra un mismo "
                             "precio de referencia: si el Brent cae por debajo del costo, el barril marginal deja de ser rentable.")],
                     className="why-item"),
            html.Div([html.H4("Decisiones de inversión"),
                      html.P("Al evaluar adquisiciones, nuevos desarrollos o farm-ins, rara vez hay historia de costos del activo; "
                             "un estimado basado en sus características reduce la incertidumbre del caso de negocio.")],
                     className="why-item"),
            html.Div([html.H4("Planeación financiera y OPEX"),
                      html.P("Un modelo de referencia ayuda a detectar activos cuyo costo se desvía de lo esperado para su región, "
                             "tipo de reserva y ubicación, y a priorizar iniciativas de eficiencia.")],
                     className="why-item"),
            html.Div([html.H4("Heterogeneidad global"),
                      html.P("En los datos el costo va de ~3 a ~54 USD/Bbl: la geografía, el tipo de reserva (convencional vs no "
                             "convencional) y la ubicación (onshore/offshore) explican diferencias de un orden de magnitud.")],
                     className="why-item"),
        ], className="grid g4 why"),
    ])
    return html.Div([
        hero_block,
        why,
        html.Div(data_k, className="grid g4", style={"marginBottom": "16px"}),
        card("Pregunta de investigación", [
            html.P("¿Es posible predecir el costo de producción equivalente Brent (USD/Bbl) de un activo upstream de petróleo "
                   "que el modelo nunca ha visto, a partir de sus características geográficas, operativas y temporales?"),
            html.Div(k, className="grid g4"),
        ]),
        card("Hallazgos principales", [findings]),
    ])


# ------------------------------------------------------------------
# Pestaña 2: datos y EDA
# ------------------------------------------------------------------
def tab_eda():
    d = raw.dropna(subset=[C.TARGET, "WellheadProductionCost", "BrentDifferential"])
    f_id = px.scatter(d, x=d["WellheadProductionCost"] + d["BrentDifferential"], y=C.TARGET, opacity=0.45,
                      labels={"x": "WellheadProductionCost + BrentDifferential", C.TARGET: "BrentEquivalentProductionCost"},
                      hover_data=["CountryName", "Year"])
    f_id.update_traces(marker=dict(size=5, color=SERIES[0]))
    f_id.add_trace(go.Scatter(x=[0, 55], y=[0, 55], mode="lines", line=dict(color=MUTED, dash="dash", width=1), name="y = x"))
    style(f_id, 340, legend=False).update_layout(title="El target es la suma exacta de dos columnas")

    m = raw[raw.Year > 2014]
    miss = (m["Production"].isna().groupby(m["RegionName"]).mean() * 100).sort_values()
    f_miss = go.Figure(go.Bar(x=miss.values, y=miss.index, orientation="h", marker_color=SERIES[0],
                              hovertemplate="%{y}: %{x:.0f} %<extra></extra>"))
    style(f_miss, 340, legend=False).update_layout(title="% de filas sin Production (2015–2023)", xaxis_title="%",
                                                    margin=dict(l=210))
    regions = sorted(raw.RegionName.unique())
    return html.Div([
        html.Div([
            card("Identidad contable", [dcc.Graph(figure=f_id, config={"displayModeBar": False})],
                 "Error máximo 0.01 (redondeo) en el 100 % de las filas completas"),
            card("Mecanismo de faltantes", [dcc.Graph(figure=f_miss, config={"displayModeBar": False})],
                 "Production falta por región (cobertura de la fuente): MAR condicionado a la región"),
        ], className="grid g2"),
        card("Explorador del costo", [
            html.Div([
                html.Div([html.Label("Regiones"), dcc.Dropdown(regions, [], multi=True, id="eda-reg", placeholder="Todas")],
                         style={"flex": "2"}),
                html.Div([html.Label("Agrupar por"), dcc.Dropdown(
                    ["RegionName", "ReserveTypeName", "SupplySubTypeName", "ReserveLocationName", "MemberOpec"],
                    "SupplySubTypeName", id="eda-cat", clearable=False)], style={"flex": "1"}),
            ], className="controls"),
            html.Div([dcc.Graph(id="eda-box", config={"displayModeBar": False}),
                      dcc.Graph(id="eda-year", config={"displayModeBar": False})], className="grid g2"),
        ], "Filtra por región y elige la variable categórica"),
    ])


# ------------------------------------------------------------------
# Pestaña 3: modelos y optimización
# ------------------------------------------------------------------
def short_params(js):
    try:
        d = json.loads(js)
    except Exception:
        return js
    return ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())


def tab_modelos():
    if not len(mtA):
        return missing()
    t = mtA.copy()
    show = pd.DataFrame({
        "Modelo": t.model, "Optimizador": t.method,
        "RMSE externo": [f"{a:.3f} ± {b:.3f}" for a, b in zip(t.outer_rmse_mean, t.outer_rmse_std)],
        "MAE externo": [f"{a:.3f} ± {b:.3f}" for a, b in zip(t.outer_mae_mean, t.outer_mae_std)],
        "R² externo": [f"{a:.3f} ± {b:.3f}" for a, b in zip(t.outer_r2_mean, t.outer_r2_std)],
        "RMSE prueba": t.test_rmse.round(3), "Evaluaciones": t.total_evals,
        "Tiempo total (s)": t.search_time_total_s.round(0), "Hiperparámetros finales": t.final_params.map(short_params),
    })
    sc = searches.copy()
    f_cost = px.scatter(sc, x="search_time", y="best_inner_rmse", color="method", log_x=True, opacity=0.65,
                        color_discrete_map=METHOD_COLORS, hover_data=["model", "fold"],
                        labels={"search_time": "tiempo de búsqueda (s, log)", "best_inner_rmse": "mejor RMSE interno", "method": "Optimizador"})
    f_cost.update_traces(marker=dict(size=7, line=dict(color="white", width=1)))
    style(f_cost, 360).update_layout(title="Calidad alcanzada vs costo de cada búsqueda")
    f_ga = go.Figure()
    for m in MODEL_ORDER:
        s = ga[(ga.model == m) & (ga.gen <= 10)]
        f_ga.add_trace(go.Scatter(x=s.gen, y=s.diversity, mode="lines+markers", name=m, line=dict(color=MODEL_COLORS[m])))
    style(f_ga, 360).update_layout(title="Algoritmo genético: diversidad por generación", xaxis_title="generación",
                                   yaxis_title="distancia media entre genomas")
    return html.Div([
        card("RMSE del bucle externo por modelo y optimizador", [
            html.Div([html.Div([html.Label("Métrica"), dcc.RadioItems(
                [{"label": " RMSE", "value": "rmse"}, {"label": " MAE", "value": "mae"}, {"label": " R²", "value": "r2"},
                 {"label": " MAPE", "value": "mape"}], "rmse", id="heat-metric", inline=True,
                inputStyle={"marginLeft": "10px"})])], className="controls"),
            dcc.Graph(id="heat", config={"displayModeBar": False}),
        ], "Media de 10 pliegues externos (GroupKFold por activo); el texto incluye ± desviación estándar"),
        html.Div([
            card("Curvas anytime", [
                dcc.Dropdown(MODEL_ORDER, BEST, id="any-model", clearable=False, style={"maxWidth": "260px"}),
                dcc.Graph(id="anytime", config={"displayModeBar": False})],
                "Mejor RMSE interno encontrado hasta la evaluación t (mediana de 10 pliegues)"),
            card("Costo vs calidad", [dcc.Graph(figure=f_cost, config={"displayModeBar": False})],
                 "Cada punto es una búsqueda de 30 evaluaciones"),
        ], className="grid g2"),
        card("Diversidad genética", [dcc.Graph(figure=f_ga, config={"displayModeBar": False})],
             "Caída rápida = convergencia temprana de la población"),
        card("Tabla maestra (escenario A)", [table(show, 28)], "Ordenable por cualquier columna"),
    ])


# ------------------------------------------------------------------
# Pestaña 4: optimización computacional
# ------------------------------------------------------------------
def tab_computacional():
    if not len(scaling):
        return missing()
    f_sc = make_scaling_fig("fit_s", "Entrenamiento: tiempo vs n (log-log)")
    f_pr = make_scaling_fig("predict_s", "Inferencia (500 consultas): tiempo vs n")
    s = svo.copy()
    s["etiqueta"] = s["model"] + " · " + s["variant"]
    f_svo = go.Figure(go.Bar(x=s["speedup_fit"], y=s["etiqueta"], orientation="h",
                             marker_color=[MODEL_COLORS.get(m, SERIES[0]) for m in s.model],
                             customdata=np.c_[s["rmse"], s["delta_rmse"]],
                             hovertemplate="%{y}<br>speedup %{x:.2f}×<br>RMSE %{customdata[0]:.3f} (Δ %{customdata[1]:+.3f})<extra></extra>"))
    f_svo.add_vline(x=1, line=dict(color=MUTED, dash="dash"))
    style(f_svo, 520, legend=False).update_layout(title="Speedup de entrenamiento frente a la versión estándar",
                                                  xaxis_type="log", xaxis_title="speedup (log; > 1 = más rápido)",
                                                  margin=dict(l=380))
    cols = ["model", "variant", "complexity", "fit_s", "predict_s", "peak_mem_MB", "rmse", "speedup_fit", "delta_rmse"]
    return html.Div([
        html.Div([card("Escalamiento", [dcc.Graph(figure=f_sc, config={"displayModeBar": False})], "Pendiente ≈ exponente empírico"),
                  card("Inferencia", [dcc.Graph(figure=f_pr, config={"displayModeBar": False})])], className="grid g2"),
        card("Estándar vs optimizado", [dcc.Graph(figure=f_svo, config={"displayModeBar": False}), table(s[cols], 16, 4)]),
        html.Div([card("Successive Halving (η = 3)", [table(sh.drop(columns=[c for c in sh.columns if c in ("candidates_per_round", "resources_per_round")]), 6, 3)],
                       "Fidelidad = número de árboles"),
                  card("Paralelización (joblib)", [table(par, 6, 3)], "Validación cruzada de Random Forest")], className="grid g2"),
    ])


def make_scaling_fig(col, title):
    f = go.Figure()
    for m in MODEL_ORDER:
        s = scaling[scaling.model == m]
        f.add_trace(go.Scatter(x=s.n, y=s[col], mode="lines+markers", name=m, line=dict(color=MODEL_COLORS[m])))
    style(f, 360).update_layout(title=title, xaxis_type="log", yaxis_type="log", xaxis_title="n", yaxis_title="segundos")
    return f


# ------------------------------------------------------------------
# Pestaña 5: evaluación
# ------------------------------------------------------------------
def tab_evaluacion():
    if not len(test_metrics):
        return missing()
    imp = shap_imp.iloc[:, 0].sort_values().tail(12) if len(shap_imp) else pd.Series(dtype=float)
    f_shap = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=SERIES[0],
                              hovertemplate="%{y}: %{x:.3f} USD/Bbl<extra></extra>"))
    style(f_shap, 400, legend=False).update_layout(title="Importancia global (media |SHAP|, USD/Bbl)", margin=dict(l=190))
    return html.Div([
        card("Métricas en el conjunto de prueba (139 activos no vistos)", [table(fmt_p(test_metrics), 10, 3)],
             "p-valores: White (homocedasticidad), BDS m=2 (i.i.d.), Ljung-Box 10 rezagos (autocorrelación)"),
        card("Ajuste por subconjunto", [
            html.Div([html.Div([html.Label("Modelo"), dcc.Dropdown(MODEL_ORDER, BEST, id="ev-model", clearable=False)]),
                      html.Div([html.Label("Colorear prueba por"), dcc.Dropdown(["split", "RegionName"], "split", id="ev-color", clearable=False)])],
                     className="controls"),
            html.Div([dcc.Graph(id="ev-scatter", config={"displayModeBar": False}),
                      dcc.Graph(id="ev-year", config={"displayModeBar": False})], className="grid g2"),
        ], "Entrenamiento = ajuste en desarrollo · Validación = predicción fuera de pliegue · Prueba = hold-out"),
        html.Div([card("Residuos en prueba", [dcc.Graph(id="ev-resid", config={"displayModeBar": False})]),
                  card("ACF de residuos", [dcc.Graph(id="ev-acf", config={"displayModeBar": False})],
                       "Orden activo-año; banda ±1.96/√n")], className="grid g2"),
        card("Interpretabilidad (SHAP)", [dcc.Graph(figure=f_shap, config={"displayModeBar": False})],
             "Contribuciones One-Hot sumadas por variable original"),
    ])


# ------------------------------------------------------------------
# Pestaña 6: comparación estadística
# ------------------------------------------------------------------
def rank_fig(ranks: dict, cd: float, title: str, colors=None, h=None):
    r = pd.Series(ranks).sort_values()
    f = go.Figure()
    cols = [colors.get(k.split("|")[0], SERIES[0]) if colors else SERIES[0] for k in r.index]
    f.add_trace(go.Scatter(x=r.values, y=list(r.index), mode="markers", marker=dict(size=10, color=cols, line=dict(color="white", width=1.5)),
                           hovertemplate="%{y}: rango medio %{x:.2f}<extra></extra>"))
    best_r = r.iloc[0]
    f.add_vrect(x0=best_r, x1=best_r + cd, fillcolor=SERIES[0], opacity=0.08, line_width=0)
    f.add_annotation(x=best_r + cd, y=1.02, yref="paper", text=f"CD = {cd:.2f}", showarrow=False, font=dict(size=11, color=SERIES[0]))
    style(f, h or max(260, 22 * len(r) + 80), legend=False).update_layout(
        title=title, xaxis_title="rango medio (1 = mejor)", yaxis=dict(autorange="reversed"), margin=dict(l=200))
    return f


def fmt_p(df):
    """Muestra p-valores en notación legible (evita que p < 1e-4 aparezca como 0)."""
    d = df.copy()
    for c in d.columns:
        if c.lower().startswith("p") and d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "< 0.0001" if v < 1e-4 else f"{v:.4f}")
    return d


def mcs_tbl():
    if not len(mcs):
        return mcs
    d = mcs[["model", "RMSE prueba", "p_mcs", "in_mcs"]].copy()
    d.columns = ["Modelo", "RMSE prueba", "p-valor MCS", "¿En el conjunto?"]
    d["¿En el conjunto?"] = d["¿En el conjunto?"].map({True: "Sí", False: "No"})
    return d


def tab_estadistica():
    if not fn:
        return missing()
    fr = fn["friedman"]
    f28 = rank_fig(fn["combos"], fn["cd_combos"], "28 combinaciones: rango medio de Friedman (Nemenyi)", MODEL_COLORS)
    f7 = rank_fig(fn["models"], fn["cd_models"], "7 modelos (mejor optimizador c/u)", MODEL_COLORS, 300)
    f4 = rank_fig(fn["methods"], fn["cd_methods"], "4 optimizadores (70 bloques modelo × pliegue)", METHOD_COLORS, 240)
    f4.data[0].marker.color = [METHOD_COLORS[k] for k in pd.Series(fn["methods"]).sort_values().index]
    names = {"combos": "28 combinaciones", "models": "7 modelos", "methods": "4 optimizadores"}
    fr_tbl = fmt_p(pd.DataFrame([{"nivel": names.get(k, k), "χ²": v["chi2"], "g.l.": int(v["df"]), "p-valor": v["p_value"],
                                  "W de Kendall": v["kendall_W"]} for k, v in fr.items()]))
    b = bca[bca["Métrica"] == "RMSE"].sort_values("Valor") if len(bca) else pd.DataFrame()
    f_bca = go.Figure()
    for _, row in b.iterrows():
        c = MODEL_COLORS.get(row.Modelo.split("|")[0], SERIES[0])
        f_bca.add_trace(go.Scatter(x=[row["IC 95 % inferior"], row["Valor"], row["IC 95 % superior"]], y=[row.Modelo] * 3,
                                   mode="lines+markers", line=dict(color=c, width=3), marker=dict(size=[0, 10, 0], color=c),
                                   hovertemplate=f"{row.Modelo}<br>RMSE {row['Valor']:.3f} [{row['IC 95 % inferior']:.3f}, {row['IC 95 % superior']:.3f}]<extra></extra>"))
    style(f_bca, 320, legend=False).update_layout(title="RMSE en prueba con IC 95 % BCa", margin=dict(l=200), yaxis=dict(autorange="reversed"))
    spa_txt = (f"SPA de Hansen vs SVR lineal (E1): p consistente = {spa.get('p_c', float('nan')):.4f}, "
               f"mejor modelo del universo: {spa.get('best_model', '')}") if spa else ""
    return html.Div([
        card("Etapa 1 · Friedman (ómnibus)", [table(fr_tbl, 5, 4)], "Bloques = pliegues externos"),
        html.Div([card("Etapa 2 · Nemenyi", [dcc.Graph(figure=f28, config={"displayModeBar": False})],
                       "Los modelos dentro de la banda sombreada no difieren significativamente del primero"),
                  html.Div([card("Modelos", [dcc.Graph(figure=f7, config={"displayModeBar": False})]),
                            card("Optimizadores", [dcc.Graph(figure=f4, config={"displayModeBar": False})])])],
                 className="grid g2"),
        card("Regresión · Model Confidence Set (α = 0.10)", [table(mcs_tbl(), 15, 4)],
             "Pérdida cuadrática sobre el hold-out; bootstrap estacionario"),
        card("Comparaciones dirigidas (Giacomini-White / Clark-West)", [table(fmt_p(directed), 8, 3)],
             "p ajustados por Holm (FWER) y Benjamini-Hochberg (FDR)"),
        card("Diebold-Mariano (HLN) y SPA", [table(fmt_p(dm), 4, 3), html.P(spa_txt, className="small")],
             "IC de la diferencia de pérdida por bootstrap estacionario"),
        card("Intervalos BCa", [dcc.Graph(figure=f_bca, config={"displayModeBar": False})]),
    ])


# ------------------------------------------------------------------
# Pestaña 7: predictor
# ------------------------------------------------------------------
def tab_predictor():
    if not models or not opts:
        return missing()
    hier = pd.DataFrame(opts["hierarchy"])
    countries = sorted(hier.CountryName.unique())
    return html.Div([
        html.Div([
            card("Características del activo", [
                html.Div([
                    html.Div([html.Label("País"), dcc.Dropdown(countries, "Colombia" if "Colombia" in countries else countries[0], id="p-country", clearable=False)]),
                    html.Div([html.Label("Cuenca / elemento productivo"), dcc.Dropdown(id="p-pe", clearable=False)]),
                ], className="controls"),
                html.Div([
                    html.Div([html.Label("Tipo de reserva"), dcc.Dropdown(["Conventional", "Unconventional"], "Conventional", id="p-rt", clearable=False)]),
                    html.Div([html.Label("Subtipo"), dcc.Dropdown(sorted(raw.SupplySubTypeName.unique()), "Non-Shale", id="p-st", clearable=False)]),
                    html.Div([html.Label("Ubicación"), dcc.Dropdown(sorted(raw.ReserveLocationName.unique()), "Onshore", id="p-loc", clearable=False)]),
                ], className="controls"),
                html.Div([
                    html.Div([html.Label("Año"), dcc.Slider(2015, 2026, 1, value=2023, id="p-year",
                                                            marks={y: str(y) for y in range(2015, 2027, 2)})], style={"flex": "2"}),
                    html.Div([html.Label("Producción (MBD, vacío = desconocida)"),
                              dcc.Input(id="p-prod", type="number", value=None, min=0, debounce=True, style={"width": "100%", "padding": "6px"})]),
                ], className="controls"),
                html.P(id="p-ctx", className="small"),
            ]),
            card("Costo estimado", [
                html.Div(id="p-out", className="pred-out"),
                html.P(id="p-out-sub", className="small"),
                dcc.Graph(id="p-models", config={"displayModeBar": False}),
            ], "Escenario A: sin BrentDifferential ni costos de cabeza de pozo"),
        ], className="grid g2"),
        html.Div("Los años posteriores a 2023 son extrapolación: el modelo solo vio 2014–2023. "
                 "Las predicciones de países con pocos activos tienen mayor incertidumbre.", className="callout warn"),
    ])


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
app = Dash(__name__, title="Costo upstream · ML", suppress_callback_exceptions=True)
server = app.server  # punto de entrada para gunicorn (despliegue en Render)
TABS = [("Resumen", tab_resumen), ("Datos y EDA", tab_eda), ("Modelos y optimización", tab_modelos),
        ("Optimización computacional", tab_computacional), ("Evaluación", tab_evaluacion),
        ("Comparación estadística", tab_estadistica), ("Predictor", tab_predictor)]
app.layout = html.Div([
    html.Div([
        html.H1("Predicción del costo unitario de producción en activos upstream de Oil & Gas"),
        html.P("Proyecto de Machine Learning · Leonella Torreglosa Portillo · Maestría en Ingeniería Industrial"),
        html.Div([html.Span(t, className="chip") for t in
                  ["Regresión", "7 modelos × 4 optimizadores", "CV anidada 10×3 por activo", "S&P Global · 2014–2023"]], className="meta"),
    ], className="header"),
    html.Div(dcc.Tabs(id="tabs", value="0", children=[
        dcc.Tab(label=n, value=str(i), className="tab", selected_className="tab--selected") for i, (n, _) in enumerate(TABS)
    ]), className="tabs-row"),
    html.Div(id="content", className="container"),
])


@app.callback(Output("content", "children"), Input("tabs", "value"))
def render(tab):
    return TABS[int(tab)][1]()


@app.callback(Output("eda-box", "figure"), Output("eda-year", "figure"), Input("eda-reg", "value"), Input("eda-cat", "value"))
def eda_update(regs, cat):
    d = data if not regs else data[data.RegionName.isin(regs)]
    order = d.groupby(cat)[C.TARGET].median().sort_values().index.tolist()
    f1 = go.Figure()
    for i, c in enumerate(order):
        f1.add_trace(go.Box(y=d.loc[d[cat] == c, C.TARGET], name=str(c), marker_color=SERIES[i % 8], boxmean=True,
                            line=dict(width=1.5)))
    style(f1, 380, legend=False).update_layout(title=f"Costo por {cat}", yaxis_title="USD/Bbl")
    yr = d.groupby(["Year", cat])[C.TARGET].mean().reset_index()
    f2 = go.Figure()
    for i, c in enumerate(order[:8]):
        s = yr[yr[cat] == c]
        f2.add_trace(go.Scatter(x=s.Year, y=s[C.TARGET], mode="lines+markers", name=str(c), line=dict(color=SERIES[i % 8])))
    style(f2, 380).update_layout(title="Costo medio anual", yaxis_title="USD/Bbl", hovermode="x unified")
    return f1, f2


@app.callback(Output("heat", "figure"), Input("heat-metric", "value"))
def heat(metric):
    piv = mtA.pivot(index="model", columns="method", values=f"outer_{metric}_mean").loc[MODEL_ORDER, METHODS]
    sd = mtA.pivot(index="model", columns="method", values=f"outer_{metric}_std").loc[MODEL_ORDER, METHODS]
    txt = [[f"{piv.values[i, j]:.3f}<br>±{sd.values[i, j]:.3f}" for j in range(4)] for i in range(len(MODEL_ORDER))]
    scale = "Blues" if metric == "r2" else "Blues_r"
    f = go.Figure(go.Heatmap(z=piv.values, x=METHODS, y=MODEL_ORDER, text=txt, texttemplate="%{text}", colorscale=scale,
                             hovertemplate="%{y} · %{x}<br>%{z:.3f}<extra></extra>", xgap=2, ygap=2,
                             colorbar=dict(title=metric.upper(), thickness=12)))
    return style(f, 420, legend=False).update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=110))


@app.callback(Output("anytime", "figure"), Input("any-model", "value"))
def any_curve(m):
    f = go.Figure()
    for me in METHODS:
        s = anytime[(anytime.model == m) & (anytime.method == me)]
        f.add_trace(go.Scatter(x=s["eval"], y=s.best_so_far, mode="lines", name=me, line=dict(color=METHOD_COLORS[me], width=2.5)))
    return style(f, 360).update_layout(title=f"{m}: convergencia por optimizador", xaxis_title="evaluación t",
                                       yaxis_title="mejor RMSE interno", hovermode="x unified")


@app.callback(Output("ev-scatter", "figure"), Output("ev-year", "figure"), Output("ev-resid", "figure"), Output("ev-acf", "figure"),
              Input("ev-model", "value"), Input("ev-color", "value"))
def ev(m, color):
    p = preds[preds.model == m]
    f1 = go.Figure()
    for i, (split, s) in enumerate(p.groupby("split", sort=False)):
        if color == "RegionName" and split == "Prueba":
            for j, (rg, ss) in enumerate(s.groupby("RegionName")):
                f1.add_trace(go.Scatter(x=ss.y, y=ss.pred, mode="markers", name=rg, marker=dict(size=5, color=SERIES[j % 8], opacity=0.7)))
        elif color == "split":
            f1.add_trace(go.Scatter(x=s.y, y=s.pred, mode="markers", name=split, marker=dict(size=4.5, color=SPLIT_COLORS[split], opacity=0.45)))
    mx = p.y.max() * 1.03
    f1.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=MUTED, dash="dash", width=1), showlegend=False))
    style(f1, 380).update_layout(title="Real vs predicho", xaxis_title="real (USD/Bbl)", yaxis_title="predicho")
    f2 = go.Figure()
    for split, s in p.groupby("split", sort=False):
        t = s.groupby("Year")[["y", "pred"]].mean()[s.groupby("Year").size() >= 10]
        f2.add_trace(go.Scatter(x=t.index, y=t.y, mode="lines", name=f"{split} real", line=dict(color=SPLIT_COLORS[split], width=2)))
        f2.add_trace(go.Scatter(x=t.index, y=t.pred, mode="lines+markers", name=f"{split} predicho",
                                line=dict(color=SPLIT_COLORS[split], width=2, dash="dash"), marker=dict(size=6)))
    style(f2, 380).update_layout(title="Media anual real vs predicha", xaxis_title="Año", yaxis_title="USD/Bbl", hovermode="x unified")
    te = p[p.split == "Prueba"]
    e = te.y - te.pred
    f3 = go.Figure(go.Histogram(x=e, nbinsx=50, marker_color=SERIES[0], opacity=0.85))
    style(f3, 320, legend=False).update_layout(title=f"Distribución del residuo (media {e.mean():.2f}, d.e. {e.std():.2f})",
                                               xaxis_title="real − predicho (USD/Bbl)", yaxis_title="frecuencia", bargap=0.04)
    a = racf[(racf.model == m) & (racf.lag > 0)]
    f4 = go.Figure(go.Bar(x=a.lag, y=a.acf, marker_color=SERIES[0], width=0.35))
    band = a.band.iloc[0] if len(a) else 0.07
    f4.add_hrect(y0=-band, y1=band, fillcolor=SERIES[0], opacity=0.1, line_width=0)
    style(f4, 320, legend=False).update_layout(title="Autocorrelación de residuos (prueba)", xaxis_title="rezago", yaxis_title="ACF")
    return f1, f2, f3, f4


@app.callback(Output("p-pe", "options"), Output("p-pe", "value"), Output("p-ctx", "children"), Input("p-country", "value"))
def pe_opts(country):
    hier = pd.DataFrame(opts["hierarchy"])
    h = hier[hier.CountryName == country]
    pes = sorted(h.ProductionElementName.unique())
    n_assets = raw.loc[raw.CountryName == country, C.GROUP].nunique()
    ctx = f"Región: {h.RegionName.iloc[0]} · {h.MemberOpec.iloc[0]} · {n_assets} activo(s) de este país en los datos"
    return pes, pes[0], ctx


@app.callback(Output("p-out", "children"), Output("p-out-sub", "children"), Output("p-models", "figure"),
              Input("p-country", "value"), Input("p-pe", "value"), Input("p-rt", "value"), Input("p-st", "value"),
              Input("p-loc", "value"), Input("p-year", "value"), Input("p-prod", "value"))
def predict(country, pe, rt, st, loc, year, prod):
    hier = pd.DataFrame(opts["hierarchy"])
    h = hier[hier.CountryName == country].iloc[0]
    row = pd.DataFrame([{"Year": year, "Production": np.nan if prod in (None, "") else float(prod),
                         "RegionName": h.RegionName, "CountryName": country, "ReserveTypeName": rt,
                         "SupplySubTypeName": st, "ReserveLocationName": loc, "ProductionElementName": pe or h.ProductionElementName,
                         "MemberOpec": h.MemberOpec}])
    out = {m: float(models[m].predict(row)[0]) for m in MODEL_ORDER if m in models}
    s = pd.Series(out).sort_values()
    f = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                         marker_color=[SERIES[0] if k == BEST else "#b7d3f6" for k in s.index],
                         text=[f"{v:.2f}" for v in s.values], textposition="outside",
                         hovertemplate="%{y}: %{x:.2f} USD/Bbl<extra></extra>"))
    style(f, 300, legend=False).update_layout(title="Predicción de cada modelo (USD/Bbl)", margin=dict(l=110, r=40),
                                              xaxis=dict(range=[0, s.max() * 1.18]))
    rmse = test_metrics.set_index("Modelo").loc[BEST, "RMSE"] if len(test_metrics) else np.nan
    return (f"{out[BEST]:.2f} USD/Bbl",
            f"Modelo {BEST}. Error típico en activos nuevos: ±{rmse:.2f} USD/Bbl (RMSE de prueba). "
            f"Rango entre los 7 modelos: {s.min():.2f}–{s.max():.2f}.", f)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
