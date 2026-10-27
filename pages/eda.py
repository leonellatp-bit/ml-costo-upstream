"""Sección 2: EDA interactivo. Los filtros globales actualizan todos los gráficos de la página."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, callback, dcc, html

from src import config as C
from src import plots as P
from src.dash_data import data, raw

dash.register_page(__name__, path="/eda", name="EDA", title="EDA · Costo upstream", order=1)

T = C.TARGET
LABELS = {T: "Costo equivalente Brent (USD/Bbl)", "Year": "Año", "Production": "Producción (MBD)",
          "logProduction": "log(1 + producción)", "WellheadProductionCost": "Costo en cabeza de pozo",
          "BrentDifferential": "Diferencial Brent", "WellheadDevelopmentCost": "Costo de desarrollo (cabeza de pozo)",
          "BrentEquivalentDevelopmentCost": "Costo de desarrollo equiv. Brent",
          "RegionName": "región", "CountryName": "país", "ReserveTypeName": "tipo de reserva",
          "SupplySubTypeName": "subtipo", "ReserveLocationName": "ubicación", "MemberOpec": "OPEP",
          "ProductionElementName": "elemento productivo"}
NUMS = [T, "Year", "logProduction", "WellheadProductionCost", "BrentDifferential",
        "WellheadDevelopmentCost", "BrentEquivalentDevelopmentCost"]
CATS = ["RegionName", "ReserveTypeName", "SupplySubTypeName", "ReserveLocationName", "MemberOpec"]
GCFG = {"displayModeBar": False, "responsive": True}


def _base():
    d = data().copy()
    d["logProduction"] = np.log1p(d["Production"].clip(lower=0))
    return d


def filtered(regions, years, rtypes, locs):
    d = _base()
    if regions:
        d = d[d.RegionName.isin(regions)]
    if years:
        d = d[d.Year.between(years[0], years[1])]
    if rtypes:
        d = d[d.ReserveTypeName.isin(rtypes)]
    if locs:
        d = d[d.ReserveLocationName.isin(locs)]
    return d


def insight(id_):
    return html.P(id=id_, className="insight")


def graph_card(title, gid, iid, extra=None, sub=None):
    return dbc.Card([dbc.CardHeader(title), dbc.CardBody([
        html.P(sub, className="small") if sub else None, extra,
        dcc.Loading(dcc.Graph(id=gid, config=GCFG), type="dot", color=P.BLUE), insight(iid)])], className="mb-3 h-100")


def layout():
    r = raw()
    years = (int(r.Year.min()), int(r.Year.max()))
    filters = dbc.Card(dbc.CardBody(dbc.Row([
        dbc.Col([dbc.Label("Regiones"), dcc.Dropdown(sorted(r.RegionName.unique()), [], multi=True, id="f-reg",
                                                     placeholder="Todas")], md=4),
        dbc.Col([dbc.Label("Años"), dcc.RangeSlider(years[0], years[1], 1, value=list(years), id="f-year",
                                                   marks={y: str(y) for y in range(years[0], years[1] + 1, 3)},
                                                   tooltip={"placement": "bottom"})], md=4),
        dbc.Col([dbc.Label("Tipo de reserva"), dcc.Checklist(sorted(r.ReserveTypeName.unique()), [], id="f-rt",
                                                            inline=True, inputClassName="me-1", labelClassName="me-3")], md=2),
        dbc.Col([dbc.Label("Ubicación"), dcc.Checklist(sorted(r.ReserveLocationName.unique()), [], id="f-loc",
                                                      inputClassName="me-1", labelClassName="me-2")], md=2),
    ])), className="mb-3 filters")
    return html.Div([
        html.H2("Análisis exploratorio", className="page-title"),
        html.P("Los filtros de esta barra actualizan todos los gráficos. Cada gráfico incluye una frase de interpretación "
               "calculada sobre la selección actual.", className="lead-sm"),
        filters,
        html.Div(id="f-count", className="small mb-2"),
        dbc.Row([
            dbc.Col(graph_card("Distribución del objetivo", "g-hist", "i-hist"), lg=6),
            dbc.Col(graph_card("Tendencia temporal", "g-year", "i-year"), lg=6),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(graph_card("Costo por categoría (valores atípicos incluidos)", "g-box", "i-box",
                               extra=dcc.Dropdown([{"label": LABELS[c].capitalize(), "value": c} for c in CATS],
                                                  "SupplySubTypeName", id="box-cat", clearable=False, className="mb-2"),
                               sub="Caja = rango intercuartil; puntos = observaciones fuera de 1.5 × RIC."), lg=6),
            dbc.Col(graph_card("Valores faltantes", "g-miss", "i-miss",
                               extra=dcc.RadioItems([{"label": " por variable", "value": "var"},
                                                     {"label": " Production por región", "value": "reg"}], "reg",
                                                    id="miss-mode", inline=True, inputClassName="me-1",
                                                    labelClassName="me-3", className="mb-2")), lg=6),
        ], className="g-3 mt-1"),
        dbc.Card([dbc.CardHeader("Explorador bivariado"), dbc.CardBody([
            dbc.Row([
                dbc.Col([dbc.Label("Eje x"), dcc.Dropdown([{"label": LABELS[c], "value": c} for c in NUMS], "logProduction",
                                                         id="ex-x", clearable=False)], md=4),
                dbc.Col([dbc.Label("Eje y"), dcc.Dropdown([{"label": LABELS[c], "value": c} for c in NUMS], T,
                                                         id="ex-y", clearable=False)], md=4),
                dbc.Col([dbc.Label("Color"), dcc.Dropdown([{"label": LABELS[c].capitalize(), "value": c} for c in CATS],
                                                         "ReserveLocationName", id="ex-c")], md=4),
            ], className="mb-2"),
            dcc.Loading(dcc.Graph(id="g-ex", config=GCFG), type="dot", color=P.BLUE), insight("i-ex")])],
            className="mb-3 mt-3"),
        dbc.Row([
            dbc.Col(graph_card("Matriz de correlación", "g-corr", "i-corr",
                               sub="Spearman (robusta a asimetría y outliers) sobre las variables numéricas."), lg=7),
            dbc.Col(dbc.Card([dbc.CardHeader("Hallazgo clave: identidad contable"), dbc.CardBody([
                dcc.Graph(id="g-id", config=GCFG), html.P(
                    "El objetivo es exactamente WellheadProductionCost + BrentDifferential (error ≤ 0.01). Por eso ambas "
                    "variables se excluyen como predictoras: usarlas sería fuga de información.", className="insight")])],
                className="mb-3 h-100"), lg=5),
        ], className="g-3"),
    ])


@callback(Output("f-count", "children"), Output("g-hist", "figure"), Output("i-hist", "children"),
          Output("g-year", "figure"), Output("i-year", "children"), Output("g-corr", "figure"), Output("i-corr", "children"),
          Output("g-id", "figure"),
          Input("f-reg", "value"), Input("f-year", "value"), Input("f-rt", "value"), Input("f-loc", "value"))
def update_main(regs, years, rts, locs):
    d = filtered(regs, years, rts, locs)
    if len(d) < 5:
        e = P.empty("Sin datos para la selección")
        return "Sin datos para la selección actual.", e, "", e, "", e, "", e
    y = d[T]
    cnt = f"Selección: {len(d):,} filas · {d[C.GROUP].nunique()} activos · {d.CountryName.nunique()} países".replace(",", ".")
    i1 = (f"Mediana {y.median():.1f} USD/Bbl (media {y.mean():.1f}); asimetría {y.skew():.2f}: "
          + ("cola derecha pesada, por eso se reportan MAE y MAPE además del RMSE." if y.skew() > 0.5
             else "distribución aproximadamente simétrica en esta selección."))
    cnt_y = d.groupby("Year")[T].size()
    g = d.groupby("Year")[T].median()[cnt_y >= 30]  # años con pocas filas (p. ej. 2014) no definen la tendencia
    i2 = (f"La mediana va de {g.iloc[0]:.1f} ({g.index[0]}) a {g.iloc[-1]:.1f} ({g.index[-1]}); mínimo en {g.idxmin()} "
          f"({g.min():.1f}); se omiten años con menos de 30 filas. El año entra como predictor para capturar el ciclo común.") if len(g) > 1 else "Muy pocos años con datos suficientes en la selección."
    num = d[NUMS].dropna(axis=1, how="all")
    c = num.corr(method="spearman")
    s = c[T].drop(T).dropna().sort_values(key=np.abs, ascending=False)
    i3 = (f"Con el objetivo, la correlación más fuerte es {LABELS.get(s.index[0], s.index[0])} (ρ = {s.iloc[0]:.2f}). "
          "Los costos en cabeza de pozo y el diferencial son componentes del objetivo y no se usan como predictores.") if len(s) else ""
    dd = d.dropna(subset=["WellheadProductionCost", "BrentDifferential"])
    fid = P.go.Figure(P.go.Scattergl(x=dd.WellheadProductionCost + dd.BrentDifferential, y=dd[T], mode="markers",
                                     marker=dict(size=5, color=P.BLUE, opacity=0.5),
                                     hovertemplate="suma %{x:.2f} · objetivo %{y:.2f}<extra></extra>"))
    mx = float(dd[T].max()) if len(dd) else 1
    fid.add_trace(P.go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=P.AMBER, dash="dash")))
    fid.update_layout(title="Objetivo vs suma de sus componentes", xaxis_title="Cabeza de pozo + diferencial",
                      yaxis_title="Objetivo (USD/Bbl)", height=330, showlegend=False)
    return (cnt, P.target_hist(y), i1, P.year_trend(d, T), i2, P.corr_heatmap(c, LABELS), i3, fid)


@callback(Output("g-box", "figure"), Output("i-box", "children"),
          Input("f-reg", "value"), Input("f-year", "value"), Input("f-rt", "value"), Input("f-loc", "value"),
          Input("box-cat", "value"))
def update_box(regs, years, rts, locs, cat):
    d = filtered(regs, years, rts, locs)
    if len(d) < 5:
        return P.empty("Sin datos"), ""
    med = d.groupby(cat)[T].median().sort_values()
    q1, q3 = d[T].quantile([.25, .75])
    out = ((d[T] < q1 - 1.5 * (q3 - q1)) | (d[T] > q3 + 1.5 * (q3 - q1))).mean() * 100
    txt = (f"La mediana va de {med.iloc[0]:.1f} ({med.index[0]}) a {med.iloc[-1]:.1f} USD/Bbl ({med.index[-1]}). "
           f"{out:.1f} % de las filas son atípicas por el criterio 1.5 × RIC: se conservan porque son activos reales "
           "(bitumen, syncrude), justo los de mayor riesgo.")
    return P.box_by(d, cat, T, LABELS[cat]), txt


@callback(Output("g-miss", "figure"), Output("i-miss", "children"),
          Input("f-reg", "value"), Input("f-year", "value"), Input("f-rt", "value"), Input("f-loc", "value"),
          Input("miss-mode", "value"))
def update_missing(regs, years, rts, locs, mode):
    d = filtered(regs, years, rts, locs)
    if len(d) < 5:
        return P.empty("Sin datos"), ""
    if mode == "var":
        cols = ["Production", "WellheadProductionCost", "BrentDifferential", "WellheadDevelopmentCost",
                "BrentEquivalentDevelopmentCost"] + CATS
        pct = d[cols].isna().mean() * 100
        pct.index = [LABELS.get(k, k) for k in pct.index]
        top = pct.idxmax()
        return P.missing_bar(pct, "% de faltantes por variable"), (
            f"La variable con más faltantes es {top} ({pct.max():.0f} %). Las categóricas están completas.")
    pct = d["Production"].isna().groupby(d["RegionName"]).mean() * 100
    full = pct[pct >= 99].index.tolist()
    txt = (f"Production falta casi por completo en: {', '.join(full)}. " if full else "") + \
        "El patrón depende de la región (MAR): se imputa con la mediana más un indicador de faltante dentro del pipeline."
    return P.missing_bar(pct, "% de filas sin Production por región"), txt


@callback(Output("g-ex", "figure"), Output("i-ex", "children"),
          Input("f-reg", "value"), Input("f-year", "value"), Input("f-rt", "value"), Input("f-loc", "value"),
          Input("ex-x", "value"), Input("ex-y", "value"), Input("ex-c", "value"))
def update_explorer(regs, years, rts, locs, x, y, color):
    d = filtered(regs, years, rts, locs).dropna(subset=[x, y])
    if len(d) < 5:
        return P.empty("Sin datos"), ""
    rho = d[[x, y]].corr(method="spearman").iloc[0, 1]
    strength = "fuerte" if abs(rho) >= 0.5 else ("moderada" if abs(rho) >= 0.3 else "débil")
    n_txt = f"{len(d):,}".replace(",", ".")
    txt = f"Correlación de Spearman ρ = {rho:.2f} ({strength}) entre {LABELS[x]} y {LABELS[y]} en {n_txt} filas."
    if color:
        med = d.groupby(color)[y].median().sort_values()
        txt += f" Por {LABELS[color]}, la mediana de {LABELS[y]} va de {med.iloc[0]:.1f} ({med.index[0]}) a {med.iloc[-1]:.1f} ({med.index[-1]})."
    return P.explorer(d, x, y, color, LABELS), txt
