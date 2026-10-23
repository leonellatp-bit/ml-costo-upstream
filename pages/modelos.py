"""Sección 3: Modelos. Tabla de líderes, predicción en vivo con intervalo y SHAP local,
diagnóstico, interpretabilidad global, comparación estadística y el modelo original."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dash_table, dcc, html, no_update

from src import plots as P
from src.dash_data import LABELS, available_models, jfile, metadata, model, table
from src.explain import conformal_interval, grouped_shapley

dash.register_page(__name__, path="/modelos", name="Modelos", title="Modelos · Costo upstream", order=2)
GCFG = {"displayModeBar": False, "responsive": True}

PSEUDO = """Entrada: datos {(xᵢ, yᵢ, aᵢ)} (aᵢ = activo), jerarquía padre(·), M, η, d, λ, γ, κ
F₀(x) ← media(y)
para m = 1 … M:
    rᵢ ← yᵢ − F_{m−1}(xᵢ)                          # gradiente negativo (pérdida cuadrática)
    r̄ₐ ← media de rᵢ en el activo a                 # se cuenta por activo, no por fila
    para cada categórica c, de la raíz a las hojas de la jerarquía:
        S_k ← Σ_{a∈k} r̄ₐ ;  n_k ← #activos en k
        π_k ← Σ_j W_kj · ẽ_padre,j   (raíz: media global)
        ẽ_k ← (S_k + κ·π_k) / (n_k + κ)              # contracción empírica Bayes
    zᵢ ← [numéricas de xᵢ, ẽ_c(xᵢ) para cada c]     # categoría no vista → hereda ẽ del padre
    h_m ← árbol por histogramas sobre z con ganancia
          ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ  y hojas w* = ΣR/(H+λ)
    F_m ← F_{m−1} + η·h_m
Salida: F_M y, por iteración, las tablas ẽ (para predecir filas nuevas)"""

RELATED = [
    ("Codificación jerárquica del objetivo (Micci-Barreca, 2001)", "Media del objetivo por categoría contraída hacia el nivel padre",
     "Se aplica una vez como preprocesamiento; HGSB recodifica el residuo en cada iteración y cuenta activos"),
    ("XGBoost (Chen y Guestrin, 2016)", "Boosting de 2.º orden, hojas −G/(H+λ)", "Categóricas por One-Hot; no usa jerarquía"),
    ("LightGBM (Ke et al., 2017)", "Ordena categorías por G/(H + cat_smooth)", "Contrae hacia 0 y cuenta filas; no hereda del padre"),
    ("CatBoost (Prokhorenkova et al., 2018)", "Estadísticos ordenados del objetivo", "Codifica el objetivo, no el residuo de cada iteración; sin jerarquía"),
    ("GPBoost / MERF (Sigrist, 2022; Hajjem et al., 2014)", "Boosting/árboles + efectos aleatorios", "Efectos aditivos por grupo, no dentro de los cortes del árbol"),
    ("Hierarchical Shrinkage (Agarwal et al., 2022)", "Contrae hojas hacia nodos ancestros del árbol", "Jerarquía del árbol, no de las categorías del dominio"),
]


def leaderboard():
    lb = table("leaderboard")
    if not len(lb):
        return dbc.Alert("Ejecute tools/build_artifacts.py para generar la tabla de líderes.", color="warning")
    d = pd.DataFrame({
        "Modelo": lb.label, "Optimizador": lb.method,
        "RMSE externo (media ± d.e.)": [f"{a:.3f} ± {b:.3f}" for a, b in zip(lb.outer_rmse_mean, lb.outer_rmse_std)],
        "MAE externo": [f"{a:.3f} ± {b:.3f}" for a, b in zip(lb.outer_mae_mean, lb.outer_mae_std)],
        "R² externo": [f"{a:.3f} ± {b:.3f}" for a, b in zip(lb.outer_r2_mean, lb.outer_r2_std)],
        "RMSE prueba": lb.test_rmse.round(3), "Cobertura IC 90 % (prueba)": (100 * lb.coverage_test).round(1),
        "Inferencia (ms/fila)": lb.infer_ms_per_row.round(3), "Tamaño (MB)": lb.size_mb.round(1),
        "En el MCS": lb.in_mcs.map({True: "Sí", False: "No"}),
    }).sort_values("RMSE prueba")
    return dash_table.DataTable(
        d.to_dict("records"), [{"name": c, "id": c} for c in d.columns], sort_action="native", filter_action="native",
        page_size=10, style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "Inter, system-ui, sans-serif", "fontSize": 13, "padding": "7px 10px",
                    "border": "none", "borderBottom": f"1px solid {P.GRID}", "textAlign": "left", "whiteSpace": "normal"},
        style_header={"fontWeight": 600, "backgroundColor": "#F4F6F8", "color": P.INK},
        style_data_conditional=[{"if": {"filter_query": '{Modelo} contains "HGSB"'}, "backgroundColor": "#FDF3E6",
                                 "fontWeight": 600}],
        style_filter={"backgroundColor": "#FBFBFC"})


def predictor_form():
    o = jfile("options")
    if not o:
        return dbc.Alert("Faltan las opciones del predictor (tools/build_artifacts.py).", color="warning")
    hier = pd.DataFrame(o["hierarchy"])
    countries = sorted(hier.CountryName.unique())
    sch = o.get("schema", {})
    yr = sch.get("Year", {"min": 2014, "max": 2023})
    pr = sch.get("Production", {"min": 0, "max": 5000})
    return html.Div([
        dbc.Row([
            dbc.Col([dbc.Label("País"), dcc.Dropdown(countries, "Colombia" if "Colombia" in countries else countries[0],
                                                     id="p-country", clearable=False)], md=6),
            dbc.Col([dbc.Label("Elemento productivo / cuenca"), dcc.Dropdown(id="p-pe", clearable=False)], md=6),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([dbc.Label("Tipo de reserva"), dcc.Dropdown(["Conventional", "Unconventional"], "Conventional",
                                                                 id="p-rt", clearable=False)], md=4),
            dbc.Col([dbc.Label("Subtipo"), dcc.Dropdown(id="p-st", clearable=False)], md=4),
            dbc.Col([dbc.Label("Ubicación"), dcc.Dropdown(["Onshore", "Offshore Shallow", "Offshore Deep"], "Onshore",
                                                          id="p-loc", clearable=False)], md=4),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([dbc.Label(f"Año ({yr['min']}–{yr['max']} observado)"),
                     dcc.Slider(yr["min"], yr["max"] + 3, 1, value=yr["max"], id="p-year",
                                marks={y: str(y) for y in range(yr["min"], yr["max"] + 4, 3)})], md=7),
            dbc.Col([dbc.Label("Producción (MBD; vacío = desconocida)"),
                     dbc.Input(id="p-prod", type="number", min=0, max=pr["max"] * 2, step="any",
                               placeholder=f"rango observado 0–{pr['max']:.0f}")], md=5),
        ], className="mb-3"),
        html.Div(id="p-ctx", className="small mb-2"),
        dbc.Button("Predecir", id="p-go", color="primary", n_clicks=0, className="me-2"),
        html.Div(id="p-alert", className="mt-2"),
    ])


def original_section():
    v = jfile("verdict")
    tests = table("tests")
    abl = table("ablation")
    conv = table("convergence")
    badge = dbc.Badge(v.get("label", "pendiente"), color={"superior": "success", "no superior": "warning"}.get(
        v.get("status"), "secondary"), className="verdict-badge")
    tcols = ["Prueba", "H0", "Supuestos", "Estadístico", "p crudo", "p ajustado (Holm)", "Tamaño del efecto", "Conclusión"]
    ttab = (dbc.Table.from_dataframe(tests[[c for c in tcols if c in tests.columns]], striped=False, hover=True,
                                     responsive=True, size="sm", className="mb-0") if len(tests) else
            dbc.Alert("Ejecute run_original_stats.py.", color="warning"))
    atab = (dbc.Table.from_dataframe(abl, hover=True, responsive=True, size="sm", className="mb-0") if len(abl) else None)
    return dbc.Card([dbc.CardHeader(["Modelo original: HGSB · Hierarchical Geo-Shrinkage Gradient Boosting  ", badge]),
                     dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.H5("Idea"),
                html.P(["En cada iteración de boosting, cada categoría geográfica se reemplaza por la media del residuo de "
                        "sus activos, ", html.B("contraída hacia la de su categoría padre"),
                        " (país → región, elemento productivo → región, subtipo → tipo): ẽ = (S + κ·π_padre)/(n + κ). "
                        "Así, los 45 países con un solo activo toman información de su región en vez de quedar en la "
                        "categoría “infrecuente” del One-Hot, y un país nunca visto hereda el valor de su región."]),
                html.P(["Sustento: la fórmula es la media posterior de un modelo normal-normal (empirical Bayes / James-Stein) "
                        "y ordenar categorías por residuo medio da la partición binaria óptima para pérdida cuadrática "
                        "(Fisher, 1958). Todo el núcleo está programado en numpy (", html.Code("src/original_model.py"), ")."]),
                html.H5("Diferencias con trabajos cercanos", className="mt-3"),
                dbc.Table([html.Thead(html.Tr([html.Th("Método"), html.Th("Qué hace"), html.Th("Qué cambia el HGSB")])),
                           html.Tbody([html.Tr([html.Td(a), html.Td(b), html.Td(c)]) for a, b, c in RELATED])],
                          size="sm", responsive=True, hover=True),
            ], lg=6),
            dbc.Col([html.H5("Pseudocódigo"), html.Pre(PSEUDO, className="pseudo")], lg=6),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=P.convergence(conv) if len(conv) else P.empty("Sin curva"), config=GCFG), lg=6),
            dbc.Col([html.H5("Estudio de ablación (RMSE externo, 10 pliegues)"), atab,
                     html.P("Cada fila retira un componente manteniendo los hiperparámetros de cada pliegue.",
                            className="small mt-1")], lg=6),
        ], className="mt-2"),
        html.H5("Pruebas de hipótesis frente al mejor benchmark", className="mt-3"),
        ttab,
        dbc.Alert([html.B("Veredicto: "), v.get("text", "pendiente")],
                  color="success" if v.get("status") == "superior" else "warning", className="mt-3 mb-0"),
    ])], className="mb-3")


def layout():
    models = available_models()
    lb = table("leaderboard")
    default = lb.sort_values("test_rmse").model.iloc[0] if len(lb) else (models[0] if models else None)
    mcs = table("mcs")
    return html.Div([
        html.H2("Modelos", className="page-title"),
        html.P("Mejor combinación de cada modelo (por RMSE medio del bucle externo) más el modelo original. Los modelos se "
               "cargan desde artefactos ya entrenados: el dashboard no entrena ni optimiza.", className="lead-sm"),
        dbc.Card([dbc.CardHeader("Tabla de líderes"), dbc.CardBody([leaderboard(), html.P(
            "Ordena con las flechas de cada columna y filtra escribiendo en la fila superior (p. ej. “Sí” en la columna MCS).",
            className="small mt-2 mb-0")])], className="mb-3"),
        dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col([dbc.Label("Modelo seleccionado (controla predicción, diagnóstico e interpretabilidad)"),
                     dcc.Dropdown([{"label": LABELS[m], "value": m} for m in models], default, id="m-sel",
                                  clearable=False)], md=6),
            dbc.Col(html.Div(id="m-meta", className="small"), md=6),
        ])), className="mb-3 sticky-select"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Predicción en vivo"), dbc.CardBody(predictor_form())], className="h-100"), lg=5),
            dbc.Col(dbc.Card([dbc.CardHeader("Resultado"), dbc.CardBody(dcc.Loading([
                html.Div(id="p-out", className="pred-out"), html.Div(id="p-int", className="pred-int"),
                dcc.Graph(id="p-shap", config=GCFG), dcc.Graph(id="p-all", config=GCFG)], type="dot", color=P.BLUE))],
                className="h-100"), lg=7),
        ], className="g-3 mb-3"),
        dbc.Card([dbc.CardHeader("Diagnóstico del modelo seleccionado"), dbc.CardBody([
            dbc.Row([dbc.Col(dcc.Graph(id="d-rvp", config=GCFG), lg=6), dbc.Col(dcc.Graph(id="d-res", config=GCFG), lg=6)]),
            dbc.Row([dbc.Col(dcc.Graph(id="d-hist", config=GCFG), lg=6), dbc.Col(dcc.Graph(id="d-acf", config=GCFG), lg=6)]),
            html.P(id="d-ins", className="insight")])], className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Interpretabilidad global (SHAP por grupos)"), dbc.CardBody([
                dcc.Graph(id="g-shap", config=GCFG), html.P(id="g-shap-ins", className="insight")])], className="h-100"), lg=5),
            dbc.Col(dbc.Card([dbc.CardHeader("Comparación estadística: Model Confidence Set con el modelo original"),
                              dbc.CardBody([dcc.Graph(figure=P.mcs_bar(mcs) if len(mcs) else P.empty("Sin MCS"), config=GCFG),
                                            html.P("Ámbar = HGSB. Los modelos con p_MCS ≥ 0.10 forman el conjunto de los "
                                                   "mejores: no se distinguen estadísticamente del mejor.", className="insight")])],
                             className="h-100"), lg=7),
        ], className="g-3 mb-3"),
        original_section(),
    ])


# ----------------------------------------------------------------- callbacks
@callback(Output("m-meta", "children"), Input("m-sel", "value"))
def show_meta(m):
    md = metadata(m) if m else {}
    if not md:
        return ""
    hp = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in md.get("hyperparameters", {}).items())
    return [html.B(f"{md.get('method', '')} · "), f"hiperparámetros: {hp}"]


@callback(Output("p-pe", "options"), Output("p-pe", "value"), Output("p-ctx", "children"), Input("p-country", "value"))
def pe_options(country):
    hier = pd.DataFrame(jfile("options")["hierarchy"])
    h = hier[hier.CountryName == country]
    pes = sorted(h.ProductionElementName.unique())
    n = jfile("options").get("assets_per_country", {}).get(country, 0)
    warn = " · pocos activos: mayor incertidumbre" if n <= 1 else ""
    return pes, pes[0], f"Región: {h.RegionName.iloc[0]} · {h.MemberOpec.iloc[0]} · {n} activo(s) en los datos{warn}"


@callback(Output("p-st", "options"), Output("p-st", "value"), Input("p-rt", "value"))
def st_options(rt):
    m = {"Conventional": ["Non-Shale", "Shale", "Kerogen/Shale"], "Unconventional": ["Bitumen", "Syncrude"]}
    o = m.get(rt, ["Non-Shale"])
    return o, o[0]


def _row(country, pe, rt, st, loc, year, prod):
    hier = pd.DataFrame(jfile("options")["hierarchy"])
    h = hier[hier.CountryName == country].iloc[0]
    return pd.DataFrame([{"Year": year, "Production": np.nan if prod in (None, "") else float(prod),
                          "RegionName": h.RegionName, "CountryName": country, "ReserveTypeName": rt,
                          "SupplySubTypeName": st, "ReserveLocationName": loc,
                          "ProductionElementName": pe or h.ProductionElementName, "MemberOpec": h.MemberOpec}])


@callback(Output("p-out", "children"), Output("p-int", "children"), Output("p-shap", "figure"), Output("p-all", "figure"),
          Output("p-alert", "children"),
          Input("p-go", "n_clicks"), Input("m-sel", "value"),
          State("p-country", "value"), State("p-pe", "value"), State("p-rt", "value"), State("p-st", "value"),
          State("p-loc", "value"), State("p-year", "value"), State("p-prod", "value"))
def predict(_, m, country, pe, rt, st, loc, year, prod):
    if not m or model(m) is None:
        return no_update, no_update, no_update, no_update, dbc.Alert("Modelo no disponible.", color="danger")
    alerts = []
    try:
        if prod not in (None, "") and (not np.isfinite(float(prod)) or float(prod) < 0):
            return no_update, no_update, no_update, no_update, dbc.Alert(
                "La producción debe ser un número ≥ 0 (o dejarse vacía).", color="danger")
        row = _row(country, pe, rt, st, loc, int(year), prod)
        if year > jfile("options").get("schema", {}).get("Year", {}).get("max", 2023):
            alerts.append(dbc.Alert("Año fuera del periodo observado: la predicción es una extrapolación.", color="warning",
                                    className="py-2 mb-1"))
        est = model(m)
        yhat = float(est.predict(row)[0])
        q = jfile("conformal").get(m, {}).get("q", np.nan)
        lo, hi = conformal_interval([yhat], q)
        bg = table("background")
        phi, base = grouped_shapley(est.predict, row, bg)
        all_preds = pd.Series({k: float(model(k).predict(row)[0]) for k in available_models()})
        cov = jfile("conformal").get(m, {}).get("coverage_test", np.nan)
        return (f"{yhat:.2f} USD/Bbl",
                f"Intervalo de predicción 90 % (conformal): {lo[0]:.2f} – {hi[0]:.2f} USD/Bbl · cobertura en prueba "
                f"{100 * cov:.0f} %",
                P.shap_waterfall(phi.iloc[0], base, yhat), P.models_bar(all_preds, m, LABELS), alerts)
    except Exception as exc:  # entrada inválida: mensaje claro sin detener la app
        return no_update, no_update, no_update, no_update, dbc.Alert(f"No se pudo predecir: {exc}", color="danger")


@callback(Output("d-rvp", "figure"), Output("d-res", "figure"), Output("d-hist", "figure"), Output("d-acf", "figure"),
          Output("d-ins", "children"), Output("g-shap", "figure"), Output("g-shap-ins", "children"), Input("m-sel", "value"))
def diagnostics(m):
    p = table("predictions")
    p = p[p.model == m]
    if not len(p):
        e = P.empty("Sin predicciones")
        return e, e, e, e, "", e, ""
    te = p[p.split == "Prueba"]
    a = table("acf")
    a = a[a.model == m]
    rm = {s: float(np.sqrt(np.mean((g.y - g.pred) ** 2))) for s, g in p.groupby("split")}
    r1 = a[a.lag == 1].acf.iloc[0] if len(a[a.lag == 1]) else np.nan
    ins = (f"RMSE entrenamiento {rm.get('Entrenamiento', np.nan):.2f} · validación (fuera de pliegue) "
           f"{rm.get('Validación', np.nan):.2f} · prueba {rm.get('Prueba', np.nan):.2f} USD/Bbl. "
           f"El residuo crece con el nivel de costo (heterocedasticidad) y persiste dentro de cada activo "
           f"(ACF rezago 1 = {r1:.2f}): el error dominante es un efecto propio del activo.")
    sg = table("shap_global")
    s = sg[sg.model == m].set_index("player")["mean_abs"] if len(sg) else pd.Series(dtype=float)
    gi = (f"Para {LABELS[m]}, {s.idxmax()} es el factor dominante ({s.max():.2f} USD/Bbl de efecto medio)."
          if len(s) else "")
    return (P.real_vs_pred(p), P.resid_vs_pred(te), P.resid_hist(te), P.acf_bar(a), ins,
            P.shap_global_bar(s, "Importancia global (media |SHAP|)") if len(s) else P.empty("Sin SHAP"), gi)
