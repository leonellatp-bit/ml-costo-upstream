"""Sección 1: Contexto (narrativa estática y resumen visual)."""
import dash
import dash_bootstrap_components as dbc
from dash import html

from src import config as C
from src.dash_data import data, jfile, raw, table

dash.register_page(__name__, path="/", name="Contexto", title="Contexto · Costo upstream", order=0)

DICT = [
    ("BrentEquivalentProductionCost", "numérica (USD/Bbl)", "Objetivo",
     "Costo de producir un barril en cabeza de pozo más el diferencial frente al Brent."),
    ("Year", "numérica (año)", "Predictora", "Año de la observación (2014–2023)."),
    ("Production", "numérica (MBD)", "Predictora", "Producción del activo; log1p. Falta en 3 regiones (MAR por región)."),
    ("RegionName", "categórica (14)", "Predictora", "Región geográfica."),
    ("CountryName", "categórica (99)", "Predictora", "País; 45 países tienen un solo activo."),
    ("ProductionElementName", "categórica", "Predictora", "Elemento productivo / cuenca de referencia."),
    ("ReserveTypeName", "categórica (2)", "Predictora", "Convencional o no convencional."),
    ("SupplySubTypeName", "categórica (5)", "Predictora", "Non-Shale, Shale, Kerogen/Shale, Bitumen, Syncrude."),
    ("ReserveLocationName", "categórica (3)", "Predictora", "Onshore, offshore somero u offshore profundo."),
    ("MemberOpec", "categórica (2)", "Predictora", "País miembro o no de la OPEP."),
    ("WellheadProductionCost, BrentDifferential", "numéricas", "Excluidas",
     "Sumandos exactos del objetivo (identidad contable): fuga de información."),
    ("CostProductionElementName", "identificador", "Agrupación", "Activo: define las particiones por grupo; nunca es predictora."),
]

STEPS = [("ETL", "S&P Global · 4.595 filas · auditoría de fuga"),
         ("EDA", "faltantes MAR, efectos geográficos, deriva temporal"),
         ("Benchmarks", "7 modelos × 4 optimizadores (Grid, Random, Bayes, GA)"),
         ("CV anidada", "10 pliegues externos × 3 internos, por activo"),
         ("Estadística", "Friedman → Nemenyi → MCS → GW/CW → DM"),
         ("Modelo original", "HGSB: boosting con contracción jerárquica"),
         ("Producción", "artefactos + dashboard en Render")]


def kpi(label, value, note=""):
    return dbc.Col(dbc.Card(dbc.CardBody([html.Div(label, className="kpi-label"), html.Div(value, className="kpi-value"),
                                          html.Div(note, className="kpi-note")])), xs=6, lg=3, className="mb-3")


def layout():
    r, d = raw(), data()
    lb = table("leaderboard")
    verdict = jfile("verdict")
    best_txt = ""
    if len(lb):
        b = lb.sort_values("test_rmse").iloc[0]
        best_txt = f"{b['label']}: RMSE {b['test_rmse']:.2f} USD/Bbl en 139 activos nunca vistos (R² {b['test_r2']:.2f})"
    return html.Div([
        html.Div([
            html.Img(src=dash.get_asset_url("hero_oilfield.svg"),
                     alt="Ilustración de un campo petrolero con balancines, tanques y una plataforma offshore",
                     className="hero-img"),
            html.Div([
                html.Div("Contexto del problema", className="hero-kicker"),
                html.H1("¿Cuánto cuesta producir un barril en un activo que aún no conocemos?"),
                html.P("Estimamos el costo unitario de producción (USD por barril, equivalente Brent) de activos upstream "
                       "a partir de su geografía, tipo de reserva, ubicación, año y producción."),
                html.Div(best_txt, className="hero-figure") if best_txt else None,
            ], className="hero-text"),
        ], className="hero"),

        dbc.Row([kpi("Observaciones", f"{len(r):,}".replace(",", "."), f"{len(d):,} con costo".replace(",", ".")),
                 kpi("Activos", f"{r[C.GROUP].nunique()}", f"{d[C.GROUP].nunique()} con costo observado"),
                 kpi("Países · regiones", f"{r.CountryName.nunique()} · {r.RegionName.nunique()}", "cobertura global"),
                 kpi("Periodo", f"{r.Year.min()}–{r.Year.max()}", "panel activo-año")]),

        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Por qué importa"), dbc.CardBody([
                html.P(["El costo por barril define qué activos son rentables a un precio dado del Brent, cuáles se "
                        "desarrollan y cómo se asigna el OPEX en un portafolio. La presión sobre ese costo es real: la ",
                        html.B("inversión upstream en petróleo caería 6 % en 2025"),
                        ", la primera caída anual desde 2020, impulsada por el recorte en petróleo de esquisto de EE. UU. ",
                        html.A("(IEA, World Energy Investment 2025)",
                               href="https://www.iea.org/news/global-energy-investment-set-to-rise-to-33-trillion-in-2025-amid-economic-uncertainty-and-energy-security-concerns",
                               target="_blank"), "."]),
                html.Ul([
                    html.Li([html.B("Inversión: "), "al evaluar adquisiciones o nuevos desarrollos casi nunca hay historia "
                             "de costos del activo; un estimado por características reduce la incertidumbre."]),
                    html.Li([html.B("OPEX: "), "un modelo de referencia detecta activos que cuestan más de lo esperado para su perfil."]),
                    html.Li([html.B("Heterogeneidad: "), "en los datos el costo va de ~3 a ~54 USD/Bbl según geografía y tipo de reserva."]),
                ]),
            ])]), lg=6, className="mb-3"),
            dbc.Col(dbc.Card([dbc.CardHeader("Pregunta de investigación y objetivos"), dbc.CardBody([
                html.P(html.I("¿Es posible predecir el costo unitario de producción (USD/Bbl, equivalente Brent) de un activo "
                              "upstream que el modelo nunca ha visto, a partir de sus características estructurales?")),
                html.P(html.B("Objetivo general: "), className="mb-1"),
                html.P("Construir, validar y desplegar un modelo que generalice a activos nuevos."),
                html.Ul([
                    html.Li("Comparar 7 modelos × 4 optimizadores con validación cruzada anidada por activo."),
                    html.Li("Proponer un modelo original (HGSB) y verificar estadísticamente si supera a los benchmarks."),
                    html.Li("Interpretar qué variables mueven el costo (SHAP) y ofrecer predicción en vivo con intervalo."),
                ]),
                html.P([html.B("Tarea: "), "regresión supervisada sobre un panel activo-año."], className="mb-0"),
            ])]), lg=6, className="mb-3"),
        ]),

        dbc.Card([dbc.CardHeader("Metodología"), dbc.CardBody(
            html.Div([html.Div([html.Div(s, className="step-title"), html.Div(t, className="step-text")], className="step")
                      for s, t in STEPS], className="pipeline"))], className="mb-3"),

        dbc.Card([dbc.CardHeader("Datos y diccionario de variables"), dbc.CardBody([
            html.P("Fuente: S&P Global Commodity Insights, módulo Supply Cost and Ownership (licencia corporativa, uso "
                   f"académico). {f'{len(r):,}'.replace(',', '.')} registros activo-año, 2014–2023, 17 columnas."),
            dbc.Table([html.Thead(html.Tr([html.Th("Variable"), html.Th("Tipo"), html.Th("Rol"), html.Th("Descripción")])),
                       html.Tbody([html.Tr([html.Td(html.Code(v)), html.Td(t), html.Td(rl), html.Td(ds)]) for v, t, rl, ds in DICT])],
                      bordered=False, hover=True, responsive=True, size="sm", className="mb-0"),
        ])], className="mb-3"),

        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Limitaciones, ética y sesgo"), dbc.CardBody(html.Ul([
                html.Li("695 activos independientes; 96 % convencionales: más incertidumbre en no convencionales extremos."),
                html.Li("45 de 99 países con un solo activo y Production ausente en África y Asia Central: posible sesgo geográfico."),
                html.Li("El modelo estima un costo de referencia; no reemplaza la ingeniería de costos de un activo real."),
                html.Li("Datos bajo licencia: el repositorio no publica datos crudos ni nombres de activos con su costo."),
                html.Li("Años posteriores a 2023 son extrapolación."),
            ], className="mb-0"))]), lg=6, className="mb-3"),
            dbc.Col(dbc.Card([dbc.CardHeader("Autoría y versión"), dbc.CardBody([
                html.P([html.B("Autora: "), "Leonella Torreglosa Portillo · Maestría en Ingeniería Industrial"]),
                html.P([html.B("Versión: "), "3.0 (Entregable 3) · octubre de 2026"]),
                html.P([html.B("Código e informe: "),
                        html.A("GitHub", href="https://github.com/leonellatp-bit/ml-costo-upstream", target="_blank"), " · ",
                        html.A("Jupyter Book", href="https://leonellatp-bit.github.io/ml-costo-upstream", target="_blank")]),
                html.P(verdict.get("summary", ""), className="small mb-0"),
            ])]), lg=6, className="mb-3"),
        ]),
    ])
