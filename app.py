"""Punto de entrada del dashboard (Dash multipágina): crea la app y expone ``server``.

Local:   python app.py      ->  http://127.0.0.1:8050
Render:  gunicorn app:server   (ver render.yaml)
"""
from __future__ import annotations

import warnings

import dash
import dash_bootstrap_components as dbc
from dash import html

warnings.filterwarnings("ignore")

app = dash.Dash(
    __name__,
    use_pages=True,                                   # una página por sección (carpeta pages/)
    external_stylesheets=[dbc.themes.FLATLY],         # tema Bootstrap
    title="Costo unitario upstream · ML",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # servidor WSGI para gunicorn (Render)

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Contexto", href="/", active="exact")),
        dbc.NavItem(dbc.NavLink("EDA", href="/eda", active="exact")),
        dbc.NavItem(dbc.NavLink("Modelos", href="/modelos", active="exact")),
    ],
    brand="Costo unitario de producción upstream · ML",
    brand_href="/",
    color="primary",
    dark=True,
    fixed="top",
    className="navbar-main",
)

app.layout = html.Div([
    navbar,
    dbc.Container(dash.page_container, fluid=True, className="page"),
    html.Footer([
        html.Span("Leonella Torreglosa Portillo · Maestría en Ingeniería Industrial · Machine Learning"),
        html.Span(" · versión 3.0 · octubre 2026 · datos: S&P Global Commodity Insights (licencia académica)"),
    ], className="footer"),
])

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
