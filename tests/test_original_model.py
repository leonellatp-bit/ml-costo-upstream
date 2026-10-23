"""Pruebas unitarias del modelo original HGSB (guía, sección 5.4).

Ejecutar desde la raíz:  pytest -q tests/
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.original_model import HierarchicalShrinkageBoosting as HGSB  # noqa: E402

HIER = {"region": None, "country": "region"}


def synthetic(n_assets=120, years=5, seed=0, noise=0.3):
    """Datos con solución conocida: y = efecto región + efecto país + 0.5·(año − 2020) + ruido."""
    rng = np.random.default_rng(seed)
    regions = {f"R{i}": rng.normal(0, 3) for i in range(4)}
    countries = {f"C{j}": (f"R{j % 4}", rng.normal(0, 1)) for j in range(20)}
    rows = []
    for a in range(n_assets):
        c = f"C{rng.integers(20)}"
        r, ce = countries[c]
        for t in range(years):
            yr = 2018 + t
            rows.append({"region": r, "country": c, "Year": yr, "asset": f"A{a}",
                         "y": 10 + regions[r] + ce + 0.5 * (yr - 2020) + rng.normal(0, noise)})
    df = pd.DataFrame(rows)
    return df, regions, countries


def make(**kw):
    base = dict(hierarchy=HIER, numeric=["Year"], group_col="asset", n_estimators=150,
                learning_rate=0.1, max_depth=3, kappa=2.0, oof_folds=1, random_state=7)
    base.update(kw)
    return HGSB(**base)


# (i) aprende en un conjunto sintético con solución conocida ----------------------------
def test_learns_known_function():
    df, _, _ = synthetic(seed=1)
    new, _, _ = synthetic(seed=1)  # mismos efectos, ruido distinto -> generalización
    new["y"] = new["y"]  # objetivo con su propio ruido
    m = make().fit(df.drop(columns="y"), df["y"])
    pred = m.predict(new.drop(columns="y"))
    rmse = np.sqrt(np.mean((new["y"] - pred) ** 2))
    assert rmse < 0.6, rmse                      # cerca del ruido (0.3 + error de estimación)
    assert rmse < 0.25 * new["y"].std()          # muy por debajo de predecir la media


def test_unseen_category_falls_back_to_parent():
    df, regions, countries = synthetic(seed=2, noise=0.05)
    m = make(n_estimators=300).fit(df.drop(columns="y"), df["y"])
    # país nuevo en la región R1 -> debe heredar el nivel de la región R1
    probe = pd.DataFrame({"region": ["R1", "R2"], "country": ["NUEVO", "NUEVO"], "Year": [2020, 2020],
                          "asset": ["x", "y"]})
    p = m.predict(probe)
    assert np.sign(p[0] - p[1]) == np.sign(regions["R1"] - regions["R2"])


def test_shrinkage_limits():
    """κ = 0 -> media de la categoría; κ -> ∞ -> prior del padre (propiedad de la fórmula)."""
    df, _, _ = synthetic(seed=3)
    X = df.drop(columns="y")
    for kappa, check in [(0.0, "mean"), (1e9, "prior")]:
        m = make(kappa=kappa, n_estimators=1).fit(X, df["y"])
        codes = m._codes(X)
        _, asset = np.unique(X["asset"], return_inverse=True)
        m._structure(codes, asset, asset.max() + 1)
        r_asset = np.random.default_rng(0).normal(size=asset.max() + 1)
        t, root = m._tables(r_asset, np.ones(len(r_asset), bool))
        if check == "mean":
            ca = m._cat_asset["region"]
            for k in range(len(t["region"])):
                assert np.isclose(t["region"][k], r_asset[ca == k].mean())
        else:
            assert np.allclose(t["region"], root, atol=1e-6)
            assert np.allclose(t["country"], m._W["country"] @ t["region"], atol=1e-6)


# (ii) reproducibilidad con la misma semilla -------------------------------------------
def test_reproducible_same_seed():
    df, _, _ = synthetic(seed=4)
    X, y = df.drop(columns="y"), df["y"]
    p1 = make(subsample=0.7, colsample=0.8).fit(X, y).predict(X)
    p2 = make(subsample=0.7, colsample=0.8).fit(X, y).predict(X)
    p3 = make(subsample=0.7, colsample=0.8, random_state=99).fit(X, y).predict(X)
    assert np.array_equal(p1, p2)
    assert not np.array_equal(p1, p3)


# (iii) manejo de entradas inválidas -----------------------------------------------------
@pytest.mark.parametrize("kw", [dict(n_estimators=0), dict(learning_rate=0), dict(learning_rate=1.5),
                                dict(max_depth=0), dict(kappa=-1), dict(subsample=0),
                                dict(encoding="xyz"), dict(oof_folds=-2), dict(n_bins=1)])
def test_invalid_params(kw):
    df, _, _ = synthetic()
    with pytest.raises(ValueError):
        make(**kw).fit(df.drop(columns="y"), df["y"])


def test_invalid_inputs():
    df, _, _ = synthetic()
    X, y = df.drop(columns="y"), df["y"]
    with pytest.raises(TypeError):
        make().fit(X.to_numpy(), y)
    with pytest.raises(ValueError):
        make().fit(X.drop(columns="country"), y)
    with pytest.raises(ValueError):
        make().fit(X, y.iloc[:-1])
    y_nan = y.copy()
    y_nan.iloc[0] = np.nan
    with pytest.raises(ValueError):
        make().fit(X, y_nan)
    X_inf = X.copy()
    X_inf["Year"] = X_inf["Year"].astype(float)
    X_inf.loc[0, "Year"] = np.inf
    with pytest.raises(ValueError):
        make().fit(X_inf, y)
    with pytest.raises(NotFittedError):
        make().predict(X)
    m = make(n_estimators=5).fit(X, y)
    with pytest.raises(ValueError):
        m.predict(X.drop(columns="region"))


# (iv) compatibilidad con sklearn.base.clone y con la serialización ---------------------
def test_sklearn_clone():
    m = make(kappa=4.2)
    c = clone(m)
    assert c.get_params() == m.get_params()
    assert not hasattr(c, "trees_")
    df, _, _ = synthetic()
    m.fit(df.drop(columns="y"), df["y"])
    c2 = clone(m)
    assert not hasattr(c2, "trees_")


def test_joblib_round_trip(tmp_path):
    df, _, _ = synthetic()
    X, y = df.drop(columns="y"), df["y"]
    m = make().fit(X, y)
    joblib.dump(m, tmp_path / "m.joblib", compress=3)
    m2 = joblib.load(tmp_path / "m.joblib")
    assert np.array_equal(m.predict(X), m2.predict(X))


# El identificador del activo no es predictor ---------------------------------------------
def test_group_column_not_used_at_predict():
    df, _, _ = synthetic()
    X, y = df.drop(columns="y"), df["y"]
    m = make().fit(X, y)
    X2 = X.copy()
    X2["asset"] = "otro"
    assert np.array_equal(m.predict(X), m.predict(X2))
    assert np.array_equal(m.predict(X), m.predict(X.drop(columns="asset")))


def test_onehot_ablation_runs():
    df, _, _ = synthetic()
    m = make(encoding="onehot", n_estimators=50).fit(df.drop(columns="y"), df["y"])
    assert np.isfinite(m.predict(df.drop(columns="y"))).all()


# Propiedad teórica (proposición 3 del capítulo 5): con subsample = 1 la pérdida de entrenamiento no aumenta
def test_training_loss_non_increasing():
    df, _, _ = synthetic(seed=5)
    X, y = df.drop(columns="y"), df["y"].to_numpy()
    m = make(subsample=1.0, colsample=1.0, learning_rate=0.3, n_estimators=60).fit(X, y)
    losses = [np.mean((y - p) ** 2) for p in m.staged_predict(X)]
    assert all(b <= a + 1e-9 for a, b in zip(losses, losses[1:]))
    assert losses[-1] < 0.2 * np.var(y)
