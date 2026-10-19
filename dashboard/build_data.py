"""Prepara los datos resumidos que consume el dashboard (results/dashboard/).

Ejecutar después de run_nested.py y de los notebooks:
    python dashboard/build_data.py
"""
import json
import pickle
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as C  # noqa: E402
from src.data import load_model_data, split_dev_test, xyg  # noqa: E402
from src.evaluation import acf  # noqa: E402
from src.models import MODEL_ORDER  # noqa: E402
from src.nested_cv import anytime_table, collect  # noqa: E402

OUT = C.RESULTS_DIR / "dashboard"
OUT.mkdir(exist_ok=True)

res = collect()
resA = [r for r in res if r["scenario"] == "A_estructural"]
mt = pd.read_csv(C.RESULTS_DIR / "master_table.csv")
mtA = mt[mt.scenario == "A_estructural"]

# 1) Curvas anytime (mediana entre pliegues) y diversidad del GA
at = anytime_table(resA)
at.groupby(["model", "method", "eval"])["best_so_far"].median().reset_index().to_csv(OUT / "anytime_median.csv", index=False)
gens = [{"model": r["model"], "fold": r["fold"], **g} for r in resA
        if r["method"] == "Genetica" and r["fold"] != "final" for g in r["extra"]["generations"]]
pd.DataFrame(gens).groupby(["model", "gen"])[["diversity", "best"]].mean().reset_index().to_csv(OUT / "ga_diversity.csv", index=False)
pd.DataFrame([{"model": r["model"], "method": r["method"], "fold": r["fold"], "search_time": r["search_time"],
               "best_inner_rmse": r["best_inner_rmse"], "outer_rmse": r["metrics"]["rmse"]}
              for r in resA if r["fold"] != "final"]).to_csv(OUT / "searches.csv", index=False)

# 2) Predicciones por subconjunto del mejor modelo de cada familia
df = load_model_data()
dev, test = split_dev_test(df)
X, y, g = xyg(dev)
Xt, yt, _ = xyg(test)
best = mtA.loc[mtA.groupby("model")["outer_rmse_mean"].idxmin()].set_index("model")
rows = []
acf_rows = []
for m in MODEL_ORDER:
    me = best.loc[m, "method"]
    pipe = pickle.load(open(C.MODELS_DIR / f"A_estructural__{m}__{me}.pkl", "rb"))
    oof = np.full(len(dev), np.nan)
    for r in resA:
        if r["model"] == m and r["method"] == me and r["fold"] != "final":
            oof[r["index"]] = r["pred"]
    fin = next(r for r in resA if r["model"] == m and r["method"] == me and r["fold"] == "final")
    for split, frame, yy, pp in [("Entrenamiento", dev, y, pipe.predict(X)), ("Validación", dev, y, oof),
                                 ("Prueba", test, yt, fin["pred"])]:
        rows.append(pd.DataFrame({"model": m, "method": me, "split": split, "y": yy, "pred": pp,
                                  "Year": frame["Year"].values, "RegionName": frame["RegionName"].values}))
    e = yt - fin["pred"]
    for k, v in enumerate(acf(e, 20)):
        acf_rows.append({"model": m, "lag": k, "acf": v, "band": 1.96 / np.sqrt(len(e))})
    shutil.copy(C.MODELS_DIR / f"A_estructural__{m}__{me}.pkl", OUT / f"model_{m}.pkl")
pd.concat(rows).to_csv(OUT / "predictions.csv", index=False)
pd.DataFrame(acf_rows).to_csv(OUT / "residual_acf.csv", index=False)

# 3) Opciones del predictor (jerarquía geográfica observada)
raw = pd.read_csv(C.DATA_PATH, sep=";")
hier = raw.drop_duplicates(["RegionName", "CountryName", "ProductionElementName", "MemberOpec"])[
    ["RegionName", "CountryName", "ProductionElementName", "MemberOpec"]]
combos = raw.drop_duplicates(["CountryName", "ReserveTypeName", "SupplySubTypeName", "ReserveLocationName"])[
    ["CountryName", "ReserveTypeName", "SupplySubTypeName", "ReserveLocationName"]]
json.dump({"hierarchy": hier.to_dict("records"), "combos": combos.to_dict("records"),
           "best_model": str(best["outer_rmse_mean"].idxmin()),
           "best_methods": best["method"].to_dict(),
           "production_median": float(df["Production"].median())},
          open(OUT / "predictor_options.json", "w"), indent=1)

print("Datos del dashboard escritos en", OUT)
