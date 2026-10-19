"""Etapa 1 del pipeline: validación cruzada anidada combinatoria.

Uso:  python run_nested.py
Salida: results/cache/*.pkl, results/master_table.csv, results/fold_rmse_*.csv,
        results/anytime.csv, results/models/*.pkl
"""
import warnings

warnings.filterwarnings("ignore")

from src import config as C  # noqa: E402
from src.data import load_model_data, split_dev_test  # noqa: E402
from src.nested_cv import anytime_table, collect, fold_matrix, master_table, run_experiment  # noqa: E402

if __name__ == "__main__":
    C.set_global_seed()
    df = load_model_data()
    dev, test = split_dev_test(df)

    # Escenario principal: 7 modelos x 4 optimizadores = 28 combinaciones
    run_experiment(dev, test, "A_estructural")
    # Escenario alternativo (sensibilidad a BrentDifferential): 7 modelos con Optuna
    run_experiment(dev, test, "B_con_BD", methods=["Bayesiana"])

    res = collect()
    mt = master_table(res)
    mt.to_csv(C.RESULTS_DIR / "master_table.csv", index=False)
    for sc in C.SCENARIOS:
        fold_matrix(res, sc, "rmse").to_csv(C.RESULTS_DIR / f"fold_rmse_{sc}.csv")
        fold_matrix(res, sc, "mae").to_csv(C.RESULTS_DIR / f"fold_mae_{sc}.csv")
    anytime_table(res).to_csv(C.RESULTS_DIR / "anytime.csv", index=False)
    print(mt[["scenario", "model", "method", "outer_rmse_mean", "outer_rmse_std", "test_rmse",
              "search_time_total_s"]].to_string())
