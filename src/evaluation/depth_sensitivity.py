"""
Sensibilidade de max_depth por modelo, a partir do historico ja gravado
do Optuna (reports/optuna_study.db) -- sem re-treinar nada.

O professor questionou se faz sentido o Random Forest ter chegado a
max_depth=11 e o XGBoost a max_depth=6 no reports/optuna_best_params.json
(ele leu max_depth=12/8, que sao os defaults de fallback em train.py,
usados so se optuna_best_params.json nao existir -- ver nota no proprio
DEFAULT_PARAMS).

Este script agrega os 20 trials de cada estudo por valor de max_depth
para mostrar que o espaco foi de fato explorado (RF: 4-20, XGBoost: 3-10)
e que o ROC-AUC medio forma uma curva com pico, nao um platô nem uma
borda -- ou seja, os valores escolhidos nao sao arbitrarios.

    python -m src.evaluation.depth_sensitivity
"""

import json
import logging
from pathlib import Path

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")
STORAGE = f"sqlite:///{REPORTS_DIR / 'optuna_study.db'}"
STUDIES = {
    "random_forest": "random_forest_alfabetizacao",
    "xgboost": "xgboost_alfabetizacao",
}


def depth_curve(study_name: str) -> list[dict]:
    study = optuna.load_study(study_name=study_name, storage=STORAGE)
    trials = [t for t in study.trials if t.value is not None]
    depths = np.array([t.params["max_depth"] for t in trials])
    values = np.array([t.value for t in trials])

    curva = []
    for d in sorted(set(depths.tolist())):
        mask = depths == d
        curva.append({
            "max_depth": int(d),
            "n_trials": int(mask.sum()),
            "roc_auc_medio": float(values[mask].mean()),
        })
    return curva


def main() -> None:
    resumo = {}
    for nome, study_name in STUDIES.items():
        curva = depth_curve(study_name)
        resumo[nome] = curva
        pico = max(curva, key=lambda r: r["roc_auc_medio"])
        log.info("[%s] pico em max_depth=%d (ROC-AUC medio=%.4f)", nome, pico["max_depth"], pico["roc_auc_medio"])
        print(f"\n=== {nome} ===")
        print(f"{'max_depth':>10} {'n_trials':>10} {'roc_auc_medio':>15}")
        for r in curva:
            print(f"{r['max_depth']:>10} {r['n_trials']:>10} {r['roc_auc_medio']:>15.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "depth_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    log.info("Resumo salvo em reports/depth_sensitivity.json")


if __name__ == "__main__":
    main()
