import json
import logging
import time
from pathlib import Path

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from xgboost import XGBClassifier

import config
from src.preprocessing.features import build_full_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")
MODEL_NAMES = ["logistic", "random_forest", "xgboost"]


def _scale_pos_weight(y) -> float:
    """XGBoost nao tem class_weight -- scale_pos_weight = neg/pos e o equivalente."""
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return n_neg / n_pos if n_pos else 1.0


def _build_model(model_name, trial, y=None):
    if model_name == "logistic":
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 1e2, log=True),
            max_iter=1000,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            max_depth=trial.suggest_int("max_depth", 4, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            scale_pos_weight=_scale_pos_weight(y) if y is not None else 1.0,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="auc",
        )
    raise ValueError(f"Modelo nao reconhecido: {model_name}")


def _cv_score(model, X, y, groups, n_splits, trial):
    """
    CV manual (em vez de cross_val_score) para poder reportar score
    parcial a cada fold e permitir pruning de trials ruins no meio da
    validacao -- cross_val_score so devolve o resultado no final.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups=groups)):
        pipe = build_full_pipeline(model, apply_undersampling=False)
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        proba = pipe.predict_proba(X.iloc[va_idx])[:, 1]
        scores.append(roc_auc_score(y.iloc[va_idx], proba))

        trial.report(float(np.mean(scores)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def make_objective(model_name, X, y, groups, n_splits):
    def objective(trial):
        model = _build_model(model_name, trial, y=y)
        return _cv_score(model, X, y, groups, n_splits, trial)

    return objective


def run_study(model_name, X, y, groups, n_trials=None, n_splits=3):
    n_trials = n_trials or config.OPTUNA_N_TRIALS
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=f"{model_name}_alfabetizacao",
        direction="maximize",
        sampler=TPESampler(seed=config.RANDOM_STATE),
        pruner=MedianPruner(n_warmup_steps=1),
        storage=config.OPTUNA_STORAGE,
        load_if_exists=True,
    )

    start = time.time()
    study.optimize(
        make_objective(model_name, X, y, groups, n_splits),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    elapsed = time.time() - start

    log.info(
        "[%s] %d trials em %.1fs | melhor ROC-AUC=%.4f | params=%s",
        model_name, n_trials, elapsed, study.best_value, study.best_params,
    )
    return study


def subsample_for_search(X, y, groups):
    """
    Otimizacao Optuna roda em subamostra (config.OPTUNA_SAMPLE_SIZE) em
    vez da base de treino inteira (~2,6M linhas): cada trial precisa
    treinar N modelos (um por fold), e em 3,3M linhas isso inviabiliza
    dezenas de trials em tempo de desenvolvimento local. O split final
    (treino/teste) e o treino do modelo campeao usam a base completa --
    so a BUSCA de hiperparametros e feita na subamostra. Decisao de
    FinOps/tempo documentada, no mesmo espirito das decisoes de
    particionamento da Fase 2.
    """
    frac = min(1.0, config.OPTUNA_SAMPLE_SIZE / len(X))
    if frac >= 1.0:
        return X, y, groups

    splitter = GroupShuffleSplit(n_splits=1, train_size=frac, random_state=config.RANDOM_STATE)
    idx, _ = next(splitter.split(X, y, groups=groups))
    log.info(
        "Subamostra para busca de hiperparametros: %s de %s linhas (%.1f%%)",
        f"{len(idx):,}", f"{len(X):,}", frac * 100,
    )
    return X.iloc[idx], y.iloc[idx], groups.iloc[idx]


def main(models=None) -> None:
    """
    models=None roda os 3 (pode estourar 10min em maquina local); passar
    uma lista com 1 nome via --model permite rodar cada modelo numa
    chamada separada e depois consolidar com merge_partial_results().
    """
    import google.auth
    from google.cloud import bigquery

    from src.preprocessing.features import prepare_data

    models = models or MODEL_NAMES

    credentials, _ = google.auth.default()
    client = bigquery.Client(project=config.GCP_PROJECT_ID, credentials=credentials)

    X_train, X_test, y_train, y_test, groups_train, groups_test = prepare_data(client)
    X_search, y_search, groups_search = subsample_for_search(X_train, y_train, groups_train)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for model_name in models:
        study = run_study(model_name, X_search, y_search, groups_search, n_splits=config.OPTUNA_CV_SPLITS_SEARCH)
        partial_path = REPORTS_DIR / f"optuna_partial_{model_name}.json"
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(
                {"best_roc_auc_cv": study.best_value, "best_params": study.best_params},
                f, indent=2, ensure_ascii=False,
            )
        log.info("Resultado parcial salvo em %s", partial_path)

    merge_partial_results()


def merge_partial_results() -> dict:
    """Junta reports/optuna_partial_<modelo>.json em optuna_best_params.json."""
    best_params = {}
    for model_name in MODEL_NAMES:
        partial_path = REPORTS_DIR / f"optuna_partial_{model_name}.json"
        if partial_path.exists():
            with open(partial_path, encoding="utf-8") as f:
                best_params[model_name] = json.load(f)

    out_path = REPORTS_DIR / "optuna_best_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    log.info("Consolidado (%d/%d modelos) em %s", len(best_params), len(MODEL_NAMES), out_path)
    return best_params


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, default=None)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    if args.merge_only:
        merge_partial_results()
    else:
        main(models=[args.model] if args.model else None)
