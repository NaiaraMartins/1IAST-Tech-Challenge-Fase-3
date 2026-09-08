import json
import logging
from pathlib import Path

import google.auth
import joblib
import pandas as pd
from google.cloud import bigquery
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from xgboost import XGBClassifier

import config
from src.preprocessing.features import build_full_pipeline, prepare_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
BEST_PARAMS_PATH = REPORTS_DIR / "optuna_best_params.json"

# FALLBACK apenas -- usados so se reports/optuna_best_params.json nao existir
# (ver load_best_params() abaixo). Os hiperparametros de fato usados no treino
# final e reportados no README (secao 8) sao os tunados pelo Optuna: RF
# max_depth=11, XGBoost max_depth=6. Ver src/evaluation/depth_sensitivity.py
# para a analise de por que essas profundidades (e nao estas aqui) fazem sentido.
DEFAULT_PARAMS = {
    "logistic": {"C": 1.0},
    "random_forest": {
        "n_estimators": 200, "max_depth": 12, "min_samples_leaf": 2, "max_features": "sqrt",
    },
    "xgboost": {
        "n_estimators": 200, "max_depth": 8, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0,
    },
}


def load_best_params() -> dict:
    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        log.info("Usando hiperparametros otimizados pelo Optuna (%s).", BEST_PARAMS_PATH)
        return data
    log.warning(
        "%s nao encontrado -- rode 'python -m src.modeling.tune' antes para otimizar. "
        "Usando hiperparametros default.", BEST_PARAMS_PATH,
    )
    return {}


def _scale_pos_weight(y) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return n_neg / n_pos if n_pos else 1.0


def build_model(model_name: str, params: dict, y_train=None):
    if model_name == "logistic":
        return LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=config.RANDOM_STATE, **params,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced", random_state=config.RANDOM_STATE, n_jobs=-1, **params,
        )
    if model_name == "xgboost":
        return XGBClassifier(
            scale_pos_weight=_scale_pos_weight(y_train) if y_train is not None else 1.0,
            random_state=config.RANDOM_STATE, n_jobs=-1,
            tree_method="hist", eval_metric="auc", **params,
        )
    raise ValueError(f"Modelo nao reconhecido: {model_name}")


def evaluate_model(pipeline, model_name, X_test, y_test) -> dict:
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    report_dict = classification_report(y_test, predictions, output_dict=True)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "f1_macro": f1_score(y_test, predictions, average="macro"),
        "recall_nao_alfabetizado": report_dict.get("0", report_dict.get("0.0", {})).get("recall", 0.0),
        "recall_alfabetizado": report_dict.get("1", report_dict.get("1.0", {})).get("recall", 0.0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    log.info("===== %s =====", model_name)
    for k, v in metrics.items():
        if k != "model":
            log.info("%s: %.4f", k.upper(), v)

    return metrics


def main() -> None:
    credentials, _ = google.auth.default()
    client = bigquery.Client(project=config.GCP_PROJECT_ID, credentials=credentials)

    X_train, X_test, y_train, y_test, groups_train, groups_test = prepare_data(client)

    best = load_best_params()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for model_name in ["logistic", "random_forest", "xgboost"]:
        params = best.get(model_name, {}).get("best_params", DEFAULT_PARAMS[model_name])
        log.info("Treinando %s com params: %s", model_name, params)

        model = build_model(model_name, params, y_train=y_train)
        # Pipeline unico (preprocessor + modelo) -- e o artefato salvo,
        # nao mais modelo/preprocessor separados. Balanceamento via
        # class_weight/scale_pos_weight, sem descartar dados (ver preprocessing.py).
        pipeline = build_full_pipeline(model, apply_undersampling=False)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, model_name, X_test, y_test)
        metrics["params"] = json.dumps(params, ensure_ascii=False)
        results.append(metrics)

        model_path = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(pipeline, model_path)
        log.info("Pipeline completo salvo em %s", model_path)

    df_results = pd.DataFrame(results)
    df_results.to_csv(REPORTS_DIR / "model_results.csv", index=False)

    champion = df_results.loc[df_results["roc_auc"].idxmax(), "model"]
    log.info("Modelo campeao (maior ROC-AUC no teste agrupado por aluno): %s", champion)

    print("\n===== COMPARACAO DOS MODELOS =====")
    print(
        df_results[
            ["model", "accuracy", "f1_macro", "recall_nao_alfabetizado", "recall_alfabetizado", "roc_auc"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
