import argparse
import logging
from pathlib import Path

import google.auth
import joblib
import pandas as pd
from google.cloud import bigquery
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import config
from src.preprocessing.features import prepare_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

THRESHOLDS = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]


def run_threshold_simulation(y_test, y_proba, thresholds=THRESHOLDS) -> pd.DataFrame:
    """
    Simula o impacto de diferentes limiares de corte na deteccao de
    alunos em risco (classe 0 = Nao Alfabetizado). Em politicas de
    Busca Ativa, o custo de um Falso Negativo (nao identificar uma
    crianca que precisa de apoio) e maior que o de um Falso Positivo.
    """
    results = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        results.append({
            "threshold": t,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
            "recall_nao_alfabetizado": round(recall_score(y_test, y_pred, pos_label=0), 4),
            "precision_nao_alfabetizado": round(precision_score(y_test, y_pred, pos_label=0), 4),
        })
    return pd.DataFrame(results)


def main(model_filename: str = "xgboost.joblib") -> pd.DataFrame:
    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado em {model_path}. Rode 'python -m src.modeling.train' antes.")

    log.info("Carregando pipeline: %s", model_path)
    pipeline = joblib.load(model_path)

    credentials, _ = google.auth.default()
    client = bigquery.Client(project=config.GCP_PROJECT_ID, credentials=credentials)
    _, X_test, _, y_test, _, _ = prepare_data(client)

    y_proba = pipeline.predict_proba(X_test)[:, 1]

    df_sim = run_threshold_simulation(y_test, y_proba)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"threshold_simulation_{model_filename.replace('.joblib', '')}.csv"
    df_sim.to_csv(out_path, index=False)

    log.info("\n===== SIMULACAO DE THRESHOLD (%s) =====\n%s", model_filename, df_sim.to_string(index=False))

    best_row = df_sim.loc[df_sim["recall_nao_alfabetizado"].idxmax()]
    log.info(
        "Threshold com maior recall p/ risco (Nao Alfabetizado): %.2f (recall=%.4f, precision=%.4f)",
        best_row["threshold"], best_row["recall_nao_alfabetizado"], best_row["precision_nao_alfabetizado"],
    )
    log.info("Simulacao salva em %s", out_path)
    return df_sim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="random_forest.joblib")
    args = parser.parse_args()
    main(args.model)
