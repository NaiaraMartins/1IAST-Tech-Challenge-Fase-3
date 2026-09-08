"""
A/B da variavel peso_aluno.

O professor questionou se peso_aluno faz sentido como feature preditiva --
e provavelmente o peso amostral do SAEB (usado para ponderar estimativas
populacionais), nao uma causa real de alfabetizacao. Este teste compara o
conjunto completo de features contra o mesmo conjunto sem peso_aluno,
mantendo FIXOS o split agrupado por id_aluno, os hiperparametros do Optuna
e a semente -- de modo que a unica diferenca entre os dois bracos seja essa
variavel.

Roda tambem CV agrupada no braco sem peso_aluno para o modelo campeao, mesmo
controle de vazamento da secao 7.5 do README: se o ROC-AUC de teste subir
ACIMA do de CV, o "ganho" e suspeito e nao deve ser reportado como melhoria.

    python -m src.evaluation.ab_peso_aluno
"""

import json
import logging
from pathlib import Path

import google.auth
import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_score

import config
from src.modeling.train import DEFAULT_PARAMS, build_model, load_best_params
from src.preprocessing import features as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")
CV_FOLDS_AB = 3
MODELOS = ["logistic", "random_forest", "xgboost"]


def main() -> None:
    credentials, _ = google.auth.default()
    client = bigquery.Client(project=config.GCP_PROJECT_ID, credentials=credentials)

    df = F.load_data(client)
    X, y, groups = F.split_features_target(df)
    X_train, X_test, y_train, y_test, g_train, _ = F.train_test_split_grouped(X, y, groups)
    log.info("Treino: %s | Teste: %s", f"{len(X_train):,}", f"{len(X_test):,}")

    sem_peso_cols = [c for c in F.FEATURE_COLUMNS if c != "peso_aluno"]
    bracos = {
        "com_peso_aluno": F.FEATURE_COLUMNS,
        "sem_peso_aluno": sem_peso_cols,
    }

    best = load_best_params()
    linhas = []

    for braco, cols in bracos.items():
        num = [c for c in cols if c in F.NUMERIC_FEATURES]
        cat = [c for c in cols if c in F.CATEGORICAL_FEATURES]

        for nome in MODELOS:
            params = best.get(nome, {}).get("best_params", DEFAULT_PARAMS[nome])
            modelo = build_model(nome, params, y_train=y_train)

            # Reaproveita o pipeline de producao, restringindo as colunas ao braco.
            orig_num, orig_cat = F.NUMERIC_FEATURES, F.CATEGORICAL_FEATURES
            F.NUMERIC_FEATURES, F.CATEGORICAL_FEATURES = num, cat
            try:
                pipe = F.build_full_pipeline(modelo, apply_undersampling=False)
                log.info("[%s] treinando %s (%d features)...", braco, nome, len(cols))
                pipe.fit(X_train[cols], y_train)
                proba = pipe.predict_proba(X_test[cols])[:, 1]

                auc = roc_auc_score(y_test, proba)
                pred = (proba >= 0.5).astype(int)
                linhas.append({
                    "braco": braco,
                    "model": nome,
                    "n_features": len(cols),
                    "roc_auc_teste": auc,
                    "accuracy": accuracy_score(y_test, pred),
                    "f1_macro": f1_score(y_test, pred, average="macro"),
                })
                log.info("[%s] %s -> ROC-AUC teste = %.4f", braco, nome, auc)
            finally:
                F.NUMERIC_FEATURES, F.CATEGORICAL_FEATURES = orig_num, orig_cat

    res = pd.DataFrame(linhas)

    # Controle de vazamento no braco sem peso_aluno: CV agrupada por id_aluno.
    campeao = res[res.braco == "sem_peso_aluno"].sort_values("roc_auc_teste").iloc[-1]["model"]
    log.info("Rodando CV agrupada (%d folds) para %s no braco sem_peso_aluno...", CV_FOLDS_AB, campeao)
    params = best.get(campeao, {}).get("best_params", DEFAULT_PARAMS[campeao])

    orig_num, orig_cat = F.NUMERIC_FEATURES, F.CATEGORICAL_FEATURES
    F.NUMERIC_FEATURES = [c for c in orig_num if c != "peso_aluno"]
    try:
        pipe = F.build_full_pipeline(build_model(campeao, params, y_train=y_train))
        cv_scores = cross_val_score(
            pipe, X_train[sem_peso_cols], y_train,
            groups=g_train, cv=GroupKFold(n_splits=CV_FOLDS_AB),
            scoring="roc_auc", n_jobs=1,
        )
    finally:
        F.NUMERIC_FEATURES, F.CATEGORICAL_FEATURES = orig_num, orig_cat

    cv_media = float(np.mean(cv_scores))
    auc_teste = float(res[(res.braco == "sem_peso_aluno") & (res.model == campeao)]["roc_auc_teste"].iloc[0])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(REPORTS_DIR / "ab_peso_aluno.csv", index=False)

    pivot = res.pivot(index="model", columns="braco", values="roc_auc_teste")
    pivot["delta"] = pivot["com_peso_aluno"] - pivot["sem_peso_aluno"]

    resumo = {
        "campeao_sem_peso_aluno": campeao,
        "roc_auc_teste": auc_teste,
        "roc_auc_cv_media": cv_media,
        "cv_folds": CV_FOLDS_AB,
        "delta_teste_menos_cv": auc_teste - cv_media,
        "suspeita_de_vazamento": bool(auc_teste - cv_media > 0.01),
        "delta_por_modelo": {m: float(pivot.loc[m, "delta"]) for m in pivot.index},
    }
    with open(REPORTS_DIR / "ab_peso_aluno_resumo.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)

    print("\n===== A/B: ROC-AUC no teste agrupado por aluno (com vs. sem peso_aluno) =====")
    print(pivot.to_string())
    print(f"\nCampeao sem peso_aluno: {campeao}")
    print(f"  ROC-AUC teste .... {auc_teste:.4f}")
    print(f"  ROC-AUC CV ({CV_FOLDS_AB}f) .. {cv_media:.4f}")
    print(f"  teste - CV ....... {auc_teste - cv_media:+.4f}"
          f"  {'<-- SUSPEITA DE VAZAMENTO' if resumo['suspeita_de_vazamento'] else '(consistente)'}")


if __name__ == "__main__":
    main()
