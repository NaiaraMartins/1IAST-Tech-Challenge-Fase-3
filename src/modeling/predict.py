import logging
from pathlib import Path

import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = Path("models")


def load_pipeline(model_filename: str = "xgboost.joblib"):
    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado em: {model_path}")
    log.info("Carregando pipeline: %s", model_path)
    return joblib.load(model_path)


def predict_risk(
    df_input: pd.DataFrame,
    model_filename: str = "xgboost.joblib",
    threshold: float = 0.55,
) -> pd.DataFrame:
    """
    Recebe dados brutos (mesmas colunas da Gold ML) e aplica o pipeline
    completo (preprocessor + modelo em um unico objeto) direto -- nao ha
    mais preprocessor separado pra carregar.
    """
    pipeline = load_pipeline(model_filename)

    log.info("Calculando probabilidades de alfabetizacao...")
    probabilities_alfabetizado = pipeline.predict_proba(df_input)[:, 1]

    df_results = df_input.copy()
    df_results["probabilidade_alfabetizado"] = probabilities_alfabetizado.round(4)
    df_results["predicao_final"] = (probabilities_alfabetizado >= threshold).astype(int)
    df_results["status_risco"] = df_results["predicao_final"].map({
        0: "ALTO RISCO (Busca Ativa)",
        1: "Baixo Risco / Adequado",
    })

    log.info("Inferencia concluida com sucesso!")
    return df_results


def main() -> None:
    """
    Exemplo de inferencia: dois alunos ficticios em contextos opostos --
    um municipio consolidado e um municipio critico (ver perfis de cluster
    em reports/q3_perfil_clusters.csv). As colunas espelham exatamente
    FEATURE_COLUMNS de src/preprocessing/features.py.
    """
    sample_data = pd.DataFrame([
        {
            "taxa_alfabetizacao_municipio": 83.8,
            "media_portugues_municipio": 780.0,
            "proporcao_abaixo_basico": 5.0,
            "proporcao_basico": 25.0,
            "proporcao_adequado_avancado": 70.0,
            "inse_municipio": 4.73,
            "peso_aluno": 1.0,
            "taxa_alfabetizacao_escola_prior": 0.82,
            "n_alunos_prior_escola": 45,
            "tem_historico_escola": 1,
            "meta_2024": 60.0,
            "percentual_participacao": 94.7,
            "nivel_alfabetizacao": 4,
            "rede": "Municipal",
            "sigla_uf_code": "42",
        },
        {
            "taxa_alfabetizacao_municipio": 36.4,
            "media_portugues_municipio": 700.0,
            "proporcao_abaixo_basico": 45.0,
            "proporcao_basico": 40.0,
            "proporcao_adequado_avancado": 15.0,
            "inse_municipio": 4.43,
            "peso_aluno": 1.0,
            "taxa_alfabetizacao_escola_prior": 0.31,
            "n_alunos_prior_escola": 28,
            "tem_historico_escola": 1,
            "meta_2024": 12.0,
            "percentual_participacao": 84.3,
            "nivel_alfabetizacao": 0,
            "rede": "Municipal",
            "sigla_uf_code": "29",
        },
    ])

    log.info("Executando inferencia de teste...")
    results = predict_risk(sample_data, model_filename="random_forest.joblib", threshold=0.55)

    print("\n===== RESULTADOS DA PREDICAO (INFERENCIA) =====")
    print(results[["sigla_uf_code", "taxa_alfabetizacao_municipio", "inse_municipio",
                   "probabilidade_alfabetizado", "status_risco"]].to_string(index=False))


if __name__ == "__main__":
    main()
