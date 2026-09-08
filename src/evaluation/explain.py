import argparse
import logging
from pathlib import Path

import google.auth
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from google.cloud import bigquery

import config
from src.preprocessing.features import prepare_data
from src.visualization.eda_plots import UF_NOMES
from src.visualization.style import BLUE, INK_MUTED, apply_style

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
IMAGES_DIR = Path("images")

TOP_N_IMPORTANCE = 20


def _nome_legivel(nome: str) -> str:
    """
    Nomes crus pos-ColumnTransformer (ex.: "numeric__taxa_alfabetizacao_municipio",
    "categorical__sigla_uf_code_23") viram ilegiveis num grafico com 40+ barras.
    Tira o prefixo do transformer e traduz o codigo IBGE de UF para a sigla.
    """
    nome = nome.split("__", 1)[-1]
    if nome.startswith("sigla_uf_code_"):
        codigo = nome.replace("sigla_uf_code_", "")
        return f"UF: {UF_NOMES.get(codigo, codigo)}"
    if nome.startswith("rede_"):
        return f"Rede: {nome.replace('rede_', '')}"
    return nome


def generate_shap_and_importance(model_filename: str = "xgboost.joblib", sample_size: int = 2000) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado em {model_path}. Rode 'python -m src.modeling.train' antes.")

    log.info("Carregando pipeline: %s", model_path)
    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    credentials, _ = google.auth.default()
    client = bigquery.Client(project=config.GCP_PROJECT_ID, credentials=credentials)
    _, X_test, _, y_test, _, _ = prepare_data(client)

    # Nomes reais pos-ColumnTransformer (nao hardcoded -- refletem as
    # categorias que o OneHotEncoder efetivamente viu no treino).
    feature_names = list(preprocessor.get_feature_names_out())
    X_test_processed = preprocessor.transform(X_test)

    sample_size = min(sample_size, X_test_processed.shape[0])
    rng = np.random.default_rng(config.RANDOM_STATE)
    sample_idx = rng.choice(X_test_processed.shape[0], size=sample_size, replace=False)
    X_sample = X_test_processed[sample_idx]

    model_tag = model_filename.replace(".joblib", "")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        pct = importances / importances.sum() * 100
        df_imp = pd.DataFrame({
            "variavel": feature_names,
            "importancia": importances,
            "contribuicao_pct": np.round(pct, 2),
        }).sort_values("contribuicao_pct", ascending=False)

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        df_imp.to_csv(REPORTS_DIR / f"feature_importance_{model_tag}.csv", index=False)
        log.info("\n===== FEATURE IMPORTANCE (%s) =====\n%s", model_tag, df_imp.to_string(index=False))

        # O CSV completo tem 40+ variaveis (uma por categoria do OneHotEncoder) --
        # ilegivel num unico grafico. Mostra as top N e dobra o resto numa unica
        # barra "Outras", em vez de truncar a informacao em silencio.
        top = df_imp.head(TOP_N_IMPORTANCE).copy()
        resto = df_imp.iloc[TOP_N_IMPORTANCE:]
        top["rotulo"] = top["variavel"].apply(_nome_legivel)
        if not resto.empty:
            outras = pd.DataFrame([{
                "variavel": "outras", "rotulo": f"Outras {len(resto)} variaveis (UF, rede)",
                "importancia": resto["importancia"].sum(),
                "contribuicao_pct": round(resto["contribuicao_pct"].sum(), 2),
            }])
            df_plot = pd.concat([top, outras], ignore_index=True)
        else:
            df_plot = top
        df_plot = df_plot.sort_values("contribuicao_pct", ascending=True)

        altura = max(6, 0.32 * len(df_plot) + 1.5)
        plt.figure(figsize=(10, altura))
        cores = [INK_MUTED if v == "outras" else BLUE for v in df_plot["variavel"]]
        plt.barh(df_plot["rotulo"], df_plot["contribuicao_pct"], color=cores)
        plt.title(f"Fatores que mais impactam a alfabetizacao ({model_tag})", fontsize=12, fontweight="bold")
        plt.xlabel("Contribuicao relativa (%)")
        plt.grid(axis="x")
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / f"feature_importance_{model_tag}.png", dpi=300)
        plt.close()

    log.info("Calculando valores SHAP (amostra de %d registros)...", sample_size)
    model_class = model.__class__.__name__
    if model_class in ("XGBClassifier", "RandomForestClassifier"):
        explainer = shap.TreeExplainer(model)
    else:
        background = shap.sample(
            X_test_processed, min(200, X_test_processed.shape[0]), random_state=config.RANDOM_STATE
        )
        explainer = shap.Explainer(model, background)

    shap_values = explainer(X_sample)
    if shap_values.values.ndim == 3:
        # TreeExplainer em classificador binario devolve (n, features, classes) --
        # ficamos com a classe positiva (1 = Alfabetizado).
        shap_values = shap_values[..., 1]
    shap_values.feature_names = feature_names

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title("Impacto direcional das variaveis na alfabetizacao (SHAP)", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / f"shap_summary_{model_tag}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("Por que este aluno especifico foi classificado assim? (SHAP)", fontsize=11, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / f"shap_waterfall_{model_tag}.png", dpi=300)
    plt.close()

    log.info("Graficos salvos em %s", IMAGES_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="random_forest.joblib")
    args = parser.parse_args()
    generate_shap_and_importance(args.model)
