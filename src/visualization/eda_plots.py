import logging
from pathlib import Path

import google.auth
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from google.cloud import bigquery

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

IMAGES_DIR = Path("images")
GEOJSON_UF_PATH = Path("data/geo/br_uf.geojson")

UF_NOMES = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
    "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
    "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP", "41": "PR",
    "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF",
}

NUMERIC_FEATURES = [
    "taxa_alfabetizacao_municipio",
    "media_portugues_municipio",
    "proporcao_abaixo_basico",
    "proporcao_basico",
    "proporcao_adequado_avancado",
    "inse_municipio",
    "peso_aluno",
    "taxa_alfabetizacao_escola_prior",
    "n_alunos_prior_escola",
    "tem_historico_escola",
]


def load_gold_data(client: bigquery.Client) -> pd.DataFrame:
    query = f"""
        SELECT
            taxa_alfabetizacao_municipio,
            media_portugues_municipio,
            proporcao_abaixo_basico,
            proporcao_basico,
            proporcao_adequado_avancado,
            inse_municipio,
            peso_aluno,
            taxa_alfabetizacao_escola_prior,
            n_alunos_prior_escola,
            tem_historico_escola,
            rede,
            sigla_uf_code,
            alfabetizado
        FROM `{config.GCP_PROJECT_ID}.{config.BQ_DATASET_GOLD}.{config.ML_FEATURES_TABLE}`
        WHERE rede IN ('Municipal', 'Estadual')
    """
    log.info("Carregando dados para Analise Exploratoria...")
    return client.query(query).to_dataframe()


def run_exploratory_analysis(df: pd.DataFrame) -> dict:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    log.info("Gerando estatisticas descritivas...")
    stats = {
        "shape": df.shape,
        "null_count": df.isna().sum().to_dict(),
        "target_distribution": df["alfabetizado"].value_counts(normalize=True).to_dict(),
        "describe_numeric": df[NUMERIC_FEATURES].describe().to_dict(),
    }

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="alfabetizado", hue="alfabetizado", palette="Blues_r", legend=False)
    plt.title("Distribuicao da Variavel Alvo (Alfabetizado)")
    plt.xlabel("Status de Alfabetizacao")
    plt.ylabel("Quantidade de registros aluno x edicao")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "target_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    df_corr = df[NUMERIC_FEATURES].copy()
    df_corr["target_num"] = df["alfabetizado"].map({"Não": 0, "Nao": 0, "Sim": 1, 0: 0, 1: 1})
    corr_matrix = df_corr.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues", square=True)
    plt.title("Matriz de Correlacao - Camada Gold")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "correlation_matrix.png", dpi=300)
    plt.close()

    if "inse_municipio" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="alfabetizado", y="inse_municipio", hue="alfabetizado", palette="Set2", legend=False)
        plt.title("Distribuicao do INSE Municipal por Status de Alfabetizacao")
        plt.xlabel("Alfabetizado")
        plt.ylabel("INSE do Municipio")
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "inse_vs_target.png", dpi=300)
        plt.close()

    if "sigla_uf_code" in df.columns:
        plot_taxa_alfabetizacao_uf_mapa(df)

    log.info("Analise exploratoria concluida! Graficos salvos em: %s", IMAGES_DIR)
    return stats


def plot_taxa_alfabetizacao_uf_mapa(df: pd.DataFrame, geojson_path: Path = GEOJSON_UF_PATH) -> pd.Series:
    """
    Mapa coropletico da taxa de alfabetizacao por UF -- complementa o grafico
    de barras equivalente (mesma serie de dados). Sugestao do Prof. Gabriel
    Ortelan (feedback de 05/09) para tornar a disparidade territorial
    (Hipotese H3 da EDA) mais intuitiva do que a leitura de 27 barras.

    geojson_path aceita caminho relativo -- por padrao relativo ao diretorio
    de execucao (cwd=raiz do projeto). Chamado do notebook, passar
    "../data/geo/br_uf.geojson".
    """
    import geopandas as gpd

    uf = df["sigla_uf_code"].map(UF_NOMES)
    taxa_uf = df.groupby(uf)["alfabetizado"].apply(lambda s: (s == "Sim").mean() * 100)

    br_uf = gpd.read_file(geojson_path)
    br_uf["taxa_alfabetizacao"] = br_uf["sigla"].map(taxa_uf)

    sem_dado = sorted(set(br_uf["sigla"]) - set(taxa_uf.index))
    if sem_dado:
        log.warning("UFs sem nenhuma linha na gold (aparecem em branco no mapa): %s", sem_dado)

    fig, ax = plt.subplots(figsize=(9, 9))
    br_uf.plot(
        column="taxa_alfabetizacao", cmap="RdYlGn", legend=True,
        edgecolor="white", linewidth=0.6, ax=ax,
        legend_kwds={"label": "% alfabetizados", "shrink": 0.6},
    )
    for _, row in br_uf.iterrows():
        centroid = row.geometry.representative_point()
        ax.annotate(row["sigla"], (centroid.x, centroid.y), ha="center", fontsize=8, fontweight="bold")
    ax.set_title("Taxa de alfabetizacao por UF", fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(IMAGES_DIR / "taxa_alfabetizacao_uf_mapa.png", dpi=300)
    plt.close()

    return taxa_uf


def main() -> None:
    credentials, _ = google.auth.default()
    client = bigquery.Client(project=config.GCP_PROJECT_ID, credentials=credentials)
    df = load_gold_data(client)
    run_exploratory_analysis(df)


if __name__ == "__main__":
    main()
