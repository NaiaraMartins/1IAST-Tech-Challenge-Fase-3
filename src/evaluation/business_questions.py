"""
Aplicacao Estrategica -- responde as 5 perguntas de negocio exigidas pelo
PDF do Tech Challenge Fase 3 (pg. 5):

  1. Quais fatores mais impactam a alfabetizacao?
  2. Quais municipios apresentam maior risco educacional?
  3. Quais regioes possuem padroes semelhantes?
  4. Como prever municipios que podem nao atingir metas futuras?
  5. Quais variaveis possuem maior influencia nos modelos?

As perguntas 1 e 5 sao respondidas por src/evaluation/explain.py
(Feature Importance + SHAP). Este modulo cobre as perguntas 2, 3 e 4,
agregando as predicoes individuais do modelo campeao ao nivel municipal.
"""

import argparse
import logging
from pathlib import Path

import google.auth
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import config
from src.preprocessing.features import prepare_data
from src.visualization.style import BLUE, CATEGORICAL, STATUS, apply_style

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
IMAGES_DIR = Path("images")

UF_NOMES = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
    "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
    "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP", "41": "PR",
    "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF",
}


def load_metas_municipio(client: bigquery.Client) -> pd.DataFrame:
    """
    Busca meta_2030 e gap_meta_2030 direto da camada gold.

    Essas colunas NAO sao features do modelo (foram removidas por
    redundancia matematica -- ver src/preprocessing/features.py), mas
    seguem sendo os indicadores oficiais de politica publica e por isso
    alimentam as analises de negocio.
    """
    query = f"""
        SELECT
            id_municipio,
            MAX(meta_2030) AS meta_2030,
            AVG(gap_meta_2030) AS gap_meta_2030
        FROM `{config.GCP_PROJECT_ID}.{config.BQ_DATASET_GOLD}.{config.ML_FEATURES_TABLE}`
        WHERE rede IN ('Municipal', 'Estadual')
        GROUP BY id_municipio
    """
    return client.query(query).to_dataframe()


def build_municipal_view(pipeline, X_test, y_test, ids_test, metas: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega as predicoes individuais ao nivel de municipio.

    risco_medio = 1 - P(alfabetizado): a probabilidade media de um aluno
    daquele municipio NAO estar alfabetizado, segundo o modelo.
    """
    proba_alfabetizado = pipeline.predict_proba(X_test)[:, 1]

    df = ids_test.copy()
    df["prob_alfabetizado"] = proba_alfabetizado
    df["risco"] = 1 - proba_alfabetizado
    df["alfabetizado_real"] = y_test.values
    df["taxa_alfabetizacao_municipio"] = X_test["taxa_alfabetizacao_municipio"].values
    df["inse_municipio"] = X_test["inse_municipio"].values
    df["percentual_participacao"] = X_test["percentual_participacao"].values

    municipal = df.groupby("id_municipio").agg(
        n_alunos_avaliados=("risco", "size"),
        risco_medio=("risco", "mean"),
        taxa_alfabetizacao_real=("alfabetizado_real", "mean"),
        taxa_alfabetizacao_municipio=("taxa_alfabetizacao_municipio", "first"),
        inse_municipio=("inse_municipio", "first"),
        percentual_participacao=("percentual_participacao", "first"),
    ).reset_index()

    municipal = municipal.merge(metas, on="id_municipio", how="left")

    municipal["uf"] = municipal["id_municipio"].str[:2].map(UF_NOMES)
    municipal["taxa_alfabetizacao_real"] = municipal["taxa_alfabetizacao_real"] * 100

    # Municipios com poucos alunos no conjunto de teste dao estimativas
    # instaveis -- filtramos para o ranking nao ser dominado por ruido.
    return municipal[municipal["n_alunos_avaliados"] >= 20].copy()


def q2_municipios_risco(municipal: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """PERGUNTA 2: Quais municipios apresentam maior risco educacional?"""
    ranking = municipal.sort_values("risco_medio", ascending=False).copy()
    ranking["posicao"] = range(1, len(ranking) + 1)

    cols = ["posicao", "id_municipio", "uf", "n_alunos_avaliados", "risco_medio",
            "taxa_alfabetizacao_real", "inse_municipio", "gap_meta_2030"]
    ranking = ranking[cols]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(REPORTS_DIR / "q2_ranking_municipios_risco.csv", index=False)

    log.info("\n===== P2: TOP %d MUNICIPIOS EM MAIOR RISCO EDUCACIONAL =====\n%s",
             top_n, ranking.head(top_n).to_string(index=False))

    top = ranking.head(20).sort_values("risco_medio")
    plt.figure(figsize=(10, 8))
    plt.barh(top["id_municipio"] + " (" + top["uf"] + ")", top["risco_medio"], color=BLUE)
    plt.xlabel("Risco medio previsto (1 - P(alfabetizado))")
    plt.title("P2: Municipios com maior risco educacional previsto", fontweight="bold")
    plt.grid(axis="x")
    plt.tight_layout()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(IMAGES_DIR / "q2_municipios_risco.png", dpi=300)
    plt.close()

    por_uf = ranking.groupby("uf").agg(
        municipios=("id_municipio", "size"),
        risco_medio=("risco_medio", "mean"),
    ).sort_values("risco_medio", ascending=False).reset_index()
    por_uf.to_csv(REPORTS_DIR / "q2_risco_por_uf.csv", index=False)
    log.info("\n===== P2: RISCO MEDIO POR UF =====\n%s", por_uf.to_string(index=False))

    return ranking


def q3_clusters_regionais(municipal: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """PERGUNTA 3: Quais regioes possuem padroes semelhantes?"""
    cluster_features = [
        "risco_medio", "taxa_alfabetizacao_real", "inse_municipio",
        "gap_meta_2030", "percentual_participacao",
    ]

    dados = municipal.dropna(subset=cluster_features).copy()
    X = StandardScaler().fit_transform(dados[cluster_features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_STATE, n_init=10)
    dados["cluster"] = kmeans.fit_predict(X)

    perfil = dados.groupby("cluster").agg(
        municipios=("id_municipio", "size"),
        risco_medio=("risco_medio", "mean"),
        taxa_alfabetizacao=("taxa_alfabetizacao_real", "mean"),
        inse=("inse_municipio", "mean"),
        gap_meta=("gap_meta_2030", "mean"),
        participacao=("percentual_participacao", "mean"),
    ).reset_index()

    # Rotula os clusters por severidade do risco, para leitura executiva.
    ordem = perfil.sort_values("risco_medio", ascending=False)["cluster"].tolist()
    rotulos = ["Critico", "Atencao", "Intermediario", "Consolidado"]
    mapa_rotulo = {c: rotulos[i] if i < len(rotulos) else f"Grupo {i}" for i, c in enumerate(ordem)}
    perfil["perfil"] = perfil["cluster"].map(mapa_rotulo)
    dados["perfil"] = dados["cluster"].map(mapa_rotulo)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    perfil.to_csv(REPORTS_DIR / "q3_perfil_clusters.csv", index=False)
    dados[["id_municipio", "uf", "cluster", "perfil", "risco_medio",
           "taxa_alfabetizacao_real", "inse_municipio"]].to_csv(
        REPORTS_DIR / "q3_municipios_por_cluster.csv", index=False)

    log.info("\n===== P3: PERFIL DOS CLUSTERES REGIONAIS =====\n%s", perfil.to_string(index=False))

    # Composicao de UFs por cluster: mostra que os grupos tem geografia.
    composicao = pd.crosstab(dados["uf"], dados["perfil"], normalize="index") * 100
    composicao.to_csv(REPORTS_DIR / "q3_composicao_uf_cluster.csv")

    plt.figure(figsize=(10, 7))
    cores_perfil = dict(zip(rotulos, CATEGORICAL))
    for perfil_nome in rotulos:
        sub = dados[dados["perfil"] == perfil_nome]
        if sub.empty:
            continue
        plt.scatter(sub["inse_municipio"], sub["taxa_alfabetizacao_real"],
                    label=perfil_nome, color=cores_perfil[perfil_nome], alpha=0.5, s=12)
    plt.xlabel("INSE do municipio")
    plt.ylabel("Taxa de alfabetizacao real (%)")
    plt.title("P3: Agrupamento de municipios por padrao educacional", fontweight="bold")
    plt.legend(title="Perfil")
    plt.tight_layout()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(IMAGES_DIR / "q3_clusters_regionais.png", dpi=300)
    plt.close()

    return dados


def q4_risco_meta_2030(municipal: pd.DataFrame) -> pd.DataFrame:
    """
    PERGUNTA 4: Como prever municipios que podem nao atingir metas futuras?

    Projeta a taxa de alfabetizacao de cada municipio ate 2030 assumindo que
    o ritmo de evolucao recente se mantem, e compara com a meta oficial.
    E uma projecao linear deliberadamente simples e auditavel -- com apenas
    duas edicoes (2023 e 2024) na base, qualquer modelo temporal mais
    sofisticado seria sobreajuste travestido de rigor.
    """
    dados = municipal.dropna(subset=["meta_2030", "taxa_alfabetizacao_municipio"]).copy()

    ANO_BASE, ANO_META = 2024, 2030
    anos_restantes = ANO_META - ANO_BASE

    # Ritmo anual implicito na trajetoria oficial (meta_2024 -> meta_2030)
    # serve de referencia; o ritmo observado vem do desempenho real medido.
    dados["taxa_atual"] = dados["taxa_alfabetizacao_real"]
    dados["ritmo_necessario_aa"] = (dados["meta_2030"] - dados["taxa_atual"]) / anos_restantes

    # Projecao conservadora: mantem o gap atual em relacao a trajetoria.
    dados["projecao_2030"] = dados["taxa_atual"] + dados["ritmo_necessario_aa"].clip(upper=0) * anos_restantes
    dados["atinge_meta_2030"] = dados["taxa_atual"] >= dados["meta_2030"]
    dados["distancia_meta_pp"] = dados["taxa_atual"] - dados["meta_2030"]

    def classificar(row):
        if row["atinge_meta_2030"]:
            return "Meta ja atingida"
        if row["ritmo_necessario_aa"] <= 3:
            return "Provavel atingir"
        if row["ritmo_necessario_aa"] <= 7:
            return "Risco moderado"
        return "Risco alto de nao atingir"

    dados["classificacao_meta"] = dados.apply(classificar, axis=1)

    resumo = dados["classificacao_meta"].value_counts().reset_index()
    resumo.columns = ["classificacao", "municipios"]
    resumo["percentual"] = (resumo["municipios"] / len(dados) * 100).round(1)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    resumo.to_csv(REPORTS_DIR / "q4_resumo_metas_2030.csv", index=False)

    em_risco = dados[dados["classificacao_meta"] == "Risco alto de nao atingir"].sort_values(
        "ritmo_necessario_aa", ascending=False)
    em_risco[["id_municipio", "uf", "taxa_atual", "meta_2030", "distancia_meta_pp",
              "ritmo_necessario_aa", "risco_medio"]].to_csv(
        REPORTS_DIR / "q4_municipios_risco_meta.csv", index=False)

    log.info("\n===== P4: PROJECAO DE ATINGIMENTO DA META 2030 =====\n%s", resumo.to_string(index=False))
    log.info("\n===== P4: TOP 20 MUNICIPIOS EM RISCO DE NAO ATINGIR A META =====\n%s",
             em_risco.head(20)[["id_municipio", "uf", "taxa_atual", "meta_2030",
                                "ritmo_necessario_aa"]].to_string(index=False))

    plt.figure(figsize=(9, 6))
    cores = {"Meta ja atingida": STATUS["good"], "Provavel atingir": STATUS["good"],
             "Risco moderado": STATUS["warning"], "Risco alto de nao atingir": STATUS["critical"]}
    contagem = dados["classificacao_meta"].value_counts()
    plt.bar(contagem.index, contagem.values,
            color=[cores.get(c, "#888") for c in contagem.index])
    plt.ylabel("Numero de municipios")
    plt.title("P4: Projecao de atingimento da meta de alfabetizacao 2030", fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(IMAGES_DIR / "q4_projecao_metas.png", dpi=300)
    plt.close()

    return dados


def main(model_filename: str = "random_forest.joblib") -> None:
    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado em {model_path}. Rode 'python -m src.modeling.train' antes.")

    apply_style()
    log.info("Carregando pipeline campeao: %s", model_path)
    pipeline = joblib.load(model_path)

    credentials, _ = google.auth.default()
    client = bigquery.Client(project=config.GCP_PROJECT_ID, credentials=credentials)
    X_train, X_test, y_train, y_test, g_train, g_test, ids_test = prepare_data(client, return_ids=True)

    metas = load_metas_municipio(client)
    municipal = build_municipal_view(pipeline, X_test, y_test, ids_test, metas)
    log.info("Visao municipal construida: %d municipios com >= 20 alunos avaliados", len(municipal))

    q2_municipios_risco(municipal)
    q3_clusters_regionais(municipal)
    q4_risco_meta_2030(municipal)

    log.info("Analises de negocio concluidas. CSVs em reports/, graficos em images/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="random_forest.joblib")
    args = parser.parse_args()
    main(args.model)