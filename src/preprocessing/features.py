import logging
from typing import Tuple

import pandas as pd
from google.cloud import bigquery
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TARGET_COLUMN = config.TARGET_COLUMN
GROUP_COLUMN = config.GROUP_COLUMN

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
    # Metas do Compromisso Nacional Crianca Alfabetizada (PDF pg.3-4).
    # Sao trajetorias publicadas ANTES da avaliacao, nao derivadas do
    # resultado do aluno -- nao ha vazamento em usa-las no mesmo ano.
    #
    # meta_2030 e gap_meta_2030 foram TESTADAS E REMOVIDAS por redundancia
    # matematica (nao por falta de aderencia ao enunciado -- continuam na
    # tabela gold e alimentam as analises de negocio em
    # src/evaluation/business_questions.py):
    #   - meta_2030 tem UM unico valor distinto na base (80.0 para todos os
    #     municipios) -> variancia zero, o modelo atribuiu 0.0% de importancia.
    #   - gap_meta_2030 = taxa_alfabetizacao_municipio - 80.0, portanto tem
    #     correlacao 1.0000 com taxa_alfabetizacao_municipio -> nao carrega
    #     informacao nova, apenas divide a importancia com ela e faz a tabela
    #     de Feature Importance parecer que "metas explicam 17,75%" quando e
    #     a taxa municipal reapresentada.
    "meta_2024",
    "percentual_participacao",
    "nivel_alfabetizacao",
]

# Enriquecimento externo (Censo Escolar + Indicadores Educacionais), no grao
# municipio x rede. Fontes citadas nominalmente no PDF (pg.3-4). O grao de
# escola era o alvo original, mas e inalcancavel: id_escola do SAEB e um
# contador anonimizado na origem -- ver comentario extenso em build_gold_ml.py
# e a secao "Limitacoes" do README.
#
# Cobertura verificada na gold v3 (3.354.637 linhas, redes Municipal/Estadual):
#   pct_escolas_*        100.00%      atu_2ano        99.99%
#   n_escolas_censo      100.00%      afd_grupo1_pct  99.80%
#   dsu_medio             99.99%      had_2ano        85.33%
#   tdi_2ano              83.59%      ird_medio       83.59%
#
# TESTADAS E EXCLUIDAS por cobertura insuficiente (seguem na tabela gold, como
# meta_2030, para nao perder o registro da tentativa):
#   - icg_medio ......................  3,75% -- icg_nivel_complexidade_gestao_escola
#     e rotulo textual no INEP, o SAFE_CAST numerico devolve NULL na quase
#     totalidade das linhas.
#   - taxa_reprovacao_2ano_prior .... 10,50% -- e 0,00% em TODO o ano de 2023
#     (o INEP nao publicou reprovacao de 2o ano em 2022). Imputar a mediana em
#     ~90% das linhas so injetaria ruido.
#   - taxa_abandono_2ano_prior ...... 55,20% -- presente em 2024 e ausente em
#     2023, entao a propria presenca do valor vira proxy da variavel `ano`.
#   - tem_censo_escolar ............. 100% de cobertura => variancia zero,
#     inutil como flag (o equivalente de tem_historico_escola nao se aplica aqui).
CENSO_FEATURES = [
    "pct_escolas_rurais",
    "pct_escolas_biblioteca",
    "pct_escolas_internet",
    "pct_escolas_agua_potavel",
    "pct_escolas_esgoto_publico",
    "pct_escolas_energia_publica",
    "n_escolas_censo_celula",
    "atu_2ano",
    "had_2ano",
    "tdi_2ano",
    "afd_grupo1_pct",
    "ird_medio",
    "dsu_medio",
]

# Para o A/B do enriquecimento: basta trocar por [] para reproduzir o baseline
# v2 (ROC-AUC 0,6881) com o mesmo split e os mesmos hiperparametros.
NUMERIC_FEATURES = NUMERIC_FEATURES + CENSO_FEATURES

CATEGORICAL_FEATURES = ["rede", "sigla_uf_code"]

# Colunas de identificacao: nunca entram como feature do modelo, mas sao
# devolvidas por prepare_data() para as analises de negocio (ranking de
# municipios em risco, clusterizacao regional, projecao de metas).
ID_COLUMNS = ["id_municipio", "id_escola", "ano"]

# TargetEncoder em id_municipio/id_escola foi TESTADO E DESCARTADO -- ver
# secao "Vazamento same-cohort" no README. Resumo: o cross-fitting interno
# do TargetEncoder protege contra vazar o target da PROPRIA LINHA, mas nao
# contra vazar o target dos COLEGAS DE ESCOLA medidos na mesma edicao do
# SAEB. Como treino e teste sao divididos por aluno (nao por escola/ano),
# codificar id_escola pela media dos alunos de treino entrega ao modelo o
# resultado da turma do proprio aluno -- informacao que nao existiria numa
# predicao real, onde a prova daquele ano ainda nao aconteceu.
# Evidencia (mesmo split de teste, logistic com params do Optuna):
#   sem target encoding ............ 0.6820
#   so id_municipio ................ 0.6821  (+0.0001, ruido)
#   id_municipio + id_escola ....... 0.7032  (+0.0212, todo o ganho e do id_escola)
# Reforcado pelo sinal classico de vazamento: o ROC-AUC de teste (0.7032)
# ficava ACIMA do ROC-AUC da validacao cruzada (0.6840). O efeito equivalente
# e legitimo ja esta capturado por taxa_alfabetizacao_escola_prior, que usa
# apenas o ano ANTERIOR.
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(client: bigquery.Client) -> pd.DataFrame:
    """
    Carrega a gold ML apontada por config.ML_FEATURES_TABLE (v3: mesma logica
    de join de NaiaraMartins/1IAST-Tech-Challenge-Fase-3, com id_aluno
    preservado, mais o enriquecimento Censo Escolar / Indicadores Educacionais
    no grao municipio x rede).
    """
    censo_cols = "".join(f"{c},\n            " for c in CENSO_FEATURES)
    query = f"""
        SELECT
            {GROUP_COLUMN},
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
            meta_2030,
            meta_2024,
            gap_meta_2030,
            percentual_participacao,
            nivel_alfabetizacao,
            {censo_cols}
            rede,
            sigla_uf_code,
            id_municipio,
            id_escola,
            ano,
            {TARGET_COLUMN}
        FROM `{config.GCP_PROJECT_ID}.{config.BQ_DATASET_GOLD}.{config.ML_FEATURES_TABLE}`
        WHERE rede IN ('Municipal', 'Estadual')
    """
    log.info("Carregando dados da Gold ML no BigQuery...")
    df = client.query(query).to_dataframe()
    log.info("Dados carregados: %s registros", f"{len(df):,}")
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Separa features (X), target binario (y) e grupo (id_aluno, usado so
    para split/CV -- nunca entra como feature do modelo).
    """
    groups = df[GROUP_COLUMN]
    X = df[FEATURE_COLUMNS]

    y = df[TARGET_COLUMN].map({"Não": 0, "Nao": 0, "Sim": 1, "0": 0, "1": 1, 0: 0, 1: 1})
    if y.isna().any():
        raise ValueError("A variavel target possui valores nulos ou mapeamentos nao reconhecidos.")

    return X, y, groups


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def build_full_pipeline(model, apply_undersampling: bool = False) -> ImbPipeline:
    """
    Um unico objeto sklearn/imblearn Pipeline com pre-processamento +
    balanceamento + estimador -- atende a exigencia do PDF de "integracao
    do pre-processamento diretamente ao modelo" (na v1 do repo, o
    ColumnTransformer era ajustado fora do Pipeline do modelo).

    apply_undersampling=False por padrao: o desbalanceamento (~59%/41%) e
    tratado via class_weight="balanced" (Logistic/RandomForest) ou
    scale_pos_weight (XGBoost) no proprio modelo, sem descartar ~600k
    linhas da classe majoritaria por fold como o RandomUnderSampler da v1
    fazia.
    """
    steps = [("preprocessor", build_preprocessor())]
    if apply_undersampling:
        steps.append(("undersampler", RandomUnderSampler(random_state=config.RANDOM_STATE)))
    steps.append(("model", model))
    return ImbPipeline(steps=steps)


def train_test_split_grouped(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    """
    GroupShuffleSplit por id_aluno: garante que o mesmo aluno (que pode
    ter ate 2 linhas na base -- 2023 e 2024) nunca aparece em treino E
    teste ao mesmo tempo.
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return (
        X.iloc[train_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[test_idx],
        groups.iloc[train_idx], groups.iloc[test_idx],
    )


def prepare_data(client: bigquery.Client, return_ids: bool = False):
    """
    return_ids=True devolve tambem os identificadores (id_municipio,
    id_escola, ano) das linhas de teste -- usados pelas analises de negocio
    (src/evaluation/business_questions.py). Eles NUNCA entram como feature.
    """
    df = load_data(client)
    X, y, groups = split_features_target(df)

    log.info("Distribuicao do Target: 0=%s | 1=%s", f"{(y == 0).sum():,}", f"{(y == 1).sum():,}")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

    overlap = set(groups_train) & set(groups_test)
    log.info(
        "Split agrupado por id_aluno -> Treino: %s | Teste: %s | Alunos em comum (deve ser 0): %d",
        f"{len(X_train):,}", f"{len(X_test):,}", len(overlap),
    )

    if return_ids:
        ids_test = df[ID_COLUMNS].iloc[test_idx]
        return X_train, X_test, y_train, y_test, groups_train, groups_test, ids_test

    return X_train, X_test, y_train, y_test, groups_train, groups_test


if __name__ == "__main__":
    import google.auth
    credentials, _ = google.auth.default()
    client = bigquery.Client(project=config.GCP_PROJECT_ID, credentials=credentials)
    prepare_data(client)
