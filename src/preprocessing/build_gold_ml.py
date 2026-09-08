import sys
import logging

from google.cloud import bigquery
import google.auth

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, ".")
import config

GCP_PROJECT_ID = config.GCP_PROJECT_ID
BQ_DATASET_GOLD = config.BQ_DATASET_GOLD
BQ_DATASET_SILVER = config.BQ_DATASET_SILVER
ML_FEATURES_TABLE = config.ML_FEATURES_TABLE

# Projeto publico da Base dos Dados no BigQuery -- mesma origem ja usada pela
# ingestao da Fase 2 (pipeline-alfabetizacao/ingestion/batch/ingest_bronze.py).
# O enriquecimento e um JOIN cross-project: nao ha download nem nova ingestao.
BDD_PROJECT = "basedosdados"

PARTITION_BY_ANO = {ML_FEATURES_TABLE}
CLUSTER_FIELDS = {
    ML_FEATURES_TABLE: ["id_municipio", "rede"],
}

# Query herdada de NaiaraMartins/1IAST-Tech-Challenge-Fase-3 (src/gold/build_gold_ml.py),
# com id_aluno preservado para permitir split/CV agrupado por aluno (evita o mesmo
# aluno aparecer em treino e teste quando ele tem mais de um caderno na mesma edicao),
# mais 3 melhorias sobre a v2:
#
# 1. municipio_same_year: a v2 fazia AVG(taxa_alfabetizacao) misturando 2023+2024
#    pra toda linha (sem filtrar por ano) -- ou seja, uma linha de 2023 recebia
#    informacao de 2024 (que ainda nao existia). Corrigido para casar pelo MESMO
#    ano da linha.
# 2. escola_prior_year: feature nova, no grao de ESCOLA (mais fino que municipio),
#    calculada a partir do proprio alunos_clean agregado por id_escola+ano e casada
#    com ano_alvo = ano+1 -- ou seja, so usa o desempenho da escola no ano ANTERIOR
#    (nunca o mesmo ano/mesma turma), sem risco de vazamento. Cobertura: ~79% das
#    linhas de 2024 tem historico de 2023; linhas de 2023 ficam NULL (nao ha 2022
#    nos dados) -- sinalizado explicitamente por tem_historico_escola.
#    (Tentamos ligar INSE por escola tambem, mas id_escola em inse_escola_clean
#    e silver.alunos_clean nao compartilham o mesmo namespace de codigo -- 0 de
#    42.802 escolas batem mesmo apos normalizar o tipo. Confirmado por query
#    direta antes de descartar a ideia.)
# 3. sigla_uf_code: os 2 primeiros digitos do id_municipio (codigo IBGE) --
#    variavel territorial adicional, praticamente gratuita.
ML_GOLD_TABLES = {
    ML_FEATURES_TABLE: f"""
        WITH municipio_same_year AS (
            SELECT
                SUBSTR(CAST(id_municipio AS STRING), 1, 6) AS id_municipio_6dig,
                CAST(ano AS INT64) AS ano,
                AVG(SAFE_CAST(REPLACE(CAST(taxa_alfabetizacao AS STRING), ',', '.') AS FLOAT64)) AS taxa_alfabetizacao_municipio,
                AVG(SAFE_CAST(REPLACE(CAST(media_portugues AS STRING), ',', '.') AS FLOAT64)) AS media_portugues_municipio,
                AVG(SAFE_CAST(REPLACE(CAST(proporcao_abaixo_basico AS STRING), ',', '.') AS FLOAT64)) AS proporcao_abaixo_basico,
                AVG(SAFE_CAST(REPLACE(CAST(proporcao_basico AS STRING), ',', '.') AS FLOAT64)) AS proporcao_basico,
                AVG(SAFE_CAST(REPLACE(CAST(proporcao_adequado_avancado AS STRING), ',', '.') AS FLOAT64)) AS proporcao_adequado_avancado
            FROM `{GCP_PROJECT_ID}.{BQ_DATASET_SILVER}.alfabetizacao_municipio_clean`
            WHERE TRIM(serie) = '2° ano do Ensino Fundamental'
            GROUP BY 1, 2
        ),
        escola_prior_year AS (
            SELECT
                TRIM(CAST(SAFE_CAST(id_escola AS INT64) AS STRING)) AS id_escola,
                CAST(ano AS INT64) + 1 AS ano_alvo,
                AVG(CASE WHEN alfabetizado = 'Sim' THEN 1.0 ELSE 0.0 END) AS taxa_alfabetizacao_escola_prior,
                COUNT(*) AS n_alunos_prior_escola
            FROM `{GCP_PROJECT_ID}.{BQ_DATASET_SILVER}.alunos_clean`
            WHERE presenca = 'Presente'
            GROUP BY 1, 2
        ),
        inse_municipio_agregado AS (
            SELECT
                SUBSTR(CAST(id_municipio AS STRING), 1, 6) AS id_municipio_6dig,
                AVG(SAFE_CAST(REPLACE(CAST(inse AS STRING), ',', '.') AS FLOAT64)) AS inse_municipio
            FROM `{GCP_PROJECT_ID}.{BQ_DATASET_SILVER}.inse_escola_clean`
            WHERE id_municipio IS NOT NULL
            GROUP BY 1
        ),
        metas_municipio AS (
            -- Metas municipais do Compromisso Nacional Crianca Alfabetizada.
            -- O PDF lista "Metas nacionais e estaduais" e "Metas municipais"
            -- como parte da base analitica; alem de aderencia ao enunciado,
            -- percentual_participacao e nivel_alfabetizacao sao variaveis
            -- preditivas que nao estavam sendo usadas.
            -- Casado pelo MESMO ano da linha (meta e trajetoria publicada,
            -- definida antes da avaliacao -- nao deriva do resultado do aluno).
            SELECT
                SUBSTR(CAST(id_municipio AS STRING), 1, 6) AS id_municipio_6dig,
                CAST(ano AS INT64) AS ano,
                MAX(meta_alfabetizacao_2030) AS meta_2030,
                MAX(meta_alfabetizacao_2024) AS meta_2024,
                MAX(percentual_participacao) AS percentual_participacao,
                MAX(SAFE_CAST(nivel_alfabetizacao AS INT64)) AS nivel_alfabetizacao
            FROM `{GCP_PROJECT_ID}.{BQ_DATASET_SILVER}.metas_consolidadas`
            WHERE escopo = 'municipio' AND id_municipio IS NOT NULL
            GROUP BY 1, 2
        ),
        -- ------------------------------------------------------------------
        -- Enriquecimento externo: Censo Escolar + Indicadores Educacionais.
        -- Fontes citadas nominalmente no enunciado (PDF pg.3 e pg.4).
        --
        -- POR QUE NO GRAO MUNICIPIO x REDE E NAO NO GRAO DE ESCOLA:
        -- verificado por query em 31/08/2026 que silver.alunos_clean.id_escola
        -- e um contador sequencial anonimizado na origem (60000001..60042811
        -- para 42.802 escolas; o prefixo 60 nem e codigo de UF valido), e nao
        -- o CO_ENTIDADE do INEP. Cruzamento com basedosdados: saeb x censo = 0.
        -- O INSE, por contraste, casa 100% com o Censo (69.756/69.756) -- ou
        -- seja, quem esta mascarado e o SAEB, e nao existe de-para publicado.
        -- Ja o id_municipio casa 5.547/5.547 (IBGE, 7 digitos), entao o grao
        -- mais fino alcancavel e municipio x rede: 6.701 celulas contra 5.547
        -- municipios, 11,66 escolas por celula, 445 celulas com escola unica.
        --
        -- Dominios validados por query:
        --   censo.rede            -> '1' federal, '2' estadual, '3' municipal, '4' privada
        --   censo.tipo_localizacao-> '1' urbana, '2' rural
        --   indicadores.rede      -> minusculo ('estadual', 'municipal', ...)
        --   alunos_clean.rede     -> capitalizado ('Estadual', 'Municipal')
        censo_municipio_rede AS (
            -- Estrutura fisica da escola. Coletado no Censo de maio; a prova do
            -- SAEB e aplicada em out/nov. Sao caracteristicas de ENTRADA, ja
            -- conhecidas no momento da predicao -- mesmo ano nao vaza.
            SELECT
                TRIM(CAST(id_municipio AS STRING)) AS id_municipio_7dig,
                CASE TRIM(CAST(rede AS STRING))
                    WHEN '2' THEN 'Estadual'
                    WHEN '3' THEN 'Municipal'
                END AS rede,
                CAST(ano AS INT64) AS ano,
                AVG(CASE WHEN TRIM(CAST(tipo_localizacao AS STRING)) = '2' THEN 1.0 ELSE 0.0 END) AS pct_escolas_rurais,
                AVG(CASE WHEN biblioteca = 1 OR biblioteca_sala_leitura = 1 THEN 1.0 ELSE 0.0 END) AS pct_escolas_biblioteca,
                AVG(SAFE_CAST(internet AS FLOAT64)) AS pct_escolas_internet,
                AVG(SAFE_CAST(agua_potavel AS FLOAT64)) AS pct_escolas_agua_potavel,
                AVG(SAFE_CAST(esgoto_rede_publica AS FLOAT64)) AS pct_escolas_esgoto_publico,
                AVG(SAFE_CAST(energia_rede_publica AS FLOAT64)) AS pct_escolas_energia_publica,
                COUNT(*) AS n_escolas_censo_celula
            FROM `{BDD_PROJECT}.br_inep_censo_escolar.escola`
            WHERE ano BETWEEN 2023 AND 2024
              AND TRIM(CAST(rede AS STRING)) IN ('2', '3')
              AND id_municipio IS NOT NULL
            GROUP BY 1, 2, 3
        ),
        indicadores_municipio_rede AS (
            -- Turma e corpo docente no 2o ano do EF -- exatamente a serie do
            -- target. Tambem sao insumos apurados no Censo de maio, anteriores
            -- a prova: ATU (alunos por turma), HAD (horas-aula diarias), TDI
            -- (distorcao idade-serie, estrutura etaria da matricula), AFD
            -- (adequacao da formacao docente), IRD (regularidade do docente),
            -- DSU e ICG (complexidade de gestao).
            SELECT
                TRIM(CAST(id_municipio AS STRING)) AS id_municipio_7dig,
                INITCAP(TRIM(rede)) AS rede,
                CAST(ano AS INT64) AS ano,
                AVG(atu_ef_2_ano) AS atu_2ano,
                AVG(had_ef_2_ano) AS had_2ano,
                AVG(tdi_ef_2_ano) AS tdi_2ano,
                AVG(afd_ef_anos_iniciais_grupo_1) AS afd_grupo1_pct,
                AVG(ird_media_regularidade_docente) AS ird_medio,
                AVG(dsu_ef_anos_iniciais) AS dsu_medio,
                AVG(SAFE_CAST(icg_nivel_complexidade_gestao_escola AS FLOAT64)) AS icg_medio
            FROM `{BDD_PROJECT}.br_inep_indicadores_educacionais.escola`
            WHERE ano BETWEEN 2023 AND 2024
              AND TRIM(rede) IN ('estadual', 'municipal')
              AND id_municipio IS NOT NULL
            GROUP BY 1, 2, 3
        ),
        rendimento_prior_municipio_rede AS (
            -- Aprovacao/reprovacao/abandono sao RESULTADO do ano letivo, apurados
            -- no fechamento -- na mesma janela em que a crianca faz a prova e
            -- sobre a mesma coorte. Usar no mesmo ano seria o mesmo vazamento
            -- same-cohort documentado na secao 7.5 do README (TargetEncoder em
            -- id_escola). Por isso entram deslocados: ano + 1 = ano_alvo.
            SELECT
                TRIM(CAST(id_municipio AS STRING)) AS id_municipio_7dig,
                INITCAP(TRIM(rede)) AS rede,
                CAST(ano AS INT64) + 1 AS ano_alvo,
                AVG(taxa_reprovacao_ef_2_ano) AS taxa_reprovacao_2ano_prior,
                AVG(taxa_abandono_ef_2_ano) AS taxa_abandono_2ano_prior
            FROM `{BDD_PROJECT}.br_inep_indicadores_educacionais.escola`
            WHERE ano BETWEEN 2022 AND 2023
              AND TRIM(rede) IN ('estadual', 'municipal')
              AND id_municipio IS NOT NULL
            GROUP BY 1, 2, 3
        )
        SELECT
            CAST(a.ano AS INT64) AS ano,
            a.id_aluno,
            SUBSTR(CAST(a.id_municipio AS STRING), 1, 6) AS id_municipio,
            SUBSTR(CAST(a.id_municipio AS STRING), 1, 2) AS sigla_uf_code,
            TRIM(CAST(SAFE_CAST(a.id_escola AS INT64) AS STRING)) AS id_escola,
            TRIM(a.rede) AS rede,
            a.presenca,
            SAFE_CAST(REPLACE(CAST(a.peso_aluno AS STRING), ',', '.') AS FLOAT64) AS peso_aluno,
            a.alfabetizado,

            i.inse_municipio,

            m.taxa_alfabetizacao_municipio,
            m.media_portugues_municipio,
            m.proporcao_abaixo_basico,
            m.proporcao_basico,
            m.proporcao_adequado_avancado,

            e.taxa_alfabetizacao_escola_prior,
            e.n_alunos_prior_escola,
            CASE WHEN e.id_escola IS NOT NULL THEN 1 ELSE 0 END AS tem_historico_escola,

            mt.meta_2030,
            mt.meta_2024,
            mt.percentual_participacao,
            mt.nivel_alfabetizacao,
            m.taxa_alfabetizacao_municipio - mt.meta_2030 AS gap_meta_2030,

            ce.pct_escolas_rurais,
            ce.pct_escolas_biblioteca,
            ce.pct_escolas_internet,
            ce.pct_escolas_agua_potavel,
            ce.pct_escolas_esgoto_publico,
            ce.pct_escolas_energia_publica,
            ce.n_escolas_censo_celula,

            ie.atu_2ano,
            ie.had_2ano,
            ie.tdi_2ano,
            ie.afd_grupo1_pct,
            ie.ird_medio,
            ie.dsu_medio,
            ie.icg_medio,

            rp.taxa_reprovacao_2ano_prior,
            rp.taxa_abandono_2ano_prior,
            CASE WHEN ce.id_municipio_7dig IS NOT NULL THEN 1 ELSE 0 END AS tem_censo_escolar

        FROM `{GCP_PROJECT_ID}.{BQ_DATASET_SILVER}.alunos_clean` a

        LEFT JOIN inse_municipio_agregado i
            ON SUBSTR(CAST(a.id_municipio AS STRING), 1, 6) = i.id_municipio_6dig

        LEFT JOIN municipio_same_year m
            ON SUBSTR(CAST(a.id_municipio AS STRING), 1, 6) = m.id_municipio_6dig
            AND CAST(a.ano AS INT64) = m.ano

        LEFT JOIN escola_prior_year e
            ON TRIM(CAST(SAFE_CAST(a.id_escola AS INT64) AS STRING)) = e.id_escola
            AND CAST(a.ano AS INT64) = e.ano_alvo

        LEFT JOIN metas_municipio mt
            ON SUBSTR(CAST(a.id_municipio AS STRING), 1, 6) = mt.id_municipio_6dig
            AND CAST(a.ano AS INT64) = mt.ano

        -- Os joins de enriquecimento usam o id_municipio COMPLETO (7 digitos,
        -- codigo IBGE). Diferente dos joins acima, que usam 6 digitos porque as
        -- tabelas de alfabetizacao/metas trazem o codigo sem o digito verificador.
        -- Cobertura verificada: 5.547 de 5.547 municipios casam com o Censo.
        LEFT JOIN censo_municipio_rede ce
            ON TRIM(CAST(a.id_municipio AS STRING)) = ce.id_municipio_7dig
            AND TRIM(a.rede) = ce.rede
            AND CAST(a.ano AS INT64) = ce.ano

        LEFT JOIN indicadores_municipio_rede ie
            ON TRIM(CAST(a.id_municipio AS STRING)) = ie.id_municipio_7dig
            AND TRIM(a.rede) = ie.rede
            AND CAST(a.ano AS INT64) = ie.ano

        LEFT JOIN rendimento_prior_municipio_rede rp
            ON TRIM(CAST(a.id_municipio AS STRING)) = rp.id_municipio_7dig
            AND TRIM(a.rede) = rp.rede
            AND CAST(a.ano AS INT64) = rp.ano_alvo

        WHERE a.id_municipio IS NOT NULL
          AND a.id_aluno IS NOT NULL
    """,
}


def ensure_dataset(client: bigquery.Client, dataset_id: str) -> None:
    dataset_ref = bigquery.Dataset(f"{GCP_PROJECT_ID}.{dataset_id}")
    dataset_ref.location = "US"
    client.create_dataset(dataset_ref, exists_ok=True)
    log.info(f"Dataset gold '{dataset_id}' verificado ou criado com sucesso.")


def build_table(client: bigquery.Client, table_name: str, query: str) -> int:
    destination = f"{GCP_PROJECT_ID}.{BQ_DATASET_GOLD}.{table_name}"

    job_config = bigquery.QueryJobConfig(
        destination=destination,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
    )

    if table_name in PARTITION_BY_ANO:
        job_config.range_partitioning = bigquery.RangePartitioning(
            field="ano",
            range_=bigquery.PartitionRange(start=2019, end=2051, interval=1),
        )

    if table_name in CLUSTER_FIELDS:
        job_config.clustering_fields = CLUSTER_FIELDS[table_name]

    log.info(f"Executando query de compilacao da tabela gold.{table_name}...")
    job = client.query(query, job_config=job_config)
    job.result()

    table = client.get_table(destination)
    log.info(f"Tabela gold.{table_name} processada! Contem {table.num_rows:,} registros.")
    return table.num_rows


def main() -> None:
    log.info("Iniciando execucao da Camada Gold ML (versao com id_aluno)...")
    credentials, _ = google.auth.default()
    client = bigquery.Client(project=GCP_PROJECT_ID, credentials=credentials)

    ensure_dataset(client, BQ_DATASET_GOLD)

    for table_name, query in ML_GOLD_TABLES.items():
        build_table(client, table_name, query)

    log.info("Processamento da camada Gold ML concluido com sucesso!")


if __name__ == "__main__":
    main()
