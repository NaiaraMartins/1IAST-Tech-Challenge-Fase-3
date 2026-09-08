# Tech Challenge - Fase 3: Predicao da Alfabetizacao Infantil e Busca Ativa

**PosTech FIAP | IA Science**

## 1. Contexto do Problema

A alfabetizacao na idade certa e um dos principais desafios da educacao basica publica no Brasil. O **Compromisso Nacional Crianca Alfabetizada** estabelece que todas as criancas estejam alfabetizadas ao final do 2o ano do Ensino Fundamental ate 2030, medido pelo **Indicador Crianca Alfabetizada** do INEP (corte de 743 pontos na escala SAEB).

Compreender apenas os dados atuais nao basta: gestores publicos precisam **antecipar riscos**, identificar regioes vulneraveis e entender quais fatores mais impactam o indicador.

## 2. Objetivo Analitico

Desenvolver um modelo supervisionado capaz de prever se um aluno sera considerado **alfabetizado ou nao alfabetizado**, usando variaveis educacionais, territoriais e socioeconomicas provenientes da camada **Gold** construida na Fase 2, e transformar essa predicao em inteligencia aplicada a politicas de **Busca Ativa** e **Reforco Escolar**.

## 3. Origem e Colaboracao

Fork de [NaiaraMartins/1IAST-Tech-Challenge-Fase-3](https://github.com/NaiaraMartins/1IAST-Tech-Challenge-Fase-3), que construiu a base de dados, a EDA e o pipeline sklearn inicial. Este fork adiciona otimizacao com Optuna, corrige tres fontes de vazamento, incorpora as metas oficiais e responde as cinco perguntas de negocio do desafio (secao 9).

## 4. Descricao da Base Utilizada

Tabela `gold.ml_features_alunos_v3`, com grao de **aluno** (`id_aluno`), 3.354.661 registros das edicoes 2023 e 2024, construida por `src/preprocessing/build_gold_ml.py` a partir das camadas silver/gold da Fase 2:

| Fonte (Fase 2) | Contribuicao |
|---|---|
| `silver.alunos_clean` | Grao de aluno, rotulo `alfabetizado`, rede, peso amostral |
| `silver.alfabetizacao_municipio_clean` | Indicador Crianca Alfabetizada, media de portugues, proporcoes por nivel |
| `silver.inse_escola_clean` | Indicador socioeconomico (INSE), agregado por municipio |
| `silver.metas_consolidadas` | Metas municipais 2024-2030, participacao, nivel de alfabetizacao |
| `id_municipio` (IBGE) | Dados territoriais: municipio e UF |

### Enriquecimento com fontes externas

O enunciado (pg.3-4) autoriza expressamente enriquecer a base analitica com fontes externas e cita o **Censo Escolar** entre elas. A v3 acrescenta duas, ambas lidas por JOIN cross-project no BigQuery publico da Base dos Dados -- **sem download e sem nova ingestao**, reaproveitando a mesma origem ja usada pela pipeline da Fase 2:

| Fonte externa | Grao | Contribuicao |
|---|---|---|
| `basedosdados.br_inep_censo_escolar.escola` | municipio x rede x ano | Localizacao rural, biblioteca, internet, agua potavel, esgoto e energia da rede publica |
| `basedosdados.br_inep_indicadores_educacionais.escola` | municipio x rede x ano | ATU, HAD e TDI **do 2o ano do EF**; AFD, IRD e DSU do corpo docente |

**Por que municipio x rede e nao escola.** O grao de escola era o alvo, mas e inalcancavel: o `id_escola` do SAEB esta anonimizado na origem (secao 12.2). Ja o `id_municipio` casa integralmente com o Censo Escolar -- **5.547 de 5.547 municipios**, codigo IBGE de 7 digitos. Cruzando municipio com rede chega-se a **6.701 celulas** contra 5.547 municipios puros, com media de 11,66 escolas por celula; em **445 celulas ha uma unica escola**, e nelas o agregado equivale a informacao de escola.

Os indicadores foram escolhidos na versao **por serie** (`atu_ef_2_ano`, `had_ef_2_ano`, `tdi_ef_2_ano`) em vez da agregada de anos iniciais, para casar exatamente com a serie do target.

### Features do modelo

| Grupo | Variaveis |
|---|---|
| Educacionais | `taxa_alfabetizacao_municipio`, `media_portugues_municipio`, `proporcao_abaixo_basico`, `proporcao_basico`, `proporcao_adequado_avancado` |
| Socioeconomicas | `inse_municipio` |
| Metas | `meta_2024`, `percentual_participacao`, `nivel_alfabetizacao` |
| Temporais | `taxa_alfabetizacao_escola_prior`, `n_alunos_prior_escola`, `tem_historico_escola` |
| Territoriais | `sigla_uf_code`, `rede` |
| Amostral | `peso_aluno` |
| **Infraestrutura escolar** (Censo Escolar) | `pct_escolas_rurais`, `pct_escolas_biblioteca`, `pct_escolas_internet`, `pct_escolas_agua_potavel`, `pct_escolas_esgoto_publico`, `pct_escolas_energia_publica`, `n_escolas_censo_celula` |
| **Turma e docencia** (Indicadores Educacionais) | `atu_2ano`, `had_2ano`, `tdi_2ano`, `afd_grupo1_pct`, `ird_medio`, `dsu_medio` |

## 5. Estrutura do Projeto

```text
data/                        # Dados locais (nao versionados)
notebooks/
  01_analise_exploratoria.ipynb
images/                      # Graficos: EDA, Feature Importance, SHAP, perguntas de negocio
reports/                     # CSVs de metricas e analises
src/
  preprocessing/
    build_gold_ml.py         # Constroi gold.ml_features_alunos_v3 no BigQuery
    features.py              # Split agrupado, ColumnTransformer, Pipeline unico
  modeling/
    tune.py                  # Otimizacao de hiperparametros (Optuna)
    train.py                 # Treino final e comparativo
    predict.py               # Motor de inferencia
  evaluation/
    evaluate.py              # Threshold tuning (regra de negocio)
    explain.py               # Feature Importance + SHAP
    business_questions.py    # Perguntas de negocio (secao 9)
    ab_censo_enrichment.py   # A/B do enriquecimento externo (secao 8)
  visualization/
    eda_plots.py             # Graficos da analise exploratoria
config.py
requirements.txt
```

## 6. Etapas de Modelagem

1. **Camada Gold ML** (`build_gold_ml.py`): integra as fontes da Fase 2 no grao de aluno e acrescenta o enriquecimento externo (Censo Escolar e Indicadores Educacionais) por JOIN cross-project no BigQuery publico da Base dos Dados.
2. **Analise exploratoria** (`notebooks/01_analise_exploratoria.ipynb` + `eda_plots.py`): distribuicoes, correlacoes, nulos e formulacao das hipoteses H1-H4. A disparidade territorial da Hipotese H3 (secao 7 do notebook) tem tambem um mapa coropletico por UF, gerado por `plot_taxa_alfabetizacao_uf_mapa` em `eda_plots.py` -- sugestao do Prof. Gabriel Ortelan pra tornar a disparidade mais intuitiva do que 27 barras. O proprio mapa evidenciou algo que a barra escondia: **Roraima (RR) nao tem nenhuma linha na gold** (`sigla_uf_code` so cobre 26 dos 27 estados), por isso aparece em branco.

   ![Taxa de alfabetizacao por UF](images/taxa_alfabetizacao_uf_mapa.png)
3. **Pipeline de pre-processamento** integrado ao modelo em um unico objeto sklearn:
   - `SimpleImputer(median)` + `StandardScaler` nas numericas;
   - `SimpleImputer(most_frequent)` + `OneHotEncoder` nas categoricas;
   - balanceamento via `class_weight` / `scale_pos_weight`.
4. **Otimizacao** (`tune.py`): Optuna com `StratifiedGroupKFold`.
5. **Treino e avaliacao** (`train.py`, `evaluate.py`) na base completa.
6. **Interpretabilidade** (`explain.py`): Feature Importance + SHAP.
7. **Aplicacao estrategica** (`business_questions.py`).

## 7. Tratamento de Data Leakage

Esta foi a area de maior esforco do projeto. Cinco fontes de vazamento foram identificadas e tratadas (7.1 a 7.5); as secoes 7.6 a 7.8 documentam variaveis removidas por redundancia ou cobertura e o criterio que decide quais variaveis externas podem entrar no mesmo ano.

### 7.1 `proficiencia` define o rotulo
`alfabetizado = 'Sim'` equivale exatamente a `proficiencia >= 743` (corte SAEB), confirmado por query direta. A variavel foi **excluida das features**.

### 7.2 Mesmo aluno em treino e teste
Consulta a base revelou que **51,2% dos alunos aparecem em duas linhas** (edicoes 2023 e 2024). Um `train_test_split` aleatorio por linha colocaria o mesmo aluno nos dois conjuntos. Todo o projeto usa `GroupShuffleSplit` / `StratifiedGroupKFold` **agrupados por `id_aluno`**, com verificacao automatica de sobreposicao a cada execucao (`Alunos em comum: 0`).

### 7.3 Mistura de anos nos agregados municipais
A versao original calculava a media de `taxa_alfabetizacao` misturando 2023 e 2024 para todas as linhas -- uma linha de 2023 recebia informacao de 2024. Corrigido para casar pelo **mesmo ano** da linha.

### 7.4 Historico de escola do mesmo ano
`taxa_alfabetizacao_escola_prior` usa exclusivamente o **ano anterior** (`ano - 1`), nunca a mesma turma/edicao. Cobertura: ~79% das linhas de 2024; linhas de 2023 ficam sem historico (nao ha 2022 na base) e isso e sinalizado pela flag `tem_historico_escola`, em vez de fabricar valor.

### 7.5 Vazamento same-cohort via TargetEncoder (testado e revertido)

Aplicamos `sklearn.preprocessing.TargetEncoder` em `id_municipio` e `id_escola` esperando ganho legitimo. O ROC-AUC de teste subiu de 0,6820 para 0,7032 -- mas **acima do ROC-AUC da validacao cruzada (0,6840)**, sinal classico de vazamento. Teste controlado no mesmo split:

| Configuracao | ROC-AUC teste |
|---|---|
| Sem target encoding | 0,6820 |
| Apenas `id_municipio` | 0,6821 (+0,0001, ruido) |
| `id_municipio` + `id_escola` | 0,7032 (+0,0212) |

**Todo o ganho vinha de `id_escola`.** O cross-fitting interno do TargetEncoder protege contra vazar o target da propria linha, mas nao contra vazar o target dos **colegas de escola medidos na mesma edicao do SAEB** -- informacao indisponivel numa predicao real, onde a prova daquele ano ainda nao ocorreu. A tecnica foi **revertida**, e o efeito legitimo equivalente permanece capturado por `taxa_alfabetizacao_escola_prior` (ano anterior).

### 7.6 Features redundantes removidas

`meta_2030` tem **um unico valor distinto** na base (80,0 para todos os municipios) e recebeu 0,0% de importancia. `gap_meta_2030` = `taxa_alfabetizacao_municipio - 80,0`, portanto **correlacao 1,0000** com ela. Mantidas, apareciam com 17,75% de "importancia" que na verdade era a taxa municipal reapresentada. Foram removidas das features (permanecem na gold, alimentando as analises de negocio). O ROC-AUC ficou inalterado (0,6882 -> 0,6881), confirmando que nao carregavam informacao.

### 7.7 Insumo x resultado nas variaveis do Censo Escolar

O enriquecimento da v3 exigiu separar, dentro da mesma fonte, o que pode entrar no mesmo ano do que precisa ser defasado. O criterio nao e a data de publicacao, e **se a variavel e insumo ou desfecho**:

| Tipo | Variaveis | Tratamento |
|---|---|---|
| **Insumo** -- apurado no Censo de maio, anterior a prova de out/nov | infraestrutura, `atu_2ano`, `had_2ano`, `tdi_2ano`, `afd_grupo1_pct`, `ird_medio`, `dsu_medio` | Mesmo ano. Sao condicoes conhecidas antes da avaliacao e disponiveis numa predicao real. |
| **Desfecho** -- apurado no fechamento do ano letivo | `taxa_reprovacao_ef_2_ano`, `taxa_abandono_ef_2_ano` | Defasadas (`ano + 1`). Usa-las no mesmo ano reproduziria o vazamento same-cohort da secao 7.5: sao resultado da mesma coorte, na mesma janela da prova. |

`tdi_ef_2_ano` merece nota, porque a leitura apressada e classifica-la como desfecho. Distorcao idade-serie e um **estoque medido na matricula de maio** -- descreve a estrutura etaria da turma que chega, nao o que ela alcancou no fim do ano. Por isso entra no mesmo ano, junto com os demais insumos.

Na pratica as duas variaveis defasadas acabaram descartadas por cobertura (secao seguinte), mas o criterio fica registrado porque e ele que sustenta manter as outras treze no mesmo ano.

### 7.8 Variaveis do enriquecimento descartadas por cobertura

Quatro das dezessete variaveis geradas nao entraram no modelo. Seguem na tabela gold -- mesmo tratamento dado a `meta_2030` --, para preservar o registro da tentativa:

| Variavel | Cobertura | Motivo |
|---|---|---|
| `icg_medio` | 3,75% | `icg_nivel_complexidade_gestao_escola` e rotulo textual no INEP; o cast numerico devolve NULL em quase tudo. |
| `taxa_reprovacao_2ano_prior` | 10,50% | **0,00% em todo o ano de 2023** -- o INEP nao publicou reprovacao de 2o ano em 2022. |
| `taxa_abandono_2ano_prior` | 55,20% | Presente em 2024 e ausente em 2023: a propria presenca do valor viraria proxy da variavel `ano`. |
| `tem_censo_escolar` | 100% | Variancia zero -- ao contrario de `tem_historico_escola`, nao ha o que sinalizar. |

As treze restantes tem cobertura entre 83,59% e 100%.

## 8. Escolha do Algoritmo, Otimizacao e Metricas

### Otimizacao (Optuna)

`TPESampler` + `MedianPruner`, com `StratifiedGroupKFold` agrupado por aluno dentro do objective e storage persistente em SQLite. A busca roda em subamostra de 150.000 linhas (cada trial treina um modelo por fold; em 2,68M linhas isso inviabilizaria dezenas de trials localmente) -- o **treino final usa a base completa**.

| Modelo | ROC-AUC (CV) | Melhores parametros |
|---|---|---|
| Logistic | 0,6830 | `C=0.00195` |
| Random Forest | 0,6853 | `n_estimators=376, max_depth=11, min_samples_leaf=4, max_features=sqrt` |
| XGBoost | 0,6860 | `n_estimators=489, max_depth=6, learning_rate=0.0101, subsample=0.79, colsample_bytree=0.77, min_child_weight=4, reg_lambda=2.55` |

### Por que `max_depth` difere entre Random Forest (11) e XGBoost (6)

Nao e escolha manual: veio da busca Optuna (20 trials por modelo, espaco `max_depth` em [4,20] para RF e [3,10] para XGBoost). Reproduzivel por `python -m src.evaluation.depth_sensitivity`, que agrega o historico completo de `reports/optuna_study.db`:

| Random Forest -- `max_depth` | 4 | 6 | 7 | 9 | 10 | **11** | 12 | 14 | 15 | 17 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ROC-AUC medio | 0,6802 | 0,6825 | 0,6838 | 0,6850 | 0,6851 | **0,6853** | 0,6850 | 0,6839 | 0,6831 | 0,6800 | 0,6813 |

| XGBoost -- `max_depth` | 3 | 4 | 5 | **6** | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| ROC-AUC medio | 0,6846 | 0,6845 | 0,6852 | **0,6855** | 0,6602 | 0,6852 | 0,6752 | 0,6719 |

O Random Forest forma uma curva em U invertido com pico suave em 9-12: profundidades menores (4-7) *underfittam*, e a partir de 14 o desempenho cai (arvores individuais superajustadas, mesmo com a media de 376 arvores atenuando). Isso e esperado de bagging -- arvores decorrelacionadas, profundas, com a variancia controlada pela agregacao.

O XGBoost desaba a partir de `max_depth >= 9` (0,675 e 0,672, bem abaixo do pico de 0,6855). Isso e esperado de boosting sequencial: `learning_rate=0,0101` e 489 rounds significam que arvores profundas acumulam overfitting a cada round, ao inves de serem compensadas por media como no RF. `max_depth=8` (visto pelo professor no `DEFAULT_PARAMS` de fallback -- nota abaixo) ainda fica proximo do pico (0,6852), mas nao e o ponto tunado.

**Nota sobre o codigo:** `DEFAULT_PARAMS` em `train.py` (`max_depth=12` RF, `max_depth=8` XGBoost) e so o **fallback** usado quando `reports/optuna_best_params.json` nao existe -- nunca entrou em producao aqui, ja que o arquivo do Optuna esta presente. Os valores efetivamente treinados sao os desta secao (11 e 6), carregados por `load_best_params()`.

### Resultados finais (teste agrupado por aluno, 670.817 registros)

| Modelo | Accuracy | F1-Macro | Recall (Nao Alfab.) | Recall (Alfabetizado) | ROC-AUC |
|---|---|---|---|---|---|
| Logistic | 62,19% | 0,6191 | 65,59% | 59,84% | 0,6828 |
| **Random Forest (campeao)** | 62,54% | 0,6230 | **66,77%** | 59,63% | **0,6882** |
| XGBoost | 62,71% | 0,6242 | 66,03% | 60,41% | 0,6881 |

**Validacao da ausencia de vazamento:** o ROC-AUC de teste (0,6882) esta praticamente colado ao da validacao cruzada (0,6887) -- e de fato ligeiramente **abaixo** dela. Na versao com TargetEncoder, o teste ficava 2 pontos **acima** da CV: a assinatura do vazamento que foi corrigido.

O **Random Forest** foi escolhido como campeao por combinar o maior ROC-AUC com o maior recall da classe de risco (66,77%), que e a metrica operacionalmente relevante para Busca Ativa.

### A/B de `peso_aluno`: a variavel contribui de fato

O professor questionou se `peso_aluno` -- o peso amostral do SAEB, usado para ponderar estimativas populacionais -- faz sentido como feature preditiva de um aluno individual, ja que a rigor descreve o desenho amostral, nao uma causa de alfabetizacao. Reproduzivel por `python -m src.evaluation.ab_peso_aluno`: mesmo split agrupado por `id_aluno`, mesma semente e mesmos hiperparametros do Optuna nos dois bracos -- a unica diferenca e a presenca da variavel.

| Modelo | Com `peso_aluno` | Sem `peso_aluno` | Delta ROC-AUC |
|---|---|---|---|
| Logistic | 0,6828 | 0,6813 | +0,0015 |
| **Random Forest** | 0,6881 | 0,6860 | +0,0021 |
| XGBoost | 0,6880 | 0,6857 | +0,0024 |

**O ganho e real, ao contrario do enriquecimento com Censo Escolar abaixo.** Os deltas ficam na terceira casa decimal -- 30-40x maiores que o ruido de reamostragem observado nesse mesmo projeto (compare com o +0,00006 do Censo Escolar, secao seguinte) -- e se repetem nos tres algoritmos, na mesma direcao.

Controle de vazamento no braco sem `peso_aluno` (campeao Random Forest): ROC-AUC teste 0,6860 contra CV agrupada (3 folds) de 0,6866 -- teste **abaixo** da CV, sem suspeita de vazamento.

**Interpretacao.** O peso amostral do SAEB corrige o desbalanceamento entre estratos (regiao, rede, porte de escola) na hora de estimar quantidades populacionais -- na pratica, funciona como um proxy do estrato de origem do aluno, informacao que as demais 27 features nao capturam diretamente. Remove-lo custa desempenho de forma consistente; a variavel foi **mantida**.

### A/B do enriquecimento externo: resultado negativo

Reproduzivel por `python -m src.evaluation.ab_censo_enrichment`. Split, semente e hiperparametros identicos nos dois bracos -- a unica diferenca sao as 13 variaveis de Censo Escolar / Indicadores Educacionais.

| Modelo | Baseline (15 features) | Enriquecido (28 features) | Delta ROC-AUC |
|---|---|---|---|
| Logistic | 0,682765 | 0,682820 | +0,000055 |
| **Random Forest** | 0,688126 | 0,688183 | +0,000057 |
| XGBoost | 0,687942 | 0,688055 | +0,000113 |

**O ganho e nulo.** Os deltas estao na quinta casa decimal, ou seja, dentro do ruido. Treze variaveis novas, de fonte externa independente, no grao mais fino alcancavel, nao moveram o poder discriminativo do modelo.

O controle de vazamento passou limpo: no braco enriquecido, o ROC-AUC de teste (0,6882) ficou **abaixo** do de validacao cruzada (0,6887), diferenca de -0,0005. O zero e real, e nao um ganho falso mascarando um problema.

**Por que falhou, e por que o resultado importa.** A informacao municipal ja estava saturada: `taxa_alfabetizacao_municipio` e o desfecho observado daquela celula, e a infraestrutura, o corpo docente e a ruralidade sao **causas a montante desse mesmo desfecho**. Acrescentar as causas de um resultado que ja se mede diretamente nao acrescenta poder preditivo.

Esse experimento converte o teto estrutural de conjectura em evidencia. Antes, o argumento era que tres algoritmos param no mesmo lugar -- sugestivo, mas compativel com limitacao de modelo. Agora ha um teste direto: variaveis genuinamente novas, de outra fonte, nao movem nada. O teto e de **informacao**, nao de metodo. E o principal insumo da secao 12.

**Ressalva metodologica.** Os hiperparametros usados nos dois bracos vieram da busca Optuna feita sobre o conjunto **baseline** -- nao houve re-otimizacao para o conjunto enriquecido. A escolha e deliberada: manter os hiperparametros fixos e isolar o efeito das variaveis e o que torna o A/B interpretavel. Ela e conservadora no sentido de favorecer o baseline, entao um ganho real poderia estar sendo subestimado. Dada a magnitude observada (+0,00006, quarta casa abaixo do ruido de reamostragem), uma re-otimizacao nao alteraria a conclusao -- mas fica registrada como a evolucao natural caso o conjunto de variaveis seja ampliado no futuro.

**Decisao:** as 13 variaveis foram **mantidas** no modelo final. Nao custam performance -- accuracy e F1 do campeao inclusive sobem marginalmente (62,45% -> 62,58% e 0,6222 -> 0,6233) -- e o artefato entregue passa a incorporar de fato o Censo Escolar. O ganho preditivo nulo fica documentado aqui em vez de omitido.

## 9. Aplicacao Estrategica: as cinco perguntas de negocio

Executavel por `python -m src.evaluation.business_questions`.

### 9.1 Quais fatores mais impactam a alfabetizacao? / 9.5 Quais variaveis tem maior influencia nos modelos?

| Variavel | Contribuicao |
|---|---|
| `taxa_alfabetizacao_municipio` | 22,67% |
| `nivel_alfabetizacao` | 17,68% |
| `media_portugues_municipio` | 15,47% |
| `meta_2024` | 12,20% |
| `proporcao_adequado_avancado` | 5,68% |
| `proporcao_basico` | 4,52% |
| `proporcao_abaixo_basico` | 3,52% |
| UF (Ceara) | 2,37% |
| `peso_aluno` | 1,92% |
| **`ird_medio`** (regularidade docente) | **1,87%** |
| `inse_municipio` | 1,56% |

Agrupando: o **contexto educacional do municipio** (taxa, media de portugues e as tres proporcoes por nivel) responde por **51,9%**; as **metas oficiais** (`nivel_alfabetizacao`, `meta_2024`, participacao) por **31,0%**, confirmando que a trajetoria pactuada carrega sinal proprio; o **enriquecimento externo** por **6,96%**; e o INSE por apenas 1,56% -- o territorio importa mais que a renda isolada.

**Um alerta metodologico que este projeto tornou concreto.** As 13 variaveis do enriquecimento absorvem 6,96% da importancia -- `ird_medio` sozinha supera o INSE --, e ainda assim o A/B da secao 8 mostrou ganho de ROC-AUC de +0,00006. **Importancia nao e contribuicao preditiva.** Quando uma variavel e redundante com outra ja presente, o Random Forest distribui splits entre as duas e a importancia se reparte, sem que o poder discriminativo aumente. Ler a tabela acima como "regularidade docente explica 1,87% da alfabetizacao" seria exatamente o erro que a remocao de `gap_meta_2030` (secao 7.6) ja havia evitado uma vez.

![Feature Importance - Random Forest](images/feature_importance_random_forest.png)

Os graficos SHAP confirmam a direcao: maiores taxas municipais e maior nivel socioeconomico deslocam positivamente a probabilidade de alfabetizacao.

![SHAP Summary - Random Forest](images/shap_summary_random_forest.png)
![SHAP Waterfall - Random Forest](images/shap_waterfall_random_forest.png)

### 9.2 Quais municipios apresentam maior risco educacional?

Agregando o risco previsto (`1 - P(alfabetizado)`) por municipio, entre 4.177 municipios com pelo menos 20 alunos avaliados no conjunto de teste:

| UF | Municipios | Risco medio |
|---|---|---|
| SE | 61 | 0,736 |
| BA | 392 | 0,702 |
| RN | 107 | 0,679 |
| TO | 84 | 0,650 |
| AP | 16 | 0,624 |
| PA | 143 | 0,608 |

Ranking completo em `reports/q2_ranking_municipios_risco.csv`. A concentracao no Norte/Nordeste e consistente com a geografia educacional conhecida do pais.

![Municipios com maior risco educacional](images/q2_municipios_risco.png)

### 9.3 Quais regioes possuem padroes semelhantes?

KMeans (k=4) sobre risco previsto, taxa real, INSE, gap ate a meta e participacao:

| Perfil | Municipios | Risco medio | Taxa alfabetizacao | INSE | Gap ate meta |
|---|---|---|---|---|---|
| **Critico** | 849 | 0,704 | 36,1% | 4,41 | -43,0 pp |
| **Atencao** | 791 | 0,517 | 56,8% | 4,39 | -23,5 pp |
| **Intermediario** | 1.463 | 0,439 | 64,2% | 5,20 | -15,8 pp |
| **Consolidado** | 902 | 0,226 | 83,9% | 4,75 | +3,0 pp |

**Insight nao obvio:** o INSE **nao** separa os grupos de forma monotonica -- o cluster Critico (4,41) tem INSE ligeiramente **maior** que o cluster Atencao (4,39), e o Consolidado (4,75) tem INSE menor que o Intermediario (5,20). Ou seja, a diferenca entre municipios criticos e consolidados **nao e explicada primariamente por renda**, e sim por fatores territoriais e de gestao educacional. Isso e relevante para politica publica: transferencia de renda isolada nao fecharia a lacuna.

Detalhamento em `reports/q3_perfil_clusters.csv` e `q3_municipios_por_cluster.csv`.

![Agrupamento de municipios por padrao educacional](images/q3_clusters_regionais.png)

### 9.4 Como prever municipios que podem nao atingir metas futuras?

Projecao ate 2030 comparando a taxa atual com a meta oficial e o ritmo anual necessario:

| Classificacao | Municipios | % |
|---|---|---|
| Risco alto de nao atingir | **494** | 12,3% |
| Risco moderado | 1.509 | 37,6% |
| Provavel atingir | 1.349 | 33,6% |
| Meta ja atingida | 659 | 16,4% |

**494 municipios precisariam evoluir mais de 7 pontos percentuais ao ano** ate 2030 -- ritmo sem precedente historico na serie. Os casos mais extremos exigiriam mais de 12 pp/ano (ex.: municipios em RN, BA, SE e TO partindo de taxas abaixo de 10%).

Lista completa em `reports/q4_municipios_risco_meta.csv`.

![Projecao de atingimento da meta 2030](images/q4_projecao_metas.png)

## 10. Decisao de Negocio: Threshold Tuning para Busca Ativa

Em Busca Ativa, o custo de um **Falso Negativo** (nao identificar uma crianca que precisa de apoio) e maior que o de um Falso Positivo (incluir na triagem quem nao precisava).

| Threshold | Accuracy | Recall (risco) | Precision (risco) |
|---|---|---|---|
| 0,50 (padrao tecnico) | 62,54% | 66,77% | 53,32% |
| **0,55 (regra de negocio)** | 60,26% | **76,57%** | 50,90% |
| 0,60 | 56,11% | 86,97% | 47,95% |
| 0,65 | 53,24% | 91,76% | 46,35% |

Com threshold **0,55**, a cobertura de alunos em risco sobe de 66,8% para 76,6% (**+9,8 pp**), ao custo de ~2,3 pp de acuracia -- troca vantajosa quando o custo de reforco escolar extra e menor que o de deixar uma crianca sem apoio.

## 11. Interpretacao dos Resultados e Insights

1. **O territorio pesa mais que a renda.** INSE responde por apenas 1,56% da importancia, enquanto o conjunto de variaveis municipais/territoriais passa de 50%. A clusterizacao reforca: municipios criticos e consolidados tem INSE praticamente equivalente.
2. **As metas carregam sinal proprio** (~31% da importancia). Municipios com metas iniciais mais baixas (`meta_2024`) e menor `nivel_alfabetizacao` sao sistematicamente mais frageis -- a pactuacao ja refletia a fragilidade estrutural.
3. **12,3% dos municipios estao em rota de nao cumprir a meta 2030** no ritmo atual.
4. **O historico da propria escola contribui pouco isoladamente** (0,51%), mas melhorou o ROC-AUC geral -- efeito de interacao, nao de forca marginal.
5. **Metodologia importa mais que metrica.** As correcoes de vazamento reduziram o ROC-AUC aparente de 0,7032 para 0,6882; o numero menor e o unico que representa a capacidade real de generalizacao.
6. **Mais dados nao e o mesmo que mais informacao.** Treze variaveis de duas fontes externas, com boa cobertura e importancia agregada de 6,96%, nao moveram o ROC-AUC (secao 8). O gargalo do problema nao e a quantidade de variaveis -- e o grao em que elas existem.

## 12. Limitacoes do Projeto

1. **Grao das variaveis explicativas -- o teto estrutural, agora demonstrado.** O rotulo e individual, mas quase todo o contexto e agregado por municipio: dentro do mesmo municipio e rede, alunos recebem valores identicos. ROC-AUC ~0,69 e accuracy ~62% se repetem nos tres algoritmos e em todas as versoes do pipeline.

   O experimento da secao 8 fecha o argumento. Foram acrescentadas **13 variaveis de duas fontes externas independentes** (Censo Escolar e Indicadores Educacionais), no grao mais fino alcancavel, com cobertura de 83% a 100%. O ROC-AUC variou **+0,00006** -- ruido. Nao e que faltem variaveis: e que as variaveis disponiveis publicamente descrevem o **municipio**, e o municipio ja esta integralmente representado pela sua propria taxa de alfabetizacao observada. Infraestrutura, corpo docente e ruralidade sao causas a montante de um desfecho que o modelo ja enxerga diretamente.

   O teto e de **informacao**, nao de metodo. Sair dele exige dado no grao do aluno ou da escola -- bloqueado pela anonimizacao descrita no item 2.
2. **O grao de escola e inalcancavel: `id_escola` do SAEB esta anonimizado na origem.** Esta era a limitacao de maior potencial de ganho, e a investigacao fechou a porta em definitivo. Diagnostico por query direta contra o BigQuery publico da Base dos Dados:

   | Fonte | Escolas distintas | Faixa de `id_escola` |
   |---|---|---|
   | `silver.alunos_clean` (SAEB) | 42.802 | 60000001 - 60042811 |
   | `silver.inse_escola_clean` | 69.756 | 11000201 - 53068238 |
   | `basedosdados.br_inep_censo_escolar.escola` (2023) | 217.625 | 11000023 - 53086007 |

   Cruzamentos: `inse x censo` = **69.756 de 69.756 (100%)**; `saeb x censo` = **0**; `saeb x inse` = **0**.

   O identificador do SAEB percorre 60000001 a 60042811 -- um intervalo de 42.811 posicoes para 42.802 escolas distintas. E um **contador sequencial atribuido na anonimizacao**, nao o `CO_ENTIDADE` do INEP; o proprio prefixo `60` nao corresponde a nenhum codigo de UF (que vao de 11 a 53). O INSE, ao contrario, casa integralmente com o Censo Escolar, o que confirma que o namespace quebrado e o do SAEB e nao o do INSE, como se supunha antes.

   **Consequencia:** nenhuma tabela de-para externa resolve, porque o mapeamento nao foi publicado -- ele permanece com quem executou a anonimizacao. Qualquer enriquecimento no grao de escola esta descartado enquanto a base publica mantiver o identificador mascarado. Foi o que motivou a estrategia de descer ate **municipio x rede**, o grao mais fino ainda alcancavel (secao 4).

   Verificacao complementar: a tabela de origem `basedosdados.br_inep_avaliacao_alfabetizacao.alunos` tem **12 colunas, todas ja presentes no silver** -- nao ha sexo, raca/cor, idade, turno ou localizacao a recuperar upstream. O conjunto de variaveis individuais disponiveis publicamente esta esgotado.
3. **Correlacao ecologica.** Usar taxas municipais para prever desfecho individual implica o risco classico de inferencia ecologica: o modelo aprende "o municipio deste aluno historicamente vai bem/mal", nao causalidade sobre o aluno.
4. **Serie temporal curta.** Apenas duas edicoes (2023, 2024). A projecao de metas e deliberadamente linear e auditavel -- qualquer modelo temporal mais sofisticado seria sobreajuste com dois pontos.
5. **Uso restrito a suporte.** O modelo deve orientar alerta precoce e alocacao de recursos, **nunca** decisoes punitivas, ranqueamento sancionatorio ou discriminacao de escolas, alunos ou gestores.

## 13. Aplicacao Pratica para Politicas Publicas

- **Priorizacao da Busca Ativa:** com threshold 0,55, o modelo identifica ~77% dos alunos em risco, permitindo direcionar visitas domiciliares e reforco escolar.
- **Alocacao de recursos (FUNDEB):** os 849 municipios do cluster **Critico** e os 494 em risco alto de nao atingir a meta 2030 formam uma lista objetiva de prioridade.
- **Alerta antecipado de metas:** a projecao permite agir antes de 2030, nao apenas constatar o descumprimento depois.
- **Politica alem da renda:** a evidencia de que o INSE nao separa os clusters sugere que intervencoes pedagogicas e de gestao tendem a ser mais efetivas que transferencia de renda isolada para fechar a lacuna de alfabetizacao.

## 14. Possiveis Evolucoes Futuras

- **Solicitar ao INEP o de-para de `id_escola`.** Descartado como evolucao tecnica (secao 12.2): o mapeamento nao e publico. O caminho realista e institucional -- pedido formal de acesso a microdados identificados, via convenio de pesquisa. Com ele, INSE e infraestrutura passariam ao grao de escola, e essa continua sendo a maior alavanca isolada do projeto.
- **Questionarios contextuais do SAEB** (`TS_PROFESSOR`, `TS_DIRETOR`, `TS_ESCOLA`). No 2o ano nao ha questionario do aluno, mas o do professor traz formacao, experiencia e metodo de alfabetizacao adotado -- as unicas variaveis potencialmente acionaveis por politica publica em todo o desenho. Exige download e ingestao propria, fora do BigQuery.
- Enriquecer com fontes externas municipais: PNAD, Atlas do Desenvolvimento Humano, Cadastro Unico, FUNDEB. Ganho preditivo esperado baixo -- sao todas do grao de municipio, ja saturado pelo INSE -- mas ampliam a cobertura de fontes citadas no enunciado.
- **Setor censitario via geolocalizacao.** O Censo Escolar publica latitude/longitude das escolas; cruzando com os agregados do Censo Demografico 2022 por setor censitario, obtem-se contexto socioeconomico mais fino que o municipal sem depender do `id_escola`. E a unica rota identificada que contorna a anonimizacao.
- Rodar a busca Optuna na base completa e ampliar `OPTUNA_N_TRIALS` em ambiente com mais capacidade.
- Validacao `StratifiedGroupKFold` por `id_escola` para medir generalizacao a escolas nunca vistas.
- Com tres ou mais edicoes, modelo de trajetoria temporal por municipio substituindo a projecao linear.

## 15. Como Executar

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# Credenciais GCP (mesmo projeto da Fase 2) em credentials/service-account.json
export GOOGLE_APPLICATION_CREDENTIALS=credentials/service-account.json

python -m src.preprocessing.build_gold_ml                  # camada Gold ML
python -m src.visualization.eda_plots                      # EDA -> images/
python -m src.modeling.tune                                # Optuna (ou --model <nome>)
python -m src.modeling.train                               # treino + reports/model_results.csv
python -m src.evaluation.evaluate --model random_forest.joblib
python -m src.evaluation.explain  --model random_forest.joblib
python -m src.evaluation.business_questions --model random_forest.joblib
python -m src.modeling.predict                             # inferencia de exemplo
```

O notebook `notebooks/01_analise_exploratoria.ipynb` documenta a analise exploratoria e as hipoteses que orientaram a modelagem.