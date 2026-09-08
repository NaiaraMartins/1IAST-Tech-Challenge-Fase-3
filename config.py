# ============================================
# Google Cloud Platform
# ============================================

GCP_PROJECT_ID = "pipeline-alfabetizacao"

# ============================================
# BigQuery Datasets
# ============================================

BQ_DATASET_BRONZE = "bronze"
BQ_DATASET_SILVER = "silver"
BQ_DATASET_GOLD = "gold"

# ============================================
# Gold ML
# ============================================

# Tabela propria (nao sobrescreve gold.ml_features_alunos, usada pelo
# pipeline original de NaiaraMartins/1IAST-Tech-Challenge-Fase-3 no mesmo
# projeto GCP). Mesma logica de join, com id_aluno preservado.
#
# v3 acrescenta o enriquecimento com Censo Escolar e Indicadores Educacionais
# no grao municipio x rede (ver build_gold_ml.py). A v2 continua intacta no
# BigQuery e serve de baseline para o A/B -- basta apontar esta constante de
# volta para "ml_features_alunos_v2".
ML_FEATURES_TABLE = "ml_features_alunos_v3"
ML_FEATURES_TABLE_BASELINE = "ml_features_alunos_v2"

# ============================================
# Machine Learning
# ============================================

TARGET_COLUMN = "alfabetizado"
ID_COLUMN = "id_aluno"
GROUP_COLUMN = "id_aluno"

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# ============================================
# Optuna
# ============================================

OPTUNA_N_TRIALS = 20
OPTUNA_STORAGE = "sqlite:///reports/optuna_study.db"
OPTUNA_SAMPLE_SIZE = 150000  # subamostra p/ busca de hiperparametros (ver README)
OPTUNA_CV_SPLITS_SEARCH = 3

# ============================================
# Modelos
# ============================================

MODEL_PATH = "models/best_model.joblib"
