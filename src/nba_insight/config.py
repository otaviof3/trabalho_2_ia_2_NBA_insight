
from pathlib import Path

# ============================================================
# CAMINHOS PRINCIPAIS
# ============================================================

# Raiz do projeto: pasta que contém data/, artifacts/, src/ etc.
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Dataset original
DATASET_PATH = DATA_DIR / "Player Per Game.csv"

# Modelo treinado do Trabalho 1
MODEL_PATH = ARTIFACTS_DIR / "kmeans_nba_2026.joblib"


# ============================================================
# CONFIGURAÇÕES DO MODELO
# ============================================================

SEASON = 2026

N_CLUSTERS = 4

RANDOM_STATE = 42


# ============================================================
# FEATURES UTILIZADAS NO K-MEANS
# ============================================================

FEATURES = [
    "mp_per_game",
    "fg_per_game",
    "fga_per_game",
    "fg_percent",
    "x3p_per_game",
    "x3pa_per_game",
    "x3p_percent",
    "ft_per_game",
    "fta_per_game",
    "ft_percent",
    "trb_per_game",
    "ast_per_game",
    "stl_per_game",
    "blk_per_game",
    "tov_per_game",
    "pts_per_game",
]


# ============================================================
# INICIALIZAÇÃO DOS DIRETÓRIOS
# ============================================================

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)