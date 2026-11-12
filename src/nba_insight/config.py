
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

# Temporada usada como ponto de partida na análise de evolução
SEASON_ANTERIOR = 2025

# Análise de evolução: critérios (ver analytics.get_evolution_table)
MIN_JOGOS_EVOLUCAO = 20      # mínimo de jogos em cada temporada
LIMIAR_TENDENCIA = 3.0       # variação de (pts + reb + ast) por jogo
LIMIAR_MINUTOS = 5.0         # variação de minutos por jogo considerada relevante

N_CLUSTERS = 4

RANDOM_STATE = 42


# Nomes legíveis dos clusters (usados apenas como rótulo de exibição).
# Os nomes são atribuídos automaticamente por analytics.get_cluster_names(),
# a partir do perfil médio de cada cluster; os textos abaixo são só os rótulos.
NOME_RESERVAS = "Reservas"
NOME_PROTAGONISTAS = "Protagonistas"
NOME_GARRAFAO = "Garrafão"
NOME_PERIMETRO = "Perímetro"

DESCRICAO_CLUSTERS = {
    NOME_RESERVAS: "poucos minutos por jogo e baixa produção",
    NOME_PROTAGONISTAS: "mais minutos, mais pontos e assistências",
    NOME_GARRAFAO: "minutos médios, com mais rebotes, tocos e arremessos eficientes perto da cesta",
    NOME_PERIMETRO: "minutos médios, perfil de quem joga mais longe da cesta",
}


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