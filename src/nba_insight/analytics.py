from functools import lru_cache

import numpy as np
import pandas as pd

from .config import FEATURES
from .data_loader import load_and_prepare_data
from .model_loader import load_model_artifact


# ============================================================
# CARREGAMENTO DOS DADOS E DO MODELO
# ============================================================

@lru_cache(maxsize=1)
def _load_analytics_data():
    df = load_and_prepare_data()
    artifact = load_model_artifact()

    features = artifact["features"]
    X = df[features].copy()

    X_imputado = artifact["imputer"].transform(X)
    X_scaled = artifact["scaler"].transform(X_imputado)

    clusters = artifact["modelo"].predict(X_scaled)

    df = df.copy()
    df["cluster"] = clusters

    return df, X_scaled, artifact


def get_players_data():
    """
    Retorna os dados dos jogadores com seus respectivos clusters.
    """

    df, _, _ = _load_analytics_data()

    return df.copy()


def get_model_info():
    """
    Retorna informações básicas sobre o modelo utilizado.
    """

    _, _, artifact = _load_analytics_data()

    return {
        "season": artifact["season"],
        "n_clusters": artifact["n_clusters"],
        "features": artifact["features"],
        "random_state": artifact["random_state"],
    }


# ============================================================
# BUSCA DE JOGADORES
# ============================================================

def search_player(name):
    """
    Procura jogadores pelo nome.

    Primeiro tenta encontrar uma correspondência exata.
    Caso não encontre, procura nomes que contenham o texto.
    """

    if not name or not name.strip():
        raise ValueError("Informe o nome de um jogador.")

    df = get_players_data()

    name = name.strip()

    exact = df[
        df["player"].str.casefold() == name.casefold()
    ]

    if not exact.empty:
        return exact.reset_index(drop=True)

    results = df[
        df["player"].str.contains(
            name,
            case=False,
            na=False,
            regex=False
        )
    ]

    return results.reset_index(drop=True)


# ============================================================
# COMPARAÇÃO DE JOGADORES
# ============================================================

def compare_players(names):
    """
    Compara dois ou mais jogadores.

    Recebe uma lista de nomes e retorna suas estatísticas.
    """

    if len(names) < 2:
        raise ValueError(
            "Informe pelo menos dois jogadores para comparar."
        )

    results = []

    for name in names:
        player = search_player(name)

        if player.empty:
            raise ValueError(
                f"Jogador não encontrado: {name}"
            )

        if len(player) > 1:
            raise ValueError(
                f"Mais de um resultado encontrado para '{name}'. "
                "Informe o nome completo."
            )

        results.append(player.iloc[0])

    comparison = pd.DataFrame(results)

    columns = [
        "player",
        "team",
        "pos",
        "age",
        "cluster",
        *FEATURES,
    ]

    return comparison[columns].reset_index(drop=True)


# ============================================================
# JOGADORES SEMELHANTES
# ============================================================


def find_similar_players(name, limit=10):
    """
    Encontra jogadores estatisticamente semelhantes.

    Utiliza distância euclidiana no espaço padronizado
    pelas mesmas transformações do modelo original.
    """

    if limit < 1:
        raise ValueError("O limite deve ser maior que zero.")

    df, X_scaled, artifact = _load_analytics_data()

    player = search_player(name)

    if player.empty:
        raise ValueError(
            f"Jogador não encontrado: {name}"
        )

    if len(player) > 1:
        raise ValueError(
            "Mais de um jogador encontrado. Informe o nome completo."
        )

    # Recupera a posição real do jogador no DataFrame original.
    player_name = player.iloc[0]["player"]

    player_positions = np.flatnonzero(
        df["player"].to_numpy() == player_name
    )

    if len(player_positions) != 1:
        raise ValueError(
            "Não foi possível identificar uma única linha para o jogador."
        )

    player_position = player_positions[0]

    # Vetor estatístico do jogador selecionado.
    player_vector = X_scaled[player_position]

    # Calcula a distância para todos os jogadores.
    distances = np.linalg.norm(
        X_scaled - player_vector,
        axis=1
    )

    # Exclui o próprio jogador.
    distances[player_position] = np.inf

    # Seleciona os jogadores com menores distâncias.
    nearest_indices = np.argsort(distances)[:limit]

    results = df.iloc[nearest_indices][
        ["player", "team", "pos", "cluster", *FEATURES]
    ].copy()

    results["distance"] = distances[nearest_indices]

    return results.reset_index(drop=True)



# ============================================================
# ANÁLISE DOS CLUSTERS
# ============================================================

def get_cluster_summary(cluster_id=None):
    """
    Retorna a quantidade de jogadores e médias estatísticas
    dos clusters.

    Se cluster_id for informado, retorna apenas aquele cluster.
    """

    df = get_players_data()

    if cluster_id is not None:
        if cluster_id not in range(4):
            raise ValueError(
                "Cluster inválido. Informe um valor entre 0 e 3."
            )

        df = df[df["cluster"] == cluster_id].copy()

        if df.empty:
            raise ValueError(
                f"Nenhum jogador encontrado no cluster {cluster_id}."
            )

    summary = (
        df.groupby("cluster")
        .agg(
            jogadores=("player", "count"),
            **{
                feature: (feature, "mean")
                for feature in FEATURES
            }
        )
        .reset_index()
    )

    return summary


# ============================================================
# FILTRO DE JOGADORES
# ============================================================

def filter_players(filters=None, limit=50):
    """
    Filtra jogadores usando limites mínimos e máximos.

    Exemplo:
    {
        "pts_per_game_min": 20,
        "trb_per_game_min": 5,
        "ast_per_game_min": 4
    }
    """

    df = get_players_data()

    if limit < 1:
        raise ValueError("O limite deve ser maior que zero.")

    if not filters:
        return df.head(limit).reset_index(drop=True)

    filtered = df.copy()

    for key, value in filters.items():

        if key.endswith("_min"):
            feature = key[:-4]

            if feature not in FEATURES:
                raise ValueError(
                    f"Feature inválida: {feature}"
                )

            filtered = filtered[
                filtered[feature] >= value
            ]

        elif key.endswith("_max"):
            feature = key[:-4]

            if feature not in FEATURES:
                raise ValueError(
                    f"Feature inválida: {feature}"
                )

            filtered = filtered[
                filtered[feature] <= value
            ]

        else:
            raise ValueError(
                f"Filtro inválido: {key}. "
                "Utilize os sufixos _min ou _max."
            )

    return filtered.head(limit).reset_index(drop=True)


# ============================================================
# ANÁLISE DE JOGADOR HIPOTÉTICO
# ============================================================

def analyze_hypothetical_player(stats, limit=5):
    """
    Classifica um jogador hipotético em um dos clusters
    e encontra jogadores reais com estatísticas semelhantes.

    stats deve ser um dicionário com as features do modelo.

    Exemplo:
    {
        "pts_per_game": 25,
        "trb_per_game": 7,
        "ast_per_game": 5
    }

    Features não informadas serão preenchidas pelo imputer.
    """

    if not stats:
        raise ValueError(
            "Informe pelo menos uma estatística."
        )

    if limit < 1:
        raise ValueError("O limite deve ser maior que zero.")

    _, X_scaled, artifact = _load_analytics_data()

    features = artifact["features"]

    # Verificar se as estatísticas informadas são válidas
    invalid_features = [
        feature
        for feature in stats
        if feature not in features
    ]

    if invalid_features:
        raise ValueError(
            "Features inválidas: "
            + ", ".join(invalid_features)
        )

    # Criar uma linha com todas as features
    hypothetical = {
        feature: np.nan
        for feature in features
    }

    hypothetical.update(stats)

    X_hypothetical = pd.DataFrame(
        [hypothetical],
        columns=features
    )

    # Aplicar o mesmo pré-processamento do treinamento
    X_imputado = artifact["imputer"].transform(
        X_hypothetical
    )

    X_transformed = artifact["scaler"].transform(
        X_imputado
    )

    # Classificar no K-Means existente
    model = artifact["modelo"]

    cluster = int(
        model.predict(X_transformed)[0]
    )

    # Calcular jogadores reais mais próximos
    distances = np.linalg.norm(
        X_scaled - X_transformed[0],
        axis=1
    )

    nearest_indices = np.argsort(distances)[:limit]

    df = get_players_data()

    similar_players = df.iloc[nearest_indices][
        ["player", "team", "pos", "cluster", *FEATURES]
    ].copy()

    similar_players["distance"] = distances[nearest_indices]

    return {
        "cluster": cluster,
        "similar_players": similar_players.reset_index(drop=True),
        "provided_features": list(stats.keys()),
    }
