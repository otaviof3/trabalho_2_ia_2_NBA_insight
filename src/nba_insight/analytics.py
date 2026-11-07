import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import (
    FEATURES,
    NOME_GARRAFAO,
    NOME_PERIMETRO,
    NOME_PROTAGONISTAS,
    NOME_RESERVAS,
)
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

def _normalizar(texto):
    """
    Remove acentos e padroniza caixa: "Jokić" e "jokic" viram "jokic".

    Caracteres como "ł" ou "ø" não se decompõem em NFKD; por isso
    ficam como estão (o que é raro entre os jogadores da NBA).
    """

    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c))

    return texto.casefold().strip()


def search_player(name):
    """
    Procura jogadores pelo nome, ignorando acentos e maiúsculas.

    Primeiro tenta encontrar uma correspondência exata.
    Caso não encontre, procura nomes que contenham o texto.
    """

    if not name or not name.strip():
        raise ValueError("Informe o nome de um jogador.")

    df = get_players_data()

    alvo = _normalizar(name)
    nomes = df["player"].map(_normalizar)

    exact = df[nomes == alvo]

    if not exact.empty:
        return exact.reset_index(drop=True)

    results = df[
        nomes.str.contains(alvo, regex=False, na=False)
    ]

    return results.reset_index(drop=True)


def _erro_ambiguo(name, encontrados):
    """
    Monta a mensagem de erro para nomes com mais de um resultado,
    já listando os candidatos para o usuário escolher.
    """

    nomes = list(encontrados["player"].head(8))

    return (
        f"Mais de um resultado encontrado para '{name}': "
        + ", ".join(nomes)
        + ". Informe o nome completo."
    )


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
            raise ValueError(_erro_ambiguo(name, player))

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
        raise ValueError(_erro_ambiguo(name, player))

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

    n_clusters = get_model_info()["n_clusters"]

    if cluster_id is not None:
        if cluster_id not in range(n_clusters):
            raise ValueError(
                "Cluster inválido. Informe um valor entre 0 e "
                f"{n_clusters - 1}."
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
# NOMES DOS CLUSTERS
# ============================================================

def get_cluster_names():
    """
    Dá um nome legível a cada cluster, a partir do perfil médio.

    Regras (aplicadas ao K=4 do Trabalho 1):
    - menos minutos por jogo           -> reservas
    - mais minutos por jogo            -> protagonistas
    - entre os dois do meio, o que tem mais rebotes + tocos
      -> jogadores de garrafão; o outro -> jogadores de perímetro

    Se o modelo tiver outro número de clusters, usa "Cluster N".
    """

    df = get_players_data()
    medias = df.groupby("cluster")[
        ["mp_per_game", "trb_per_game", "blk_per_game"]
    ].mean()

    if len(medias) != 4:
        return {int(c): f"Cluster {int(c)}" for c in medias.index}

    por_minutos = medias["mp_per_game"].sort_values()
    reservas = por_minutos.index[0]
    protagonistas = por_minutos.index[-1]
    meio = list(por_minutos.index[1:3])

    garrafao_score = medias["trb_per_game"] + 2 * medias["blk_per_game"]
    garrafao = max(meio, key=lambda c: garrafao_score[c])
    perimetro = [c for c in meio if c != garrafao][0]

    return {
        int(reservas): NOME_RESERVAS,
        int(protagonistas): NOME_PROTAGONISTAS,
        int(garrafao): NOME_GARRAFAO,
        int(perimetro): NOME_PERIMETRO,
    }


def cluster_label(cluster_id):
    """Retorna 'Nome do cluster' para exibição."""

    return get_cluster_names().get(int(cluster_id), f"Cluster {cluster_id}")


def add_cluster_names(df):
    """
    Devolve uma cópia do DataFrame com a coluna 'cluster_nome'
    logo após a coluna 'cluster' (se ela existir).
    """

    df = df.copy()

    if "cluster" not in df.columns:
        return df

    nomes = get_cluster_names()
    posicao = list(df.columns).index("cluster") + 1

    df.insert(
        posicao,
        "cluster_nome",
        df["cluster"].map(lambda c: nomes.get(int(c), f"Cluster {c}")),
    )

    return df


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
