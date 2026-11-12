import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import (
    FEATURES,
    LIMIAR_MINUTOS,
    LIMIAR_TENDENCIA,
    MIN_JOGOS_EVOLUCAO,
    SEASON,
    SEASON_ANTERIOR,
    NOME_GARRAFAO,
    NOME_PERIMETRO,
    NOME_PROTAGONISTAS,
    NOME_RESERVAS,
)
from .data_loader import load_and_prepare_data, load_dataset, prepare_season_data
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


# ============================================================
# EVOLUÇÃO DOS JOGADORES ENTRE TEMPORADAS
# ============================================================
#
# O modelo (imputer + scaler + K-Means) foi treinado na temporada 2026.
# Para comparar temporadas, aplicamos exatamente esse mesmo modelo às
# linhas de cada temporada: a "régua" é a mesma, então a mudança de
# cluster de um jogador indica mudança de perfil estatístico.

@lru_cache(maxsize=8)
def _load_season_with_clusters(season):
    """Prepara uma temporada e classifica seus jogadores no K-Means."""

    artifact = load_model_artifact()
    features = artifact["features"]

    df = prepare_season_data(load_dataset(), season).copy()

    X = artifact["scaler"].transform(
        artifact["imputer"].transform(df[features])
    )

    df["cluster"] = artifact["modelo"].predict(X)

    return df


def _classificar_tendencia(variacao):
    """Em alta / Em queda / Estável, pela variação de pts + reb + ast."""

    if variacao >= LIMIAR_TENDENCIA:
        return "Em alta"

    if variacao <= -LIMIAR_TENDENCIA:
        return "Em queda"

    return "Estável"


def _contexto_minutagem(variacao_minutos, tendencia):
    """
    Diz se a mudança veio junto com variação de minutos, para que o
    leitor não confunda "jogou mais" com "jogou melhor".
    """

    if variacao_minutos >= LIMIAR_MINUTOS:
        return f"Jogou mais ({variacao_minutos:+.1f} min por jogo)"

    if variacao_minutos <= -LIMIAR_MINUTOS:
        return f"Jogou menos ({variacao_minutos:+.1f} min por jogo)"

    if tendencia == "Estável":
        return "Sem mudança relevante"

    return "Minutagem parecida (mudança de rendimento ou de função)"


def get_evolution_table(
    season_from=SEASON_ANTERIOR,
    season_to=SEASON,
    min_games=MIN_JOGOS_EVOLUCAO,
):
    """
    Compara cada jogador entre duas temporadas.

    Só entram jogadores que aparecem nas duas temporadas e que tenham
    pelo menos `min_games` jogos em cada uma (amostras pequenas
    distorcem as médias).

    "Produção" = pontos + rebotes + assistências por jogo. A tendência
    é "Em alta" ou "Em queda" quando a produção varia pelo menos
    LIMIAR_TENDENCIA; senão, "Estável".
    """

    if season_from >= season_to:
        raise ValueError(
            "A temporada inicial deve ser anterior à final."
        )

    if min_games < 0:
        raise ValueError("O mínimo de jogos não pode ser negativo.")

    colunas = [
        "player_id", "player", "team", "pos", "age", "g", "cluster",
        "mp_per_game", "pts_per_game", "trb_per_game", "ast_per_game",
    ]

    antes = _load_season_with_clusters(season_from)[colunas]
    depois = _load_season_with_clusters(season_to)[colunas]

    m = antes.merge(depois, on="player_id", suffixes=("_ant", "_atu"))
    m = m[(m["g_ant"] >= min_games) & (m["g_atu"] >= min_games)].copy()

    nomes = get_cluster_names()

    out = pd.DataFrame({
        "player": m["player_atu"],
        "pos": m["pos_atu"],
        "age": m["age_atu"],
        "team_ant": m["team_ant"],
        "team_atu": m["team_atu"],
        "g_ant": m["g_ant"],
        "g_atu": m["g_atu"],
        "cluster_ant": m["cluster_ant"],
        "cluster_atu": m["cluster_atu"],
        "perfil_ant": m["cluster_ant"].map(nomes),
        "perfil_atu": m["cluster_atu"].map(nomes),
        "mudou_cluster": m["cluster_ant"] != m["cluster_atu"],
        "mp_ant": m["mp_per_game_ant"],
        "mp_atu": m["mp_per_game_atu"],
        "d_mp": m["mp_per_game_atu"] - m["mp_per_game_ant"],
    })

    for sigla in ["pts", "trb", "ast"]:
        out[f"{sigla}_ant"] = m[f"{sigla}_per_game_ant"]
        out[f"{sigla}_atu"] = m[f"{sigla}_per_game_atu"]
        out[f"d_{sigla}"] = out[f"{sigla}_atu"] - out[f"{sigla}_ant"]

    out["producao_ant"] = out["pts_ant"] + out["trb_ant"] + out["ast_ant"]
    out["producao_atu"] = out["pts_atu"] + out["trb_atu"] + out["ast_atu"]
    out["d_producao"] = out["producao_atu"] - out["producao_ant"]

    out["tendencia"] = out["d_producao"].map(_classificar_tendencia)
    out["contexto_minutagem"] = [
        _contexto_minutagem(d_mp, t)
        for d_mp, t in zip(out["d_mp"], out["tendencia"])
    ]

    return out.reset_index(drop=True).round(3)


def analyze_player_evolution(
    name,
    season_from=SEASON_ANTERIOR,
    season_to=SEASON,
):
    """
    Evolução de um jogador entre duas temporadas (uma linha).

    Não aplica o mínimo de jogos, para poder analisar qualquer
    jogador que esteja nas duas; a coluna `amostra_pequena` avisa
    quando algum dos lados tem menos jogos que o recomendado.
    """

    if not name or not name.strip():
        raise ValueError("Informe o nome de um jogador.")

    tabela = get_evolution_table(season_from, season_to, min_games=0)

    alvo = _normalizar(name)
    nomes = tabela["player"].map(_normalizar)

    encontrados = tabela[nomes == alvo]

    if encontrados.empty:
        encontrados = tabela[nomes.str.contains(alvo, regex=False, na=False)]

    if encontrados.empty:
        raise ValueError(
            f"Jogador não encontrado nas duas temporadas "
            f"({season_from} e {season_to}): {name}"
        )

    if len(encontrados) > 1:
        raise ValueError(_erro_ambiguo(name, encontrados))

    resultado = encontrados.copy()
    resultado["amostra_pequena"] = (
        (resultado["g_ant"] < MIN_JOGOS_EVOLUCAO)
        | (resultado["g_atu"] < MIN_JOGOS_EVOLUCAO)
    )

    return resultado.reset_index(drop=True)


def list_players_in_evolution(
    kind="alta",
    limit=10,
    season_from=SEASON_ANTERIOR,
    season_to=SEASON,
    min_games=MIN_JOGOS_EVOLUCAO,
):
    """
    Lista jogadores em destaque na evolução entre duas temporadas.

    kind:
      "alta"        maiores aumentos de produção (tendência "Em alta")
      "queda"       maiores quedas de produção (tendência "Em queda")
      "transicoes"  quem mudou de cluster (maiores variações primeiro)
    """

    if limit < 1:
        raise ValueError("O limite deve ser maior que zero.")

    tabela = get_evolution_table(season_from, season_to, min_games)

    if kind == "alta":
        lista = tabela[tabela["tendencia"] == "Em alta"].nlargest(
            limit, "d_producao"
        )

    elif kind == "queda":
        lista = tabela[tabela["tendencia"] == "Em queda"].nsmallest(
            limit, "d_producao"
        )

    elif kind == "transicoes":
        lista = tabela[tabela["mudou_cluster"]].copy()
        lista["_abs"] = lista["d_producao"].abs()
        lista = lista.nlargest(limit, "_abs").drop(columns="_abs")

    else:
        raise ValueError(
            "Tipo inválido. Use 'alta', 'queda' ou 'transicoes'."
        )

    return lista.reset_index(drop=True)


def get_cluster_transitions(
    season_from=SEASON_ANTERIOR,
    season_to=SEASON,
    min_games=MIN_JOGOS_EVOLUCAO,
):
    """
    Matriz de transições: linhas = perfil na temporada inicial,
    colunas = perfil na final, valores = número de jogadores.
    """

    tabela = get_evolution_table(season_from, season_to, min_games)

    ordem = [
        nome for _, nome in sorted(get_cluster_names().items())
    ]

    matriz = pd.crosstab(tabela["perfil_ant"], tabela["perfil_atu"])

    return matriz.reindex(index=ordem, columns=ordem, fill_value=0)
