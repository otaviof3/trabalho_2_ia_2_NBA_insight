import sys
import json
from pathlib import Path
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from mcp.server.fastmcp import FastMCP

# Configuração do caminho do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nba_insight.analytics import (
    search_player,
    compare_players,
    find_similar_players,
    get_cluster_summary,
    filter_players,
    analyze_hypothetical_player,
    get_model_info,
    get_players_data,
    add_cluster_names,
    get_cluster_names,
)


# Inicialização antecipada do dataset e do modelo
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    print(
        "NBA Insight: carregando dataset e modelo...",
        file=sys.stderr
    )

    get_players_data()

    print(
        "NBA Insight: dados carregados com sucesso!",
        file=sys.stderr
    )

    yield {}


mcp = FastMCP("NBA Insight", lifespan=app_lifespan)


# Conversão de DataFrame para registros JSON
def dataframe_to_records(df):
    json_data = df.to_json(
        orient="records",
        force_ascii=False
    )
    return json.loads(json_data)


# 1. Buscar jogador
@mcp.tool()
def buscar_jogador(nome: str) -> list[dict]:
    """
    Busca jogadores da NBA (temporada 2026) pelo nome.

    Ignora acentos e maiúsculas ("jokic" encontra "Nikola Jokić") e aceita
    nome parcial ("James" retorna todos os jogadores com esse nome).
    Retorna as estatísticas por jogo, o time, a posição e o cluster
    (grupo de perfil estatístico) de cada jogador encontrado.
    Lista vazia significa que ninguém foi encontrado.
    """
    resultado = add_cluster_names(search_player(nome))
    return dataframe_to_records(resultado)


# 2. Comparar jogadores
@mcp.tool()
def comparar_jogadores(nomes: list[str]) -> list[dict]:
    """
    Compara as estatísticas por jogo de dois ou mais jogadores.

    Recebe uma lista de nomes (ex.: ["Nikola Jokić", "Luka Dončić"]).
    Se um nome for ambíguo, o erro lista os candidatos; use então o
    nome completo. Retorna time, posição, idade, cluster e as 16
    estatísticas usadas pelo modelo.
    """
    resultado = add_cluster_names(compare_players(nomes))
    return dataframe_to_records(resultado)


# 3. Buscar jogadores semelhantes
@mcp.tool()
def buscar_jogadores_similares(
    nome: str,
    limite: int = 10
) -> list[dict]:
    """
    Encontra os jogadores com perfil estatístico mais parecido com o
    jogador informado.

    A semelhança é a distância euclidiana (campo "distance": quanto
    menor, mais parecido) no espaço padronizado das 16 estatísticas
    do modelo. "limite" define quantos jogadores retornar (padrão 10).
    """
    resultado = add_cluster_names(find_similar_players(nome, limit=limite))
    return dataframe_to_records(resultado)


# 4. Consultar clusters
@mcp.tool()
def consultar_clusters(cluster_id: int = -1) -> list[dict]:
    """
    Resume os 4 clusters (grupos de perfil) encontrados pelo K-Means.

    Sem argumento (ou -1) retorna todos; com cluster_id de 0 a 3
    retorna só aquele. Cada cluster traz o nome do perfil, a
    quantidade de jogadores e a média das 16 estatísticas por jogo.
    """
    if cluster_id == -1:
        resultado = get_cluster_summary()
    else:
        resultado = get_cluster_summary(cluster_id)

    resultado = add_cluster_names(resultado)

    return dataframe_to_records(resultado)


# 5. Buscar jogadores por filtros estatísticos
@mcp.tool()
def buscar_jogadores(
    filtros: dict[str, float],
    limite: int = 50
) -> list[dict]:
    """
    Lista jogadores que atendem a limites estatísticos.

    "filtros" é um dicionário com o nome da estatística + sufixo _min
    ou _max. Exemplo: {"pts_per_game_min": 25, "ast_per_game_min": 5}.
    Estatísticas válidas: mp_per_game, fg_per_game, fga_per_game,
    fg_percent, x3p_per_game, x3pa_per_game, x3p_percent, ft_per_game,
    fta_per_game, ft_percent, trb_per_game, ast_per_game, stl_per_game,
    blk_per_game, tov_per_game, pts_per_game. Percentuais vão de 0 a 1.
    """
    resultado = add_cluster_names(
        filter_players(
            filters=filtros,
            limit=limite
        )
    )

    return dataframe_to_records(resultado)


# 6. Analisar perfil hipotético
@mcp.tool()
def analisar_perfil_hipotetico(
    estatisticas: dict[str, float],
    limite: int = 5
) -> dict:
    """
    Classifica um jogador hipotético em um dos clusters e encontra
    jogadores reais parecidos.

    "estatisticas" usa os mesmos nomes de estatísticas de
    buscar_jogadores (ex.: {"pts_per_game": 25, "trb_per_game": 7,
    "ast_per_game": 5}). As não informadas são preenchidas com a
    mediana da liga, então quanto mais estatísticas, mais confiável.
    Percentuais vão de 0 a 1.
    """
    resultado = analyze_hypothetical_player(
        stats=estatisticas,
        limit=limite
    )

    return {
        "cluster": resultado["cluster"],
        "cluster_nome": get_cluster_names()[resultado["cluster"]],
        "features_informadas": resultado["provided_features"],
        "jogadores_semelhantes": dataframe_to_records(
            add_cluster_names(resultado["similar_players"])
        ),
    }


# 7. Consultar informações do modelo
@mcp.tool()
def consultar_modelo() -> dict:
    """
    Informações do modelo K-Means em uso: temporada (2026), número de
    clusters, semente aleatória, as 16 estatísticas usadas e o nome
    de cada cluster.
    """
    info = get_model_info()
    info["nomes_clusters"] = {
        str(k): v for k, v in get_cluster_names().items()
    }
    return info


if __name__ == "__main__":
    mcp.run()
