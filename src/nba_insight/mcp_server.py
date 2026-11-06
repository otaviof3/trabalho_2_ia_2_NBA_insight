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
    resultado = search_player(nome)
    return dataframe_to_records(resultado)


# 2. Comparar jogadores
@mcp.tool()
def comparar_jogadores(nomes: list[str]) -> list[dict]:
    resultado = compare_players(nomes)
    return dataframe_to_records(resultado)


# 3. Buscar jogadores semelhantes
@mcp.tool()
def buscar_jogadores_similares(
    nome: str,
    limite: int = 10
) -> list[dict]:
    resultado = find_similar_players(nome, limit=limite)
    return dataframe_to_records(resultado)


# 4. Consultar clusters
@mcp.tool()
def consultar_clusters(cluster_id: int = -1) -> list[dict]:
    if cluster_id == -1:
        resultado = get_cluster_summary()
    else:
        resultado = get_cluster_summary(cluster_id)

    return dataframe_to_records(resultado)


# 5. Buscar jogadores por filtros estatísticos
@mcp.tool()
def buscar_jogadores(
    filtros: dict[str, float],
    limite: int = 50
) -> list[dict]:
    resultado = filter_players(
        filters=filtros,
        limit=limite
    )

    return dataframe_to_records(resultado)


# 6. Analisar perfil hipotético
@mcp.tool()
def analisar_perfil_hipotetico(
    estatisticas: dict[str, float],
    limite: int = 5
) -> dict:
    resultado = analyze_hypothetical_player(
        stats=estatisticas,
        limit=limite
    )

    return {
        "cluster": resultado["cluster"],
        "features_informadas": resultado["provided_features"],
        "jogadores_semelhantes": dataframe_to_records(
            resultado["similar_players"]
        ),
    }


# 7. Consultar informações do modelo
@mcp.tool()
def consultar_modelo() -> dict:
    return get_model_info()


if __name__ == "__main__":
    mcp.run()
