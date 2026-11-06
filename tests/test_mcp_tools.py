import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nba_insight import mcp_server


@pytest.fixture
def dataframe_teste():
    return pd.DataFrame([
        {
            "player": "LeBron James",
            "team": "LAL",
            "cluster": 3,
            "pts_per_game": 25.0
        },
        {
            "player": "Stephen Curry",
            "team": "GSW",
            "cluster": 3,
            "pts_per_game": 30.0
        }
    ])


# CONVERSÃO PARA JSON

def test_dataframe_to_records(dataframe_teste):
    resultado = mcp_server.dataframe_to_records(
        dataframe_teste
    )

    assert isinstance(resultado, list)
    assert len(resultado) == 2
    assert resultado[0]["player"] == "LeBron James"


# 1. BUSCAR JOGADOR

def test_buscar_jogador(monkeypatch, dataframe_teste):
    monkeypatch.setattr(
        mcp_server,
        "search_player",
        lambda nome: dataframe_teste.iloc[[0]]
    )

    resultado = mcp_server.buscar_jogador("LeBron James")

    assert len(resultado) == 1
    assert resultado[0]["player"] == "LeBron James"


# 2. COMPARAR JOGADORES

def test_comparar_jogadores(monkeypatch, dataframe_teste):
    monkeypatch.setattr(
        mcp_server,
        "compare_players",
        lambda nomes: dataframe_teste
    )

    resultado = mcp_server.comparar_jogadores([
        "LeBron James",
        "Stephen Curry"
    ])

    assert len(resultado) == 2
    assert resultado[1]["player"] == "Stephen Curry"


# 3. BUSCAR JOGADORES SEMELHANTES

def test_buscar_jogadores_similares(monkeypatch, dataframe_teste):
    def fake_find(name, limit):
        return dataframe_teste.head(limit)

    monkeypatch.setattr(
        mcp_server,
        "find_similar_players",
        fake_find
    )

    resultado = mcp_server.buscar_jogadores_similares(
        "LeBron James",
        limite=1
    )

    assert len(resultado) == 1
    assert resultado[0]["player"] == "LeBron James"


# 4. CONSULTAR CLUSTERS

def test_consultar_clusters(monkeypatch, dataframe_teste):
    monkeypatch.setattr(
        mcp_server,
        "get_cluster_summary",
        lambda cluster_id=None: dataframe_teste
    )

    resultado = mcp_server.consultar_clusters()

    assert len(resultado) == 2


def test_consultar_cluster_especifico(monkeypatch, dataframe_teste):
    chamadas = []

    def fake_summary(cluster_id=None):
        chamadas.append(cluster_id)
        return dataframe_teste.iloc[[0]]

    monkeypatch.setattr(
        mcp_server,
        "get_cluster_summary",
        fake_summary
    )

    resultado = mcp_server.consultar_clusters(3)

    assert chamadas == [3]
    assert len(resultado) == 1


# 5. BUSCAR JOGADORES POR FILTROS

def test_buscar_jogadores(monkeypatch, dataframe_teste):
    def fake_filter(filters=None, limit=50):
        return dataframe_teste.head(limit)

    monkeypatch.setattr(
        mcp_server,
        "filter_players",
        fake_filter
    )

    resultado = mcp_server.buscar_jogadores(
        filtros={"pts_per_game_min": 20},
        limite=1
    )

    assert len(resultado) == 1


# 6. ANALISAR PERFIL HIPOTÉTICO

def test_analisar_perfil_hipotetico(monkeypatch, dataframe_teste):
    def fake_analyze(stats, limit):
        return {
            "cluster": 2,
            "provided_features": list(stats.keys()),
            "similar_players": dataframe_teste.head(limit)
        }

    monkeypatch.setattr(
        mcp_server,
        "analyze_hypothetical_player",
        fake_analyze
    )

    resultado = mcp_server.analisar_perfil_hipotetico(
        estatisticas={"pts_per_game": 25},
        limite=1
    )

    assert resultado["cluster"] == 2
    assert resultado["features_informadas"] == ["pts_per_game"]
    assert len(resultado["jogadores_semelhantes"]) == 1


# 7. CONSULTAR MODELO

def test_consultar_modelo(monkeypatch):
    informacoes = {
        "season": 2026,
        "n_clusters": 4,
        "random_state": 42
    }

    monkeypatch.setattr(
        mcp_server,
        "get_model_info",
        lambda: informacoes
    )

    resultado = mcp_server.consultar_modelo()

    assert resultado == informacoes
