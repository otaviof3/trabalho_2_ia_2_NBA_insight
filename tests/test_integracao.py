"""
Testes de integração: usam o dataset e o modelo REAIS do repositório
(sem mocks), para garantir que tudo funciona de ponta a ponta.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nba_insight import analytics, mcp_server
from src.nba_insight.config import FEATURES
from src.nba_insight.data_loader import prepare_season_data


def test_dados_reais_2026():
    df = analytics.get_players_data()

    assert len(df) == 582
    assert df["player"].is_unique
    assert sorted(df["cluster"].unique()) == [0, 1, 2, 3]


def test_busca_ignora_acentos():
    assert analytics.search_player("Jokic").iloc[0]["player"] == "Nikola Jokić"
    assert analytics.search_player("doncic").iloc[0]["player"] == "Luka Dončić"


def test_nome_ambiguo_lista_candidatos():
    with pytest.raises(ValueError, match="Mais de um resultado.*James"):
        analytics.compare_players(["James", "Stephen Curry"])


def test_nomes_dos_clusters_sao_unicos_e_completos():
    nomes = analytics.get_cluster_names()

    assert sorted(nomes) == [0, 1, 2, 3]
    assert len(set(nomes.values())) == 4
    assert set(nomes.values()) == {
        "Reservas", "Protagonistas", "Garrafão", "Perímetro"
    }


def test_nomes_batem_com_o_perfil_medio():
    resumo = analytics.get_cluster_summary().set_index("cluster")
    nomes = analytics.get_cluster_names()
    por_nome = {nome: c for c, nome in nomes.items()}

    assert resumo["mp_per_game"].idxmin() == por_nome["Reservas"]
    assert resumo["mp_per_game"].idxmax() == por_nome["Protagonistas"]


def test_add_cluster_names_coloca_coluna_ao_lado():
    df = analytics.add_cluster_names(analytics.search_player("LeBron James"))
    colunas = list(df.columns)

    assert colunas[colunas.index("cluster") + 1] == "cluster_nome"


def test_similares_com_dados_reais():
    r = analytics.find_similar_players("Brandon Ingram", limit=3)

    assert len(r) == 3
    assert "Brandon Ingram" not in r["player"].values
    assert (r["distance"] >= 0).all()


def test_hipotetico_aceita_nan_em_percentual():
    stats = {f: float(analytics.get_players_data()[f].median()) for f in FEATURES}
    stats["x3pa_per_game"] = 0.0
    stats["x3p_per_game"] = 0.0
    stats["x3p_percent"] = np.nan

    r = analytics.analyze_hypothetical_player(stats, limit=3)

    assert r["cluster"] in range(4)
    assert len(r["similar_players"]) == 3


def test_mcp_retorna_nome_do_cluster():
    r = mcp_server.buscar_jogador("Jokic")

    assert r[0]["player"] == "Nikola Jokić"
    assert r[0]["cluster_nome"] in {"Reservas", "Protagonistas", "Garrafão", "Perímetro"}


def test_mcp_ferramentas_tem_descricao():
    for nome in [
        "buscar_jogador", "comparar_jogadores", "buscar_jogadores_similares",
        "consultar_clusters", "buscar_jogadores",
        "analisar_perfil_hipotetico", "consultar_modelo",
    ]:
        assert getattr(mcp_server, nome).__doc__, f"{nome} sem docstring"


def test_json_do_mcp_nao_tem_nan():
    r = mcp_server.buscar_jogadores({"pts_per_game_min": 0}, limite=600)

    assert len(r) == 582


def test_multi_time_3tm_4tm_mantem_so_o_consolidado():
    linhas = []
    base = {f: 1.0 for f in FEATURES}
    for team in ["TOT3", "AAA", "BBB", "CCC", "3TM"]:
        linhas.append(dict(base, season=2026, lg="NBA", player="Fulano", team=team))
    linhas.append(dict(base, season=2026, lg="NBA", player="Ciclano", team="AAA"))

    out = prepare_season_data(pd.DataFrame(linhas))

    assert sorted(out["player"]) == ["Ciclano", "Fulano"]
    assert out.loc[out["player"] == "Fulano", "team"].iloc[0] == "3TM"
