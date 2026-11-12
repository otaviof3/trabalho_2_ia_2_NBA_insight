import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nba_insight import analytics, mcp_server
from src.nba_insight.config import FEATURES


# ---------- regras puras (sem dados)

@pytest.mark.parametrize("variacao,esperado", [
    (5.0, "Em alta"),
    (3.0, "Em alta"),
    (2.9, "Estável"),
    (0.0, "Estável"),
    (-2.9, "Estável"),
    (-3.0, "Em queda"),
    (-10.0, "Em queda"),
])
def test_classificar_tendencia(variacao, esperado):
    assert analytics._classificar_tendencia(variacao) == esperado


def test_contexto_minutagem():
    assert "Jogou mais" in analytics._contexto_minutagem(6, "Em alta")
    assert "Jogou menos" in analytics._contexto_minutagem(-6, "Em queda")
    assert "parecida" in analytics._contexto_minutagem(1, "Em alta")
    assert "Sem mudança" in analytics._contexto_minutagem(1, "Estável")


def test_temporadas_invertidas():
    with pytest.raises(ValueError):
        analytics.get_evolution_table(2026, 2025)


def test_limite_invalido():
    with pytest.raises(ValueError):
        analytics.list_players_in_evolution("alta", limit=0)


# ---------- dados simulados (duas temporadas pequenas)

@pytest.fixture
def temporadas_falsas(monkeypatch):
    def linha(pid, nome, g, mp, pts, trb, ast, cluster):
        d = {f: 1.0 for f in FEATURES}
        d.update(player_id=pid, player=nome, team="AAA", pos="SG", age=25,
                 g=g, cluster=cluster, mp_per_game=mp, pts_per_game=pts,
                 trb_per_game=trb, ast_per_game=ast)
        return d

    antes = pd.DataFrame([
        linha("a", "Sobe Muito", 60, 20, 8, 3, 2, 0),
        linha("b", "Cai Muito", 60, 32, 20, 6, 5, 3),
        linha("c", "Fica Igual", 60, 25, 10, 4, 3, 1),
        linha("d", "Poucos Jogos", 5, 20, 8, 3, 2, 0),
        linha("e", "So Em 2025", 60, 20, 8, 3, 2, 0),
    ])
    depois = pd.DataFrame([
        linha("a", "Sobe Muito", 60, 30, 18, 6, 5, 3),
        linha("b", "Cai Muito", 60, 20, 10, 3, 2, 1),
        linha("c", "Fica Igual", 60, 25, 10.5, 4, 3, 1),
        linha("d", "Poucos Jogos", 8, 30, 20, 8, 6, 3),
        linha("f", "Calouro", 60, 20, 8, 3, 2, 0),
    ])

    por_temporada = {2025: antes, 2026: depois}

    monkeypatch.setattr(
        analytics, "_load_season_with_clusters",
        lambda season: por_temporada[season].copy(),
    )
    monkeypatch.setattr(
        analytics, "get_cluster_names",
        lambda: {0: "Reservas", 1: "Perímetro", 2: "Garrafão", 3: "Protagonistas"},
    )


def test_tabela_filtra_minimo_de_jogos_e_quem_nao_esta_nas_duas(temporadas_falsas):
    t = analytics.get_evolution_table()

    assert sorted(t["player"]) == ["Cai Muito", "Fica Igual", "Sobe Muito"]


def test_tabela_tendencias_e_mudanca_de_cluster(temporadas_falsas):
    t = analytics.get_evolution_table().set_index("player")

    assert t.loc["Sobe Muito", "tendencia"] == "Em alta"
    assert t.loc["Cai Muito", "tendencia"] == "Em queda"
    assert t.loc["Fica Igual", "tendencia"] == "Estável"
    assert t.loc["Sobe Muito", "perfil_ant"] == "Reservas"
    assert t.loc["Sobe Muito", "perfil_atu"] == "Protagonistas"
    assert bool(t.loc["Sobe Muito", "mudou_cluster"]) is True
    assert bool(t.loc["Fica Igual", "mudou_cluster"]) is False
    assert t.loc["Sobe Muito", "d_producao"] == pytest.approx(16.0)
    assert "Jogou mais" in t.loc["Sobe Muito", "contexto_minutagem"]


def test_analisar_jogador_e_amostra_pequena(temporadas_falsas):
    r = analytics.analyze_player_evolution("poucos jogos")

    assert len(r) == 1
    assert bool(r.iloc[0]["amostra_pequena"]) is True


def test_analisar_jogador_so_em_uma_temporada(temporadas_falsas):
    with pytest.raises(ValueError, match="nas duas temporadas"):
        analytics.analyze_player_evolution("Calouro")


def test_listas_e_matriz(temporadas_falsas):
    alta = analytics.list_players_in_evolution("alta")
    queda = analytics.list_players_in_evolution("queda")
    trans = analytics.list_players_in_evolution("transicoes")
    matriz = analytics.get_cluster_transitions()

    assert list(alta["player"]) == ["Sobe Muito"]
    assert list(queda["player"]) == ["Cai Muito"]
    assert set(trans["player"]) == {"Sobe Muito", "Cai Muito"}
    assert matriz.loc["Reservas", "Protagonistas"] == 1
    assert matriz.loc["Protagonistas", "Perímetro"] == 1
    assert matriz.values.sum() == 3

    with pytest.raises(ValueError):
        analytics.list_players_in_evolution("qualquer")


# ---------- dados e modelo reais

def test_real_evolucao_2025_2026():
    t = analytics.get_evolution_table()

    assert 300 < len(t) < 460
    assert set(t["tendencia"]) == {"Em alta", "Em queda", "Estável"}
    assert t["mudou_cluster"].sum() > 50
    assert not t[["producao_ant", "producao_atu"]].isna().any().any()


def test_real_jogador_conhecido_sobe():
    r = analytics.analyze_player_evolution("Nickeil Alexander-Walker").iloc[0]

    assert r["tendencia"] == "Em alta"
    assert r["perfil_atu"] == "Protagonistas"
    assert r["d_producao"] > 3


def test_real_busca_sem_acento_na_evolucao():
    r = analytics.analyze_player_evolution("Jokic")

    assert r.iloc[0]["player"] == "Nikola Jokić"


def test_real_mcp_evolucao():
    r = mcp_server.analisar_evolucao_jogador("Cam Thomas")

    assert r["tendencia"] == "Em queda"
    assert "contexto_minutagem" in r

    lista = mcp_server.listar_jogadores_em_evolucao("queda", 3)

    assert len(lista["jogadores"]) == 3
    assert "Reservas" in lista["transicoes_entre_clusters"]


def test_mcp_evolucao_tem_descricao():
    assert mcp_server.analisar_evolucao_jogador.__doc__
    assert mcp_server.listar_jogadores_em_evolucao.__doc__
