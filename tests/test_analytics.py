import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nba_insight import analytics
from src.nba_insight.config import FEATURES


class FakeModel:
    def predict(self, X):
        return np.array([2] * len(X))


@pytest.fixture
def dados_teste(monkeypatch):
    nomes = [
        "LeBron James",
        "Stephen Curry",
        "Kevin Durant",
        "Alex Smith",
        "Alex Johnson",
    ]

    df = pd.DataFrame({
        "player": nomes,
        "team": ["LAL", "GSW", "HOU", "BOS", "NYK"],
        "pos": ["SF", "PG", "SF", "SG", "PG"],
        "age": [41, 38, 38, 25, 24],
        "cluster": [3, 3, 1, 0, 2],
    })

    for i, feature in enumerate(FEATURES):
        df[feature] = [
            (j + 1) * 10 + i
            for j in range(len(nomes))
        ]

    # Valores específicos para os testes de filtros.
    df["pts_per_game"] = [25, 30, 10, 18, 22]
    df["ast_per_game"] = [5, 7, 2, 4, 6]

    X = df[FEATURES].copy()

    imputer = SimpleImputer(strategy="median")
    X_imputado = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputado)

    artifact = {
        "season": 2026,
        "n_clusters": 4,
        "features": list(FEATURES),
        "random_state": 42,
        "imputer": imputer,
        "scaler": scaler,
        "modelo": FakeModel(),
    }

    monkeypatch.setattr(
        analytics,
        "_load_analytics_data",
        lambda: (df.copy(), X_scaled, artifact)
    )

    return df, X_scaled, artifact


# CARREGAMENTO E INFORMAÇÕES DO MODELO

def test_get_players_data(dados_teste):
    df = analytics.get_players_data()

    assert len(df) == 5
    assert "cluster" in df.columns
    assert "player" in df.columns


def test_get_model_info(dados_teste):
    info = analytics.get_model_info()

    assert info["season"] == 2026
    assert info["n_clusters"] == 4
    assert info["random_state"] == 42
    assert info["features"] == list(FEATURES)


# BUSCA DE JOGADORES

def test_search_player_exact(dados_teste):
    resultado = analytics.search_player("LeBron James")

    assert len(resultado) == 1
    assert resultado.iloc[0]["player"] == "LeBron James"


def test_search_player_case_insensitive(dados_teste):
    resultado = analytics.search_player("lebron james")

    assert len(resultado) == 1


def test_search_player_partial(dados_teste):
    resultado = analytics.search_player("LeBron")

    assert len(resultado) == 1


def test_search_player_ambiguous(dados_teste):
    resultado = analytics.search_player("Alex")

    assert len(resultado) == 2


def test_search_player_empty_name(dados_teste):
    with pytest.raises(ValueError):
        analytics.search_player("   ")


# COMPARAÇÃO DE JOGADORES

def test_compare_players(dados_teste):
    resultado = analytics.compare_players([
        "LeBron James",
        "Stephen Curry"
    ])

    assert len(resultado) == 2
    assert "cluster" in resultado.columns
    assert all(feature in resultado.columns for feature in FEATURES)


def test_compare_players_less_than_two(dados_teste):
    with pytest.raises(ValueError):
        analytics.compare_players(["LeBron James"])


def test_compare_players_not_found(dados_teste):
    with pytest.raises(ValueError):
        analytics.compare_players([
            "LeBron James",
            "Jogador Inexistente"
        ])


def test_compare_players_ambiguous(dados_teste):
    with pytest.raises(ValueError):
        analytics.compare_players([
            "Alex",
            "LeBron James"
        ])


# JOGADORES SEMELHANTES

def test_find_similar_players(dados_teste):
    resultado = analytics.find_similar_players(
        "LeBron James",
        limit=2
    )

    assert len(resultado) == 2
    assert "distance" in resultado.columns
    assert "LeBron James" not in resultado["player"].values


def test_find_similar_players_invalid_limit(dados_teste):
    with pytest.raises(ValueError):
        analytics.find_similar_players("LeBron James", limit=0)


def test_find_similar_players_not_found(dados_teste):
    with pytest.raises(ValueError):
        analytics.find_similar_players("Jogador Inexistente")


# ANÁLISE DOS CLUSTERS

def test_get_cluster_summary_all(dados_teste):
    resultado = analytics.get_cluster_summary()

    assert len(resultado) == 4
    assert "cluster" in resultado.columns
    assert "jogadores" in resultado.columns


def test_get_cluster_summary_specific(dados_teste):
    resultado = analytics.get_cluster_summary(3)

    assert len(resultado) == 1
    assert resultado.iloc[0]["cluster"] == 3


def test_get_cluster_summary_invalid_cluster(dados_teste):
    with pytest.raises(ValueError):
        analytics.get_cluster_summary(4)


# FILTROS DE JOGADORES

def test_filter_players_minimum(dados_teste):
    resultado = analytics.filter_players({
        "pts_per_game_min": 20
    })

    assert all(resultado["pts_per_game"] >= 20)


def test_filter_players_maximum(dados_teste):
    resultado = analytics.filter_players({
        "pts_per_game_max": 20
    })

    assert all(resultado["pts_per_game"] <= 20)


def test_filter_players_multiple_filters(dados_teste):
    resultado = analytics.filter_players({
        "pts_per_game_min": 20,
        "ast_per_game_min": 5
    })

    assert all(resultado["pts_per_game"] >= 20)
    assert all(resultado["ast_per_game"] >= 5)


def test_filter_players_invalid_feature(dados_teste):
    with pytest.raises(ValueError):
        analytics.filter_players({
            "altura_min": 200
        })


def test_filter_players_invalid_filter(dados_teste):
    with pytest.raises(ValueError):
        analytics.filter_players({
            "pts_per_game": 20
        })


def test_filter_players_invalid_limit(dados_teste):
    with pytest.raises(ValueError):
        analytics.filter_players(limit=0)


# JOGADOR HIPOTÉTICO

def test_analyze_hypothetical_player(dados_teste):
    resultado = analytics.analyze_hypothetical_player({
        "pts_per_game": 25,
        "trb_per_game": 7,
        "ast_per_game": 5
    })

    assert resultado["cluster"] == 2
    assert len(resultado["similar_players"]) == 5
    assert "pts_per_game" in resultado["provided_features"]


def test_analyze_hypothetical_player_invalid_feature(dados_teste):
    with pytest.raises(ValueError):
        analytics.analyze_hypothetical_player({
            "altura": 200
        })


def test_analyze_hypothetical_player_empty(dados_teste):
    with pytest.raises(ValueError):
        analytics.analyze_hypothetical_player({})


def test_analyze_hypothetical_player_invalid_limit(dados_teste):
    with pytest.raises(ValueError):
        analytics.analyze_hypothetical_player(
            {"pts_per_game": 25},
            limit=0
        )
