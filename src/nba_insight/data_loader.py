
import pandas as pd

from .config import DATASET_PATH, SEASON, FEATURES


def load_dataset(path=DATASET_PATH):
    """
    Carrega o dataset completo de estatísticas dos jogadores.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("O dataset está vazio.")

    return df


def validate_columns(df):
    """
    Verifica se o dataset possui todas as colunas necessárias.
    """

    required_columns = [
        "season",
        "lg",
        "player",
        "team",
        *FEATURES,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colunas ausentes no dataset: "
            + ", ".join(missing_columns)
        )


def prepare_season_data(df, season=SEASON):
    """
    Filtra a NBA e prepara os registros de uma temporada.

    Mantém a mesma lógica de tratamento utilizada
    no projeto original de clusterização.
    """

    validate_columns(df)

    # 1. Filtrar somente a NBA
    df_nba = df[
        df["lg"] == "NBA"
    ].copy()

    # 2. Selecionar a temporada
    df_season = df_nba[
        df_nba["season"] == season
    ].copy()

    if df_season.empty:
        raise ValueError(
            f"Nenhum registro encontrado para a temporada {season}."
        )

    # 3. Identificar jogadores com registro 2TM
    players_2tm = set(
        df_season.loc[
            df_season["team"] == "2TM",
            "player"
        ]
    )

    # 4. Remover registros de equipes individuais
    # quando existe um registro consolidado 2TM
    df_season = df_season[
        ~(
            df_season["player"].isin(players_2tm)
            &
            (df_season["team"] != "2TM")
        )
    ].copy()

    # 5. Remover duplicatas de jogador + temporada
    df_season = df_season.drop_duplicates(
        subset=["player", "season"],
        keep="first"
    ).copy()

    return df_season


def load_and_prepare_data(path=DATASET_PATH, season=SEASON):
    """
    Executa o carregamento e a preparação em uma única chamada.
    """

    df = load_dataset(path)

    df_season = prepare_season_data(
        df,
        season
    )

    return df_season
