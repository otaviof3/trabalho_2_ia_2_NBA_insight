
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Permite importar os módulos do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.nba_insight.config import (
    ARTIFACTS_DIR,
    MODEL_PATH,
    FEATURES,
    SEASON,
    N_CLUSTERS,
    RANDOM_STATE,
)

from src.nba_insight.data_loader import load_and_prepare_data


def train_and_save_model():
    """
    Reproduz o pipeline do Trabalho 1 e salva
    os componentes necessários para o NBA Insight.
    """

    print("=" * 50)
    print("NBA INSIGHT - EXPORTAÇÃO DO MODELO")
    print("=" * 50)

    # 1. Carregar os dados preparados
    print("\n[1/5] Carregando dados...")

    df = load_and_prepare_data()

    print(f"Jogadores encontrados: {len(df)}")

    # 2. Selecionar as features
    print("\n[2/5] Selecionando features...")

    X = df[FEATURES].copy()

    print(f"Quantidade de features: {len(FEATURES)}")

    # 3. Tratar valores ausentes
    print("\n[3/5] Aplicando imputação...")

    imputer = SimpleImputer(strategy="median")

    X_imputado = imputer.fit_transform(X)

    # 4. Padronizar os dados
    print("\n[4/5] Aplicando padronização...")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_imputado)

    # 5. Treinar o K-Means original
    print("\n[5/5] Treinando K-Means...")

    modelo = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    clusters = modelo.fit_predict(X_scaled)

    # Criar o artefato com todos os componentes
    artifact = {
        "modelo": modelo,
        "imputer": imputer,
        "scaler": scaler,
        "features": FEATURES,
        "season": SEASON,
        "n_clusters": N_CLUSTERS,
        "random_state": RANDOM_STATE,
    }

    # Garantir que a pasta de destino exista
    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Salvar o artefato
    joblib.dump(artifact, MODEL_PATH)

    # Exibir resumo
    df_resultado = df[["player", "team"]].copy()
    df_resultado["cluster"] = clusters

    print("\n" + "=" * 50)
    print("MODELO SALVO COM SUCESSO!")
    print("=" * 50)

    print(f"\nArquivo: {MODEL_PATH}")
    print(f"Temporada: {SEASON}")
    print(f"Jogadores: {len(df)}")
    print(f"Clusters: {N_CLUSTERS}")
    print(f"Features: {len(FEATURES)}")

    print("\nQuantidade de jogadores por cluster:")

    contagem = (
        df_resultado["cluster"]
        .value_counts()
        .sort_index()
    )

    for cluster, quantidade in contagem.items():
        print(f"Cluster {cluster}: {quantidade} jogadores")

    print("\nArtefato pronto para utilização pelo NBA Insight.")


if __name__ == "__main__":
    train_and_save_model()
