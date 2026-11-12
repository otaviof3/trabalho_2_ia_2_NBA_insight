import sys
from pathlib import Path

import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nba_insight.analytics import (
    get_players_data,
    get_model_info,
    search_player,
    compare_players,
    find_similar_players,
    get_cluster_summary,
    filter_players,
    analyze_hypothetical_player,
    add_cluster_names,
    cluster_label,
    get_cluster_names,
    get_evolution_table,
    analyze_player_evolution,
    list_players_in_evolution,
    get_cluster_transitions,
)
from src.nba_insight.config import DESCRICAO_CLUSTERS

# --------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# --------------------------------------------------

st.set_page_config(
    page_title="NBA Insight",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# IDENTIDADE VISUAL
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #10141c;
        color: #f1f3f5;
    }

    section[data-testid="stSidebar"] {
        background-color: #171e29;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #f97316;
        margin-bottom: 0;
    }

    .subtitle {
        color: #aab4c3;
        font-size: 16px;
        margin-bottom: 25px;
    }

    div[data-testid="stMetric"] {
        background-color: #1c2635;
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #303c4d;
    }

    .section-title {
        color: #f97316;
        font-size: 23px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# CARREGAMENTO DOS DADOS
# --------------------------------------------------

@st.cache_data
def carregar_dados():
    return get_players_data()


@st.cache_data
def carregar_modelo():
    return get_model_info()


df = carregar_dados()
modelo = carregar_modelo()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.markdown("# 🏀 NBA INSIGHT")
st.sidebar.caption("Análise inteligente de jogadores da NBA")

pagina = st.sidebar.radio(
    "NAVEGAÇÃO",
    [
    "Dashboard",
    "Jogadores",
    "Jogadores similares",
    "Comparações",
    "Clusters",
    "Evolução",
    "Perfil hipotético",
],
)

st.sidebar.divider()

st.sidebar.markdown("### Modelo")
st.sidebar.write(f"Temporada: {modelo['season']}")
st.sidebar.write(f"Clusters: {modelo['n_clusters']}")
st.sidebar.write(f"Variáveis: {len(modelo['features'])}")

st.sidebar.caption("NBA Insight — Trabalho 2 | IA II")


# --------------------------------------------------
# CABEÇALHO
# --------------------------------------------------

st.markdown(
    '<p class="main-title">🏀 NBA INSIGHT</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="subtitle">Inteligência e análise estatística do basquete profissional</p>',
    unsafe_allow_html=True,
)


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if pagina == "Dashboard":

    st.markdown("## Visão geral da temporada")

    total_jogadores = len(df)
    total_equipes = df.loc[
    df["team"].notna()
    & ~df["team"].astype("string").str.fullmatch(r"\d+TM", na=False),
    "team"
].nunique()
    total_clusters = df["cluster"].nunique()
    media_pontos = df["pts_per_game"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Jogadores", total_jogadores)
    col2.metric("Equipes", total_equipes)
    col3.metric("Clusters", total_clusters)
    col4.metric("Média de pontos", f"{media_pontos:.1f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Distribuição dos jogadores por cluster")

        distribuicao = (
            add_cluster_names(df)["cluster_nome"]
            .value_counts()
            .reset_index()
        )

        distribuicao.columns = ["Cluster", "Jogadores"]

        fig = px.bar(
            distribuicao,
            x="Cluster",
            y="Jogadores",
            color="Cluster",
            text="Jogadores",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("### Pontuadores da temporada")

        top_jogadores = df.nlargest(
            10,
            "pts_per_game"
        )[["player", "team", "pts_per_game"]]

        fig = px.bar(
            top_jogadores.sort_values("pts_per_game"),
            x="pts_per_game",
            y="player",
            orientation="h",
            color="pts_per_game",
            template="plotly_dark",
            labels={
                "pts_per_game": "Pontos por jogo",
                "player": "Jogador",
            },
        )

        st.plotly_chart(fig, width="stretch")

    st.divider()

    st.markdown("### Dados da temporada")

    st.dataframe(
        add_cluster_names(df)[[
            "player",
            "team",
            "pos",
            "age",
            "pts_per_game",
            "trb_per_game",
            "ast_per_game",
            "cluster_nome",
        ]].sort_values("pts_per_game", ascending=False),
        width="stretch",
        hide_index=True,
    )


# --------------------------------------------------
# JOGADORES
# --------------------------------------------------

elif pagina == "Jogadores":

    st.markdown("## Consulta de jogadores")

    nome = st.text_input(
        "Digite o nome do jogador",
        placeholder="Ex.: LeBron James",
    )

    buscar = st.button("Buscar jogador", type="primary")

    # Guarda a consulta para que a escolha entre jogadores homônimos
    # (selectbox) não faça a busca desaparecer ao recarregar a página.
    if buscar:
        st.session_state["consulta_jogador"] = nome

    consulta = st.session_state.get("consulta_jogador", "")

    if consulta:

        if consulta.strip():

            resultado = search_player(consulta)

            if resultado.empty:
                st.warning("Nenhum jogador encontrado.")

            else:
                if len(resultado) > 1:
                    st.info(
                        f"{len(resultado)} jogadores encontrados. "
                        "Escolha um para ver os detalhes."
                    )
                    escolhido = st.selectbox(
                        "Jogador",
                        resultado["player"].tolist(),
                        key="escolha_jogador",
                    )
                    resultado = resultado[
                        resultado["player"] == escolhido
                    ].reset_index(drop=True)

                jogador = resultado.iloc[0]
                resultado = add_cluster_names(resultado)

                st.markdown(f"### {jogador['player']}")

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Equipe", jogador["team"])
                c2.metric("Posição", jogador["pos"])
                c3.metric("Idade", int(jogador["age"]))
                c4.metric("Perfil", cluster_label(jogador["cluster"]))

                st.markdown("### Estatísticas por jogo")

                estatisticas = {
                    "Pontos": jogador["pts_per_game"],
                    "Rebotes": jogador["trb_per_game"],
                    "Assistências": jogador["ast_per_game"],
                    "Roubos": jogador["stl_per_game"],
                    "Bloqueios": jogador["blk_per_game"],
                    "Minutos": jogador["mp_per_game"],
                }

                colunas = st.columns(3)

                for i, (nome_estatistica, valor) in enumerate(
                    estatisticas.items()
                ):
                    colunas[i % 3].metric(
                        nome_estatistica,
                        f"{valor:.1f}"
                    )

                st.markdown("### Dados completos")

                st.dataframe(
                    resultado,
                    width="stretch",
                    hide_index=True,
                )

        else:
            st.warning("Digite o nome de um jogador.")


# --------------------------------------------------
# COMPARAÇÕES
# --------------------------------------------------

elif pagina == "Comparações":

    st.markdown("## Comparação entre jogadores")

    nomes_disponiveis = sorted(df["player"].unique())

    col1, col2 = st.columns(2)

    with col1:
        jogador1 = st.selectbox(
            "Primeiro jogador",
            nomes_disponiveis,
            index=nomes_disponiveis.index("LeBron James")
            if "LeBron James" in nomes_disponiveis else 0,
        )

    with col2:
        jogador2 = st.selectbox(
            "Segundo jogador",
            nomes_disponiveis,
            index=nomes_disponiveis.index("Stephen Curry")
            if "Stephen Curry" in nomes_disponiveis else 1,
        )

    if st.button("Comparar jogadores", type="primary"):

        if jogador1 == jogador2:
            st.warning("Selecione dois jogadores diferentes.")

        else:
            resultado = add_cluster_names(
                compare_players([jogador1, jogador2])
            )

            st.markdown("### Comparação estatística")

            estatisticas = [
                "pts_per_game",
                "trb_per_game",
                "ast_per_game",
                "stl_per_game",
                "blk_per_game",
                "mp_per_game",
                "fg_percent",
                "x3p_percent",
                "ft_percent",
            ]

            dados_grafico = resultado.melt(
                id_vars=["player"],
                value_vars=estatisticas,
                var_name="Estatística",
                value_name="Valor",
            )

            fig = px.bar(
                dados_grafico,
                x="Estatística",
                y="Valor",
                color="player",
                barmode="group",
                template="plotly_dark",
            )

            st.plotly_chart(fig, width="stretch")

            st.markdown("### Dados completos")

            st.dataframe(
                resultado,
                width="stretch",
                hide_index=True,
            )

# --------------------------------------------------
# JOGADORES SIMILARES
# --------------------------------------------------

elif pagina == "Jogadores similares":

    st.markdown("## Encontre jogadores similares")

    st.write(
        "Selecione um jogador para encontrar outros atletas "
        "com características estatísticas semelhantes, "
        "considerando as 16 variáveis utilizadas pelo modelo."
    )

    nomes_disponiveis = sorted(df["player"].dropna().unique())

    col1, col2 = st.columns([3, 1])

    with col1:
        jogador = st.selectbox(
            "Selecione o jogador",
            nomes_disponiveis,
            index=nomes_disponiveis.index("LeBron James")
            if "LeBron James" in nomes_disponiveis else 0,
            key="jogador_similares",
        )

    with col2:
        limite = st.number_input(
            "Quantidade de similares",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
        )

    if st.button("Encontrar jogadores similares", type="primary"):

        resultado = add_cluster_names(
            find_similar_players(
                jogador,
                limit=int(limite),
            )
        )

        if resultado.empty:

            st.warning("Nenhum jogador similar encontrado.")

        else:

            st.markdown(
                f"### Jogadores estatisticamente semelhantes a {jogador}"
            )

            st.dataframe(
                resultado,
                width="stretch",
                hide_index=True,
            )

            st.caption(
                "A similaridade é calculada pela proximidade "
                "entre os jogadores no espaço das 16 variáveis "
                "padronizadas utilizadas pelo K-Means."
            )

# --------------------------------------------------
# CLUSTERS
# --------------------------------------------------

elif pagina == "Clusters":

    st.markdown("## Exploração dos clusters")

    resumo = add_cluster_names(get_cluster_summary())

    with st.expander("O que cada cluster representa?"):
        for nome_cluster, descricao in DESCRICAO_CLUSTERS.items():
            st.markdown(f"- **{nome_cluster}**: {descricao}")

    col1, col2 = st.columns(2)

    with col1:
        cluster_selecionado = st.selectbox(
            "Selecione um cluster",
            [-1, *sorted(get_cluster_names())],
            format_func=lambda x: (
                "Todos os clusters" if x == -1
                else f"{x} – {cluster_label(x)}"
            ),
        )

    with col2:
        st.metric("Total de clusters", total_clusters if "total_clusters" in locals() else df["cluster"].nunique())

    st.markdown("### Distribuição")

    fig = px.pie(
        resumo,
        names="cluster_nome",
        values="jogadores",
        hole=0.45,
        template="plotly_dark",
        title="Participação dos jogadores por cluster",
    )

    st.plotly_chart(fig, width="stretch")

    if cluster_selecionado == -1:
        jogadores_cluster = df.copy()
        resumo_exibido = resumo
    else:
        jogadores_cluster = df[
            df["cluster"] == cluster_selecionado
        ].copy()

        resumo_exibido = add_cluster_names(
            get_cluster_summary(cluster_selecionado)
        )

    st.markdown("### Perfil estatístico dos clusters")

    st.dataframe(
        resumo_exibido,
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Jogadores pertencentes ao grupo")

    st.dataframe(
        add_cluster_names(jogadores_cluster)[[
            "player",
            "team",
            "pos",
            "pts_per_game",
            "trb_per_game",
            "ast_per_game",
            "cluster_nome",
        ]].sort_values("pts_per_game", ascending=False),
        width="stretch",
        hide_index=True,
    )



# --------------------------------------------------
# EVOLUÇÃO ENTRE TEMPORADAS
# --------------------------------------------------

elif pagina == "Evolução":

    st.markdown("## Evolução dos jogadores (2025 → 2026)")

    st.write(
        "Cada jogador de 2025 é classificado com o mesmo modelo de 2026, "
        "então a mudança de cluster indica mudança de perfil. "
        "A **produção** é pontos + rebotes + assistências por jogo. "
        "Só entram jogadores com pelo menos 20 jogos nas duas temporadas."
    )

    @st.cache_data
    def carregar_evolucao():
        return get_evolution_table()

    evolucao = carregar_evolucao()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jogadores comparados", len(evolucao))
    c2.metric("Em alta", int((evolucao["tendencia"] == "Em alta").sum()))
    c3.metric("Em queda", int((evolucao["tendencia"] == "Em queda").sum()))
    c4.metric("Mudaram de cluster", int(evolucao["mudou_cluster"].sum()))

    st.divider()

    st.markdown("### Evolução de um jogador")

    nomes_evolucao = sorted(evolucao["player"].unique())

    jogador_evo = st.selectbox(
        "Selecione o jogador",
        nomes_evolucao,
        index=nomes_evolucao.index("Nickeil Alexander-Walker")
        if "Nickeil Alexander-Walker" in nomes_evolucao else 0,
        key="jogador_evolucao",
    )

    linha = analyze_player_evolution(jogador_evo).iloc[0]

    m1, m2, m3, m4 = st.columns([1, 1.5, 1, 1])
    m1.metric("Tendência", linha["tendencia"])
    m2.metric(
        "Perfil",
        linha["perfil_atu"],
        delta=(
            f"antes: {linha['perfil_ant']}"
            if linha["mudou_cluster"] else "sem mudança"
        ),
        delta_color="off",
    )
    m3.metric(
        "Produção por jogo",
        f"{linha['producao_atu']:.1f}",
        delta=f"{linha['d_producao']:+.1f}",
    )
    m4.metric(
        "Minutos por jogo",
        f"{linha['mp_atu']:.1f}",
        delta=f"{linha['d_mp']:+.1f}",
    )

    st.caption(f"Contexto: {linha['contexto_minutagem']}.")

    grafico = pd.DataFrame({
        "Estatística": ["Pontos", "Rebotes", "Assistências", "Minutos"] * 2,
        "Temporada": ["2025"] * 4 + ["2026"] * 4,
        "Por jogo": [
            linha["pts_ant"], linha["trb_ant"], linha["ast_ant"], linha["mp_ant"],
            linha["pts_atu"], linha["trb_atu"], linha["ast_atu"], linha["mp_atu"],
        ],
    })

    fig = px.bar(
        grafico,
        x="Estatística",
        y="Por jogo",
        color="Temporada",
        barmode="group",
        text_auto=".1f",
        template="plotly_dark",
        color_discrete_map={"2025": "#64748b", "2026": "#f97316"},
    )

    st.plotly_chart(fig, width="stretch")

    st.divider()

    st.markdown("### Quem mais mudou")

    aba_alta, aba_queda, aba_transicoes = st.tabs(
        ["Em alta", "Em queda", "Mudaram de cluster"]
    )

    colunas_tabela = [
        "player", "perfil_ant", "perfil_atu",
        "producao_ant", "producao_atu", "d_producao",
        "d_mp", "contexto_minutagem",
    ]

    with aba_alta:
        st.dataframe(
            list_players_in_evolution("alta", 15)[colunas_tabela],
            width="stretch",
            hide_index=True,
        )

    with aba_queda:
        st.dataframe(
            list_players_in_evolution("queda", 15)[colunas_tabela],
            width="stretch",
            hide_index=True,
        )

    with aba_transicoes:
        st.dataframe(
            list_players_in_evolution("transicoes", 30)[colunas_tabela],
            width="stretch",
            hide_index=True,
        )

    st.markdown("### Transições entre clusters")

    matriz = get_cluster_transitions()

    fig = px.imshow(
        matriz,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Oranges",
        template="plotly_dark",
        labels={
            "x": "Perfil em 2026",
            "y": "Perfil em 2025",
            "color": "Jogadores",
        },
    )

    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Mudar de cluster nem sempre é mudança de talento: muitas vezes "
        "o jogador só passou a jogar mais ou menos minutos. Use a coluna "
        "de contexto para diferenciar. Os grupos também têm fronteiras "
        "difusas, então jogadores no limite podem trocar de cluster com "
        "pequenas variações."
    )


# --------------------------------------------------
# PERFIL HIPOTÉTICO
# --------------------------------------------------

elif pagina == "Perfil hipotético":

    st.markdown("## Análise de perfil hipotético")

    st.write(
        "Monte um jogador fictício editando as estatísticas "
        "de desempenho. Os aproveitamentos serão calculados "
        "automaticamente a partir das conversões e tentativas."
    )

    nomes_features = {
        "mp_per_game": "Minutos por jogo",
        "fg_per_game": "Cestas de quadra convertidas por jogo",
        "fga_per_game": "Tentativas de arremesso de quadra por jogo",
        "x3p_per_game": "Cestas de três convertidas por jogo",
        "x3pa_per_game": "Tentativas de três por jogo",
        "ft_per_game": "Lances livres convertidos por jogo",
        "fta_per_game": "Tentativas de lance livre por jogo",
        "trb_per_game": "Rebotes por jogo",
        "ast_per_game": "Assistências por jogo",
        "stl_per_game": "Roubos de bola por jogo",
        "blk_per_game": "Bloqueios por jogo",
        "tov_per_game": "Turnovers por jogo",
        "pts_per_game": "Pontos por jogo",
    }

    features = modelo["features"]

    # Estatísticas que o usuário poderá editar.
    features_editaveis = [
        feature for feature in features
        if feature not in [
            "fg_percent",
            "x3p_percent",
            "ft_percent",
        ]
    ]

    st.markdown("### Estatísticas do jogador hipotético")

    st.info(
        "Os valores iniciais correspondem às medianas da temporada 2026. "
        "Você pode editar todos os campos disponíveis."
    )

    with st.form("form_perfil"):

        colunas = st.columns(3)

        estatisticas = {}

        for i, feature in enumerate(features_editaveis):

            valor_mediano = pd.to_numeric(
                df[feature],
                errors="coerce"
            ).median()

            valor_maximo = pd.to_numeric(
                df[feature],
                errors="coerce"
            ).max()

            if pd.isna(valor_mediano):
                valor_mediano = 0.0

            if pd.isna(valor_maximo):
                valor_maximo = 1.0

            limite_maximo = None

            with colunas[i % 3]:

                estatisticas[feature] = st.number_input(
                    nomes_features.get(feature, feature),
                    min_value=0.0,
                    max_value=limite_maximo,
                    value=float(valor_mediano),
                    step=0.1,
                    key=f"hipotetico_{feature}",
                )

        enviar = st.form_submit_button(
            "Analisar perfil",
            type="primary",
        )

    if enviar:

        # Calcula os aproveitamentos automaticamente.
        fg = estatisticas["fg_per_game"]
        fga = estatisticas["fga_per_game"]

        x3p = estatisticas["x3p_per_game"]
        x3pa = estatisticas["x3pa_per_game"]

        ft = estatisticas["ft_per_game"]
        fta = estatisticas["fta_per_game"]

        # Sem tentativas o aproveitamento é indefinido (NaN), não 0%:
        # o imputer do modelo o substitui pela mediana da liga.
        estatisticas["fg_percent"] = (
            fg / fga if fga > 0 else np.nan
        )

        estatisticas["x3p_percent"] = (
            x3p / x3pa if x3pa > 0 else np.nan
        )

        estatisticas["ft_percent"] = (
            ft / fta if fta > 0 else np.nan
        )

        # Garante que todas as 16 features sejam enviadas
        # na ordem esperada pelo modelo.
        estatisticas = {
            feature: estatisticas[feature]
            for feature in features
        }

        resultado = analyze_hypothetical_player(
            stats=estatisticas,
            limit=10,
        )

        st.success(
            f"O perfil hipotético foi classificado no "
            f"cluster {resultado['cluster']} – "
            f"{cluster_label(resultado['cluster'])}."
        )

        st.markdown("### Aproveitamentos calculados")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Aproveitamento de quadra",
            "—" if pd.isna(estatisticas["fg_percent"])
            else f"{estatisticas['fg_percent'] * 100:.1f}%"
        )

        c2.metric(
            "Aproveitamento de três",
            "—" if pd.isna(estatisticas["x3p_percent"])
            else f"{estatisticas['x3p_percent'] * 100:.1f}%"
        )

        c3.metric(
            "Aproveitamento de lance livre",
            "—" if pd.isna(estatisticas["ft_percent"])
            else f"{estatisticas['ft_percent'] * 100:.1f}%"
        )

        st.markdown("### Jogadores estatisticamente semelhantes")

        jogadores_similares = resultado["similar_players"]

        if jogadores_similares.empty:

            st.warning("Nenhum jogador similar encontrado.")

        else:

            st.dataframe(
                add_cluster_names(jogadores_similares),
                width="stretch",
                hide_index=True,
            )

        st.caption(
            "O modelo continua utilizando as 16 variáveis originais. "
            "Os três aproveitamentos são calculados automaticamente "
            "e enviados junto às demais estatísticas."
        )

