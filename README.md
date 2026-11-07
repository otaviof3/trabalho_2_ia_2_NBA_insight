# 🏀 NBA Insight

Plataforma de apoio à análise de desempenho de jogadores da NBA, com um modelo de Machine Learning e um servidor MCP que conecta assistentes de IA aos dados e ao modelo.

Trabalho 2 de Inteligência Artificial II (2026/02). Modalidade 3: integração com IA em contexto avançado (MCP).

Autor: Otávio Ferreira Dahlke | Professor: Rhauani Weber Aita Fazul

## 1. O problema

Quem analisa basquete (analistas de clube, comissões técnicas, jornalistas, fãs) faz perguntas como "quem joga parecido com este jogador?", "quem poderia substituí-lo?" ou "que tipo de jogador é esse, além da posição que aparece na ficha?".

A posição tradicional (armador, ala, pivô) descreve cada vez pior o que o jogador faz em quadra: tem pivô que arremessa de três e armador que pega mais rebote que ala.

A dor, então, é que comparar jogadores e achar perfis parecidos é lento, manual e depende de intuição.

| Público | O que ganha |
|---|---|
| Analistas e scouting | Achar substitutos e alternativas de perfil parecido em segundos |
| Comissões técnicas | Entender o papel estatístico de cada atleta, e não só a posição |
| Jornalistas e fãs | Consultar e comparar jogadores por conversa, sem planilha |

## 2. A solução

O NBA Insight junta três partes:

1. **Modelo de ML (K-Means):** agrupa os jogadores da temporada 2026 em 4 perfis estatísticos, sem usar a posição como entrada.
2. **Interface web (Streamlit):** dashboard, consulta de jogadores, jogadores similares, comparação, exploração dos clusters e simulação de um jogador hipotético.
3. **Servidor MCP (FastMCP):** expõe essas mesmas funções como ferramentas para um assistente de IA, como o Claude Desktop. O usuário pergunta em linguagem natural e o assistente consulta os dados e o modelo.

### Os 4 clusters

A interface mostra o cluster pelo número (0 a 3). A interpretação de cada um, feita a partir do perfil médio dos grupos, é esta:

| Cluster | Jogadores | Interpretação | Perfil médio |
|---|---|---|---|
| 0 | 222 | Reservas de baixa minutagem | ≈ 10 min e 3,6 pts por jogo |
| 1 | 206 | Jogadores de rotação perimetrais | ≈ 23 min, 10 pts, mais bolas de três |
| 2 | 67 | Especialistas de garrafão | ≈ 23 min, 9,7 pts, FG% ≈ 59% e quase 1 toco por jogo |
| 3 | 87 | Jogadores de alta minutagem | ≈ 32 min, 20,7 pts e 4,9 assistências |

## 3. Como a IA é utilizada

| Camada | Técnica | Papel |
|---|---|---|
| Aprendizado não supervisionado | K-Means (K=4), com imputação pela mediana e padronização | Descobrir perfis de jogadores e classificar novos perfis |
| Similaridade | Distância euclidiana no espaço padronizado das 16 estatísticas | Encontrar jogadores parecidos |
| Integração com assistente | Servidor MCP (FastMCP) com 7 ferramentas | Deixar um modelo de linguagem usar os dados e o modelo como ferramentas |

O assistente de linguagem natural entra só como interface. Quem calcula é o código e o modelo, então ele não precisa "chutar" estatísticas.

## 4. Arquitetura

```mermaid
flowchart TD
    A[CSV: Player Per Game] --> B[data_loader]
    C[Artefato K-Means .joblib] --> D[model_loader]
    B --> E[analytics: camada analítica]
    D --> E
    E --> F[Interface Streamlit]
    E --> G[Servidor MCP]
    F --> H[Usuário]
    G --> I[Assistente de IA]
    I --> H
```

A interface e o servidor MCP usam a mesma camada analítica (`analytics.py`), então as regras não ficam duplicadas. Mais detalhes em [`docs/architecture.md`](docs/architecture.md).

## 5. Dados e modelo

**Dataset.** O arquivo é o `Player Per Game.csv` do conjunto [NBA, ABA, BAA Stats](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats/data), do Kaggle (dados do Basketball Reference). Ele já está em `data/`.

**Recorte.** Liga NBA, temporada 2026, uma linha por jogador. Jogadores que passaram por mais de um time usam a linha agregada. No fim sobram 582 jogadores.

**Atributos (16, todos por jogo):** minutos, cestas de quadra (convertidas, tentadas e %), bolas de três (convertidas, tentadas e %), lances livres (convertidos, tentados e %), rebotes, assistências, roubos, tocos, erros e pontos.

**Pré-processamento.** Valores ausentes (por exemplo, o % de três de quem nunca tentou) são preenchidos pela mediana. Depois os dados passam pelo `StandardScaler`.

| Parâmetro do modelo | Valor |
|---|---|
| Algoritmo | K-Means |
| Clusters | 4 |
| `n_init` | 10 |
| `random_state` | 42 |
| Atributos | 16 |

O modelo treinado, junto com o imputer e o scaler, fica salvo em `artifacts/kmeans_nba_2026.joblib`. A aplicação carrega esse arquivo e não precisa treinar de novo a cada execução.

## 6. Como executar

### 6.1 Requisitos

- Python 3.10 ou superior (desenvolvido com [PREENCHER versão])
- Git (opcional, só se for clonar)
- Windows, Linux ou macOS

### 6.2 Obter o projeto

Com Git:

```bash
git clone https://github.com/otaviof3/trabalho_2_ia_2_NBA_insight.git
cd trabalho_2_ia_2_NBA_insight
```

Ou baixe o ZIP pelo GitHub (botão **Code → Download ZIP**), extraia e abra um terminal dentro da pasta extraída (`trabalho_2_ia_2_NBA_insight-main`).

Todos os comandos abaixo devem ser executados dentro dessa pasta.

### 6.3 Ambiente virtual e dependências

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6.4 Dataset

O arquivo `data/Player Per Game.csv` já vem no repositório. Se precisar baixar de novo, ele está no [Kaggle](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats/data); coloque-o na pasta `data/` com esse mesmo nome.

### 6.5 Treinar o modelo de novo (opcional)

O artefato treinado já está em `artifacts/`. Para gerá-lo de novo:

```bash
python scripts/train_model.py
```

### 6.6 Abrir a interface

```bash
streamlit run src/nba_insight/app.py
```

O terminal mostra o endereço local (normalmente http://localhost:8501).

### 6.7 Rodar o servidor MCP

```bash
python -m src.nba_insight.mcp_server
```

Para testar as ferramentas no MCP Inspector:

```bash
mcp dev src/nba_insight/mcp_server.py
```

### 6.8 Usar com o Claude Desktop

Adicione isto à configuração de servidores MCP do Claude Desktop:

```json
{
  "mcpServers": {
    "nba-insight": {
      "command": "C:\\CAMINHO\\nba-insight\\.venv\\Scripts\\python.exe",
      "args": ["-m", "src.nba_insight.mcp_server"],
      "env": { "PYTHONPATH": "C:\\CAMINHO\\nba-insight" }
    }
  }
}
```

Troque `C:\\CAMINHO\\nba-insight` pelo caminho absoluto da pasta do projeto no seu computador (no Linux ou macOS, use o caminho de `.venv/bin/python`). Depois de salvar, reinicie o Claude Desktop.

Perguntas que dá para fazer ao assistente:

- "Quais jogadores se parecem com Brandon Ingram? Liste os 5 mais próximos."
- "Compare Nikola Jokić e Luka Dončić."
- "Mostre o resumo de cada cluster."
- "Quais jogadores têm pelo menos 20 pontos e 5 assistências por jogo?"
- "Crie um jogador com 30 minutos, 22 pontos, 7 rebotes e 5 assistências e diga em que cluster ele cai."

### 6.9 Testes

```bash
pytest -v
```

## 7. Ferramentas MCP

| Ferramenta | O que faz |
|---|---|
| `buscar_jogador` | Pesquisa um jogador pelo nome |
| `comparar_jogadores` | Compara estatísticas de jogadores |
| `buscar_jogadores_similares` | Encontra jogadores estatisticamente próximos |
| `consultar_clusters` | Mostra os clusters e o resumo de cada um |
| `buscar_jogadores` | Filtra jogadores por critérios estatísticos (mínimo e máximo) |
| `analisar_perfil_hipotetico` | Classifica um jogador fictício e lista os reais mais parecidos |
| `consultar_modelo` | Retorna informações do modelo (temporada, clusters, atributos) |

## 8. Limitações

- Os grupos não são bem separados: o Silhouette Score (≈ 0,24, medido no modelo) indica estrutura real, mas com fronteiras difusas. Jogadores no limite entre dois grupos podem mudar de cluster com pequenas variações nos dados.
- Estatística por jogo mistura talento com oportunidade: um jogador bom com poucos minutos cai junto com os reservas.
- Só a temporada 2026 é analisada. Lesões, trocas de time e contexto da equipe não entram.
- A classificação em cluster não é uma previsão de desempenho futuro.
- Perfis hipotéticos muito fora do padrão da liga geram classificações pouco representativas.

## 9. Próximos passos

- **Projeção de desempenho:** um modelo preditivo, validado em temporadas futuras, com cenários do tipo "e se jogar mais minutos?".
- **Várias temporadas:** seletor de temporada e comparação entre épocas.
- **Mais estatísticas:** métricas avançadas (PER, uso, BPM) e valores por 36 minutos, para reduzir o viés da minutagem.
- **Uso real:** publicar a interface (por exemplo, no Streamlit Community Cloud) e o servidor MCP, para que pessoas usem sem instalar nada.

## 10. Estrutura do projeto

```text
.
├── artifacts/
│   └── kmeans_nba_2026.joblib
├── data/
│   └── Player Per Game.csv
├── docs/
│   ├── architecture.md
├── scripts/
│   └── train_model.py
├── src/
│   └── nba_insight/
│       ├── __init__.py
│       ├── analytics.py
│       ├── app.py
│       ├── config.py
│       ├── data_loader.py
│       ├── mcp_server.py
│       └── model_loader.py
├── tests/
│   ├── test_analytics.py
│   └── test_mcp_tools.py
├── .gitignore
├── README.md
└── requirements.txt
```

## 11. Créditos

Dados do Basketball Reference, pelo dataset de Sumitro Datta no Kaggle. Projeto acadêmico da disciplina de Inteligência Artificial II.