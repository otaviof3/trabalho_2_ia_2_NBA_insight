# 🏀 NBA Insight

Sistema de análise estatística de jogadores da NBA utilizando técnicas de Machine Learning e integração com Model Context Protocol (MCP).

## 1. Sobre o projeto

O **NBA Insight** é uma aplicação desenvolvida como parte do Trabalho 2 da disciplina de Inteligência Artificial II.

O projeto utiliza dados estatísticos de jogadores da NBA na temporada de 2026 para identificar perfis semelhantes por meio de técnicas de aprendizado de máquina não supervisionado.

A aplicação combina um modelo de agrupamento K-Means, uma interface interativa desenvolvida com Streamlit e um servidor MCP que disponibiliza as funcionalidades analíticas para assistentes de inteligência artificial compatíveis.

## 2. Funcionalidades

A aplicação disponibiliza as seguintes funcionalidades:

- **Dashboard:** visão geral da temporada, quantidade de jogadores e equipes, distribuição dos clusters e principais pontuadores.
- **Consulta de jogadores:** pesquisa de atletas e visualização de suas estatísticas individuais.
- **Jogadores similares:** identificação de atletas com características estatísticas próximas às de um jogador selecionado.
- **Comparação de jogadores:** comparação estatística entre dois atletas.
- **Exploração de clusters:** visualização dos grupos identificados pelo modelo e de suas características.
- **Perfil hipotético:** criação de um jogador fictício a partir de estatísticas editáveis, classificação em um cluster e identificação de jogadores semelhantes.
- **Integração MCP:** disponibilização das funcionalidades analíticas por meio de ferramentas que podem ser utilizadas por assistentes de IA.

## 3. Tecnologias utilizadas

- **Python:** linguagem principal.
- **Pandas e NumPy:** manipulação e processamento dos dados.
- **Scikit-learn:** preparação dos atributos e modelo K-Means.
- **Joblib:** persistência do modelo treinado.
- **Streamlit:** desenvolvimento da interface.
- **Plotly:** visualizações interativas.
- **FastMCP / MCP:** disponibilização das funcionalidades analíticas por meio do Model Context Protocol.
- **Pytest:** testes automatizados.

## 4. Modelo de Machine Learning

O projeto utiliza o algoritmo **K-Means**, uma técnica de aprendizado não supervisionado que agrupa observações de acordo com a proximidade entre suas características.

### Configuração utilizada

| Parâmetro | Valor |
|---|---|
| Temporada | 2026 |
| Algoritmo | K-Means |
| Quantidade de clusters | 4 |
| Random State | 42 |
| N_init | 10 |
| Imputação de valores ausentes | Mediana |
| Padronização | StandardScaler |
| Quantidade de atributos | 16 |

### Atributos utilizados

- Minutos por jogo;
- Cestas de quadra convertidas e tentadas;
- Aproveitamento de quadra;
- Cestas de três convertidas e tentadas;
- Aproveitamento de três;
- Lances livres convertidos e tentados;
- Aproveitamento de lances livres;
- Rebotes;
- Assistências;
- Roubos de bola;
- Bloqueios;
- Turnovers;
- Pontos.

Os 16 atributos são utilizados no treinamento e na classificação dos jogadores, após o tratamento de valores ausentes por imputação da mediana e a padronização com StandardScaler.

O modelo foi treinado e salvo em `artifacts/kmeans_nba_2026.joblib`, permitindo que a aplicação carregue o modelo existente sem precisar executar novamente o treinamento.

**Importante:** os clusters representam agrupamentos estatísticos, não posições oficiais, categorias de qualidade ou avaliações definitivas dos jogadores.

## 5. Estrutura do projeto

```text
nba-insight/
├── artifacts/
│   └── kmeans_nba_2026.joblib
├── data/
│   └── Player Per Game.csv
├── docs/
│   └── architecture.md
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
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## 6. Requisitos

- Python instalado;
- Git, caso deseje clonar o repositório;
- Windows, Linux ou macOS;
- Dependências listadas em `requirements.txt`.

## 7. Instalação

Clone o repositório ou obtenha os arquivos do projeto.

No terminal, acesse a pasta principal:

```powershell
cd caminho\para\nba-insight
```

Crie um ambiente virtual:

```powershell
python -m venv .venv
```

Ative o ambiente:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 8. Executando a aplicação

Com o ambiente virtual ativado, execute:

```powershell
streamlit run src/nba_insight/app.py
```

A interface será disponibilizada localmente pelo Streamlit. O terminal apresentará o endereço para acessar a aplicação pelo navegador.

## 9. Executando o servidor MCP

O projeto disponibiliza suas funcionalidades analíticas por meio de um servidor MCP.

Para executar:

```powershell
python -m src.nba_insight.mcp_server
```

Para testar o servidor com o MCP Inspector:

```powershell
mcp dev src/nba_insight/mcp_server.py
```

O Inspector permite visualizar e executar as ferramentas disponibilizadas pelo servidor.

### Ferramentas MCP disponíveis

| Ferramenta | Descrição |
|---|---|
| `buscar_jogador` | Pesquisa um jogador pelo nome. |
| `comparar_jogadores` | Compara estatísticas de jogadores. |
| `buscar_jogadores_similares` | Encontra atletas estatisticamente semelhantes. |
| `consultar_clusters` | Consulta os agrupamentos e seus resumos. |
| `buscar_jogadores` | Filtra jogadores de acordo com critérios estatísticos. |
| `analisar_perfil_hipotetico` | Classifica um perfil fictício e encontra jogadores semelhantes. |
| `consultar_modelo` | Retorna informações sobre o modelo utilizado. |

## 10. Integração com Claude Desktop

O servidor MCP pode ser integrado ao Claude Desktop por meio da configuração de servidores MCP.

Exemplo de configuração:

```json
{
  "mcpServers": {
    "nba-insight": {
      "command": "C:\\CAMINHO\\nba-insight\\.venv\\Scripts\\python.exe",
      "args": [
        "-m",
        "src.nba_insight.mcp_server"
      ],
      "env": {
        "PYTHONPATH": "C:\\CAMINHO\\nba-insight"
      }
    }
  }
}
```

Substitua `C:\\CAMINHO\\nba-insight` pelo caminho absoluto da pasta do projeto no computador.

Após salvar a configuração e reiniciar o Claude Desktop, o servidor poderá disponibilizar suas ferramentas ao assistente.

## 11. Treinamento do modelo

Caso seja necessário treinar novamente o modelo, execute:

```powershell
python scripts/train_model.py
```

O treinamento utiliza os dados disponíveis em `data/Player Per Game.csv` e gera o artefato em `artifacts/kmeans_nba_2026.joblib`.

A aplicação utiliza o artefato salvo, portanto não é necessário executar o treinamento a cada inicialização.

## 12. Testes

Os testes automatizados estão organizados na pasta `tests/`.

Para executá-los:

```powershell
pytest -v
```

O parâmetro `-v` permite visualizar individualmente os testes executados e seus resultados.

## 13. Observações

- Os resultados dependem dos dados disponibilizados no conjunto utilizado pelo projeto.
- A similaridade entre jogadores é calculada por distância euclidiana no espaço das variáveis padronizadas.
- A classificação em clusters não representa uma previsão de desempenho futuro.
- Perfis hipotéticos muito distantes dos padrões observados na base podem produzir classificações pouco representativas.

---

**NBA Insight — Trabalho 2 | Inteligência Artificial II**