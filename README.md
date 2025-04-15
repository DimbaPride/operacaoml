# Projeto Assistente de Anúncios Mercado Livre (Gestor de Promoções)

Este projeto visa desenvolver um assistente baseado em IA (Agente "Arquiteto de Anúncios") para auxiliar na criação e otimização de anúncios para a plataforma Mercado Livre.

## Objetivo Principal

O agente deve receber informações básicas sobre um produto e, através de pesquisa de mercado (tendências, atributos da categoria, análise de concorrentes) e inteligência artificial (LLM), gerar sugestões otimizadas para:

*   **Título do Anúncio:** Relevante, com palavras-chave importantes e seguindo boas práticas.
*   **Ficha Técnica (Atributos):** Identificando atributos obrigatórios e recomendados, e pré-preenchendo valores quando possível.
*   **Descrição do Produto:** Persuasiva, detalhada e incorporando informações relevantes.
*   **FAQ (Perguntas Frequentes):** Antecipando dúvidas comuns dos compradores.

O objetivo é fornecer um "rascunho de luxo" que acelere o processo de criação de anúncios e melhore sua qualidade, exigindo apenas o refinamento e validação final do usuário.

## Estrutura do Projeto

O código principal do agente está localizado em `src/agents/ad_creator/`:

*   `data_models.py`: Define as estruturas de dados (`dataclasses`) para input, pesquisa e output.
*   `input_handler.py`: (Uso via terminal - atualmente secundário) Coleta input do usuário via console.
*   `market_research.py`: Responsável por fazer chamadas à API do ML para buscar tendências, atributos da categoria e detalhes de concorrentes.
*   `content_generator.py`: (Ainda não implementado) Responsável por usar os dados coletados para gerar as sugestões de título, ficha, descrição e FAQ.
*   `main_flow.py`: (Ainda não implementado) Orquestrará o fluxo completo dos agentes (poderá usar LangChain/LangGraph no futuro).

A interface visual principal é o arquivo `app_visual.py` na raiz do projeto, que utiliza a biblioteca Streamlit.

## Status Atual (11/04/2025 - Conforme última interação)

1.  **Estrutura de Pastas:** Criada em `src/agents/ad_creator/`.
2.  **Modelos de Dados:** `ProductInput`, `MarketResearchOutput`, `CompetitorInfo`, `AdOutput` definidos em `data_models.py`.
3.  **Interface Visual (`app_visual.py`):**
    *   Implementada com Streamlit.
    *   Coleta input do usuário (Categoria, Nome Base, Marca, Modelo, EAN) e IDs de concorrentes (formato MLB-xxxx).
    *   Valida inputs básicos e IDs de concorrentes.
    *   Usa `logging` para registro interno.
4.  **Pesquisa de Mercado (`market_research.py`):**
    *   Implementada função `fetch_market_data`.
    *   Busca **Tendências** da categoria (`GET /trends`) via API (requer auth).
    *   Busca **Atributos** da categoria (`GET /categories/{id}/attributes`) via API.
    *   Busca **IDs de Concorrentes:**
        *   Prioriza IDs fornecidos manualmente pelo usuário na interface.
        *   Se nenhum ID manual for fornecido, tenta buscar automaticamente (`GET /search`), **mas esta chamada atualmente retorna 403 Forbidden (problema de escopo/permissão da API)**.
    *   Implementada função `fetch_competitor_details` para buscar detalhes e descrição de concorrentes individuais (`GET /items/{id}` e `GET /items/{id}/description`).
    *   `fetch_market_data` orquestra a busca de detalhes para os concorrentes selecionados (manuais ou automáticos) usando `asyncio.gather`.
    *   Retorna `MarketResearchOutput` contendo tendências, atributos da categoria e a lista de `CompetitorInfo` (detalhes dos concorrentes).
5.  **Integração UI <-> Pesquisa:**
    *   `app_visual.py` chama `fetch_market_data` após o input do usuário.
    *   A interface exibe as **Tendências** e os **Atributos da Categoria** (incluindo obrigatórios).
    *   Os detalhes dos concorrentes (`competitor_analysis`) **são buscados** pelo `market_research.py`, mas **ainda não são exibidos** na interface `app_visual.py`.

## Próximos Passos Imediatos

1.  **Exibir Análise de Concorrência na UI:** Modificar `app_visual.py` para iterar sobre `market_data.competitor_analysis` e exibir as informações coletadas (Título, Preço, Atributos, Descrição, etc.) de forma organizada para cada concorrente.
2.  **Implementar Geração de Conteúdo (`content_generator.py`):** Criar a função `generate_ad_content` que receberá `ProductInput` e `MarketResearchOutput` e gerará as sugestões (começar com Título e Ficha Técnica).
3.  **Integrar Geração à UI:** Chamar `generate_ad_content` no `app_visual.py` após a pesquisa de mercado e exibir as sugestões geradas.

## Como Executar

1.  **Certifique-se de ter Python instalado.**
2.  **Clone o repositório (se aplicável).**
3.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv venv
    # Windows (CMD/PowerShell)
    .\venv\Scripts\activate
    # Windows (Git Bash)
    source venv/Scripts/activate
    # macOS/Linux
    source venv/bin/activate
    ```
4.  **Instale as dependências:** (Nota: Criar `requirements.txt` se ainda não existir)
    ```bash
    pip install -r requirements.txt
    # Se requirements.txt não existir, instalar manualmente:
    # pip install streamlit httpx python-dotenv
    ```
5.  **Configure o `.env`:** Certifique-se de que o arquivo `.env` na raiz contém as credenciais da API do Mercado Livre (`MELI_CLIENT_ID`, `MELI_CLIENT_SECRET`, `MELI_REFRESH_TOKEN`).
6.  **Execute a aplicação visual:**
    ```bash
    streamlit run app_visual.py
    ```
7.  A interface abrirá no seu navegador.

## Considerações Futuras

*   Investigar e corrigir o erro `403 Forbidden` na busca automática de concorrentes (provavelmente ajuste de escopos da API).
*   Refinar a extração e análise de dados dos concorrentes.
*   Melhorar a geração de conteúdo usando LLMs de forma mais sofisticada.
*   Implementar a orquestração usando LangChain/LangGraph.
*   Adicionar mais opções de input (especificações detalhadas, texto de manual).
*   Salvar/Exportar os resultados gerados.
*   Criar um arquivo `requirements.txt` formal.
