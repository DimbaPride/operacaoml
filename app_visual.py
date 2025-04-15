# -*- coding: utf-8 -*-
# FINALIDADE: Interface visual (Streamlit) para o Agente Ad Creator.

import streamlit as st
import asyncio
import logging # Importar logging
import sys # Para verificar o caminho
import re # Importar regex para extrair MLB ID
from typing import List, Dict, Any, Optional
import pandas as pd
import httpx # Adicionar import
import urllib.parse # Adicionar import
import json # Adicionar import
import os # Adicionar import

# <<< Mover import essencial para antes do try >>>
from src.api.auth import MeliAuth

# --- Configuração de Logging Básico ---
# Configurar apenas se não foi configurado antes (evita duplicação se chamado de outro lugar)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO, # Nível inicial (INFO, DEBUG, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        # Poderíamos adicionar um FileHandler aqui depois, se quisermos logar em arquivo
        # handlers=[
        #     logging.StreamHandler(), # Loga no console
        #     # logging.FileHandler("ad_creator_app.log") # Loga em arquivo
        # ]
    )

# Obter um logger para este módulo
logger = logging.getLogger(__name__)

logger.info("Aplicação Streamlit iniciada.")

# Importar a estrutura de dados e, futuramente, os módulos do agente
try:
    # Verificar se o diretório src está no path (ajuda no debug de importação)
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
    logger.debug(f"Verificando sys.path para src: {src_path in sys.path}")
    # logger.debug(f"sys.path: {sys.path}")

    from src.agents.ad_creator.data_models import ProductInput, MarketResearchOutput, CompetitorInfo, GeneratedAdContent, NCMSuggestion
    from src.agents.ad_creator.market_research import fetch_market_data
    from src.agents.ad_creator.content_generator import generate_ad_content_graph
except ImportError as e:
    logger.error(f"Erro de importação: {e}", exc_info=True)
    st.error(f"Erro fatal ao importar módulos internos do agente: {e}")
    st.error("Verifique se o Streamlit está sendo executado a partir da pasta raiz do projeto ('Gestor de Promoções') e se a estrutura de pastas está correta.")
    st.stop()
except Exception as e:
    logger.error(f"Erro inesperado durante a importação: {e}", exc_info=True)
    st.error(f"Ocorreu um erro inesperado ao carregar os componentes: {e}")
    st.stop()

# --- Configuração da Página ---
st.set_page_config(
    page_title="Gerador de Anúncios ML",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 Gerador de Anúncios Otimizados para Mercado Livre")
st.caption("Insira as informações do seu produto e URLs de concorrentes para iniciar a análise.")

# --- Funções de API de Categoria (Async Core Logic) ---

# Função ASYNC para buscar categorias sugeridas (lógica principal)
async def _async_search_ml_categories(query: str, _auth: MeliAuth) -> List[Dict[str, Any]]:
    """
    Busca categorias sugeridas (domain_discovery) e, para cada sugestão,
    busca seus detalhes (incluindo path) para retornar uma lista completa.
    """
    initial_suggestions = []
    final_results_with_path = []
    
    if not query:
        return final_results_with_path
    
    # 1. Buscar sugestões iniciais via domain_discovery
    encoded_query = urllib.parse.quote(query)
    search_url = f"https://api.mercadolibre.com/sites/MLB/domain_discovery/search"
    params = {"q": encoded_query, "limit": 8}
    logger.info(f"Buscando sugestões iniciais para query: '{query}' em {search_url} com limite 8")
    
    access_token = await _auth.get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}
    
    if not headers:
        logger.error("Token de acesso indisponível para busca inicial de categorias.")
        st.error("Erro de autenticação ao buscar categorias (token).")
        return []

    try:
        response = await _auth.client.get(search_url, params=params, headers=headers)
        response.raise_for_status()
        initial_suggestions_raw = response.json()
        if isinstance(initial_suggestions_raw, list):
            # Processar para pegar IDs únicos e dados básicos
            processed_ids = set()
            for item in initial_suggestions_raw:
                 cat_id = item.get('category_id')
                 if cat_id and cat_id not in processed_ids:
                     initial_suggestions.append({
                         'id': cat_id,
                         'name': item.get('category_name'),
                         'domain_name': item.get('domain_name')
                     })
                     processed_ids.add(cat_id)
            logger.info(f"Obtidas {len(initial_suggestions)} sugestões iniciais únicas.")
        else:
            logger.warning(f"Formato inesperado na resposta do domain_discovery: {initial_suggestions_raw}")
            
    except httpx.HTTPStatusError as e:
        logger.error(f"Erro HTTP {e.response.status_code} na busca inicial: {e.response.text}")
        st.error(f"Erro {e.response.status_code} ao buscar sugestões: {e.response.text}")
        return [] # Parar aqui se a busca inicial falhar
    except Exception as e:
        logger.exception(f"Erro inesperado na busca inicial de categorias: {e}")
        st.error("Erro inesperado ao buscar sugestões.")
        return []

    # 2. Para cada sugestão, buscar detalhes e o path
    if not initial_suggestions:
         return []
         
    details_tasks = []
    for suggestion in initial_suggestions:
        cat_id = suggestion['id']
        details_url = f"https://api.mercadolibre.com/categories/{cat_id}"
        logger.info(f"Agendando busca de detalhes para Categoria ID: {cat_id}")
        # Usar o mesmo cliente httpx e headers
        details_tasks.append(_auth.client.get(details_url, headers=headers))
    
    try:
        # Executar todas as buscas de detalhes em paralelo
        detail_responses = await asyncio.gather(*details_tasks, return_exceptions=True)
        
        for i, response_or_exception in enumerate(detail_responses):
            suggestion = initial_suggestions[i]
            cat_id = suggestion['id']
            
            if isinstance(response_or_exception, Exception):
                logger.error(f"Erro ao buscar detalhes para cat_id {cat_id}: {response_or_exception}")
                continue # Pular esta categoria se detalhes falharam
                
            try:
                response = response_or_exception
                response.raise_for_status() # Levanta erro para status >= 400
                details_data = response.json()
                
                path_from_root = details_data.get('path_from_root', [])
                path_string = " > ".join([p.get('name', '?') for p in path_from_root])
                
                # Adicionar à lista final com todos os dados
                final_results_with_path.append({
                    'id': str(cat_id), # Garantir string
                    'name': str(suggestion.get('name', '?')), # Garantir string
                    'domain_name': str(suggestion.get('domain_name', '?')), # Garantir string
                    'path_string': path_string # Caminho formatado
                })
                logger.info(f"Detalhes e caminho obtidos para cat_id {cat_id}")
                
            except httpx.HTTPStatusError as e_details:
                 logger.error(f"Erro HTTP {e_details.response.status_code} ao buscar detalhes para {cat_id}: {e_details.response.text}")
            except Exception as e_details_parse:
                 logger.error(f"Erro ao processar detalhes para cat_id {cat_id}: {e_details_parse}")

    except Exception as e_gather:
         logger.exception(f"Erro durante asyncio.gather para buscar detalhes: {e_gather}")
         st.error("Erro ao buscar detalhes completos das categorias.")

    # Ordenar pelo path pode ser útil
    final_results_with_path.sort(key=lambda x: x['path_string'])
    
    logger.info(f"Retornando {len(final_results_with_path)} categorias com caminho completo.")
    return final_results_with_path

# Função ASYNC para buscar detalhes de UMA categoria (lógica principal)
# REMOVIDO: @st.cache_data(ttl=3600)
async def _async_get_category_details(category_id: str, _auth: MeliAuth) -> Optional[Dict[str, Any]]:
    """Busca detalhes de uma categoria específica, incluindo o path."""
    if not category_id:
        return None
    
    url = f"https://api.mercadolibre.com/categories/{category_id}"
    logger.info(f"Buscando detalhes para Categoria ID: {category_id} (com auth) na URL: {url}")
    
    access_token = await _auth.get_access_token()
    headers = {}
    if access_token:
        headers = {"Authorization": f"Bearer {access_token}"}
    else:
        logger.error(f"Não foi possível obter token de acesso para buscar detalhes da categoria {category_id}.")
        st.error("Erro de autenticação ao buscar detalhes da categoria.")
        return None
        
    try:
        response = await _auth.client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Detalhes obtidos para a categoria {category_id}.")
        
        # <<< LOG DETALHADO PARA VERIFICAÇÃO >>>
        if data:
            logger.info(f"--- VERIFICAÇÃO API --- Detalhes completos recebidos para {category_id}: {json.dumps(data, indent=2)}")
        else:
            logger.warning(f"--- VERIFICAÇÃO API --- Não foi possível obter detalhes para {category_id}.")
            
        return data
    except httpx.RequestError as e:
        logger.error(f"Erro de conexão ao buscar detalhes da categoria {category_id}: {e}")
        st.error(f"Erro de conexão ao buscar detalhes da categoria: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"Erro HTTP {e.response.status_code} ao buscar detalhes da categoria {category_id}: {e.response.text}")
        st.error(f"Erro {e.response.status_code} ao buscar detalhes da categoria: {e.response.text}")
    except Exception as e:
        logger.exception(f"Erro inesperado ao buscar detalhes da categoria {category_id}: {e}")
        st.error(f"Erro inesperado ao buscar detalhes da categoria.")
    return None

# --- Funções de API de Categoria (Synchronous Wrappers with Cache) ---

# Wrapper para busca de categorias (com path agora)
def search_ml_categories_cached(query: str) -> List[Dict[str, Any]]:
    """Wrapper síncrono (SEM CACHE) para buscar categorias COM CAMINHO. Gerencia Auth."""
    if not query:
        return []
    auth_cat = MeliAuth() 
    results = []
    try:
        # Chama a função async real usando run_async
        results = run_async(_async_search_ml_categories(query, auth_cat))
    except Exception as e:
        logger.error(f"Erro no wrapper cacheado search_ml_categories_cached para query '{query}': {e}", exc_info=True)
        st.error(f"Erro ao buscar categorias: {e}") # Re-lança erro para UI se necessário
    finally:
        # Garante que o cliente HTTP seja fechado
        logger.info(f"Fechando cliente HTTP da busca de categorias (wrapper) para query: '{query}'")
        run_async(auth_cat.close())
        logger.info(f"Cliente HTTP da busca de categorias (wrapper) fechado para query: '{query}'")
    return results

# Wrapper para buscar detalhes da categoria SELECIONADA
@st.cache_data(ttl=3600) # Cache AINDA APLICADO AQUI
def get_category_details_cached(category_id: str) -> Optional[Dict[str, Any]]:
    """Wrapper síncrono cacheado para buscar detalhes da categoria SELECIONADA."""
    if not category_id:
        return None
    auth_details = MeliAuth()
    details = None
    try:
        # Chama a função async real usando run_async
        details = run_async(_async_get_category_details(category_id, auth_details))
        
        # <<< LOG DETALHADO PARA VERIFICAÇÃO >>>
        if details:
            logger.info(f"--- VERIFICAÇÃO API --- Detalhes completos recebidos para {category_id}: {json.dumps(details, indent=2)}")
        else:
            logger.warning(f"--- VERIFICAÇÃO API --- Não foi possível obter detalhes para {category_id}.")
            
    except Exception as e:
        logger.error(f"Erro no wrapper cacheado get_category_details_cached para ID '{category_id}': {e}", exc_info=True)
        st.error(f"Erro ao buscar detalhes da categoria: {e}")
    finally:
        logger.info(f"Fechando cliente HTTP da busca de detalhes (wrapper) para ID: '{category_id}'")
        run_async(auth_details.close())
        logger.info(f"Cliente HTTP da busca de detalhes (wrapper) fechado para ID: '{category_id}'")
    return details

# --- Função auxiliar para rodar async dentro do Streamlit ---
# Streamlit tem seu próprio gerenciamento de loop, então rodar async requer cuidado.
# Para funções simples como a nossa, asyncio.run() dentro do manipulador de botão
# geralmente funciona, mas pode ser mais robusto usar get_event_loop().

def run_async(async_func):
    """Executa uma função assíncrona no loop de eventos do asyncio."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(async_func)

# --- Funções Auxiliares --- 
# Adicionar função para limpar URLs (apenas validação básica por enquanto)
def clean_urls(text_area_content: str) -> List[str]:
    """Limpa a entrada de URLs, removendo espaços e linhas vazias."""
    urls = []
    if text_area_content:
        lines = text_area_content.splitlines()
        for line in lines:
            cleaned_line = line.strip()
            # Validação muito básica: começa com http
            if cleaned_line and cleaned_line.lower().startswith('http'):
                urls.append(cleaned_line)
            elif cleaned_line: # Logar linha inválida se não estiver vazia
                logger.warning(f"Linha inválida ignorada na entrada de URLs: {cleaned_line}")
    return urls

# --- Inicializar session state --- 
if 'suggested_categories' not in st.session_state:
    st.session_state.suggested_categories = [] # Resultados da busca
if 'selected_category_id' not in st.session_state:
    st.session_state.selected_category_id = None # ID selecionado no selectbox
if 'selected_category_details' not in st.session_state:
     st.session_state.selected_category_details = None # Detalhes (com path) da categoria selecionada
if 'selected_category_max_title_len' not in st.session_state: # <<< Novo estado para guardar o limite
    st.session_state.selected_category_max_title_len = None
if 'search_query' not in st.session_state:
     st.session_state.search_query = ""
if 'market_research_result' not in st.session_state:
    st.session_state.market_research_result = None
if 'generated_ad_content' not in st.session_state: # <<< Novo estado para conteúdo gerado
    st.session_state.generated_ad_content = None
if 'trigger_category_search' not in st.session_state: # <<< NOVO: Sinalizador para busca
    st.session_state.trigger_category_search = False
if 'selected_llm_provider' not in st.session_state: # NOVO: Armazenar provedor LLM selecionado
    st.session_state.selected_llm_provider = "anthropic" # Valor padrão

# --- Callback para o Input de Categoria ---
def handle_category_search_change():
    """Callback acionado na mudança (Enter) do text_input de busca de categoria."""
    # Pega o valor atual do widget (a chave é 'category_search_input')
    current_value = st.session_state.get("category_search_input", "").strip()
    # Atualiza o estado principal da query
    st.session_state.search_query = current_value
    # Se o valor não for vazio, ativa o gatilho para buscar
    if current_value:
        st.session_state.trigger_category_search = True
    else:
        # Se o campo ficou vazio, limpa resultados e desativa gatilho
        st.session_state.trigger_category_search = False
        st.session_state.suggested_categories = []
        st.session_state.selected_category_id = None
        st.session_state.selected_category_details = None
        st.session_state.selected_category_max_title_len = None
    logger.info(f"Callback handle_category_search_change executado. Query: '{current_value}', Trigger: {st.session_state.trigger_category_search}")

# --- Formulário de Input ---
st.header("1. Informações do Produto e Concorrência Manual")

# Adicionar seletor de LLM na barra lateral
with st.sidebar:
    st.header("🤖 Configurações do Modelo")
    
    selected_provider = st.selectbox(
        "Selecione o Provedor de IA:",
        options=["anthropic", "openai", "groq"],
        format_func=lambda x: {
            "anthropic": "Claude (Anthropic)",
            "openai": "GPT (OpenAI)",
            "groq": "Llama (Groq)"
        }.get(x, x),
        index=["anthropic", "openai", "groq"].index(st.session_state.selected_llm_provider)
    )
    
    # Atualizar o estado da sessão quando o usuário mudar a seleção
    if selected_provider != st.session_state.selected_llm_provider:
        st.session_state.selected_llm_provider = selected_provider
        st.info(f"Provedor alterado para: {selected_provider.upper()}")
    
    # Exibir informações sobre os modelos
    with st.expander("📋 Informações sobre os modelos"):
        st.markdown("""
        - **Claude (Anthropic)**: Excelente para textos longos e criativos
        - **GPT (OpenAI)**: Bom equilíbrio entre criatividade e precisão técnica
        - **Llama (Groq)**: Alta velocidade com bom desempenho geral
        """)
    
    st.write("---")

# Seção de Busca de Categoria
st.subheader("Categoria do Produto*")

# Usar o callback on_change
st.text_input(
    "Pesquisar categoria (pressione Enter para buscar):",
    key="category_search_input", # A chave identifica o widget no session_state
    placeholder="Digite parte do nome, ex: Celular, Perfume, Camiseta...",
    help="Digite o nome ou tipo do produto e pressione Enter para buscar categorias.",
    on_change=handle_category_search_change # <<< Definir o callback
)

# --- Lógica de busca AGORA VERIFICA O SINALIZADOR ---
if st.session_state.trigger_category_search:
    query_to_search = st.session_state.search_query
    logger.info(f"Gatilho de busca ativo para query: '{query_to_search}'")
    
    with st.spinner(f"Buscando categorias sugeridas para '{query_to_search}'..."):
        try:
            # Chama a função wrapper cacheada síncrona com a query do estado
            suggested_cats = search_ml_categories_cached(query_to_search)
            
            logger.info(f"Função search_ml_categories_cached retornou: {len(suggested_cats)} categorias.")
            logger.debug(f"Categorias retornadas para '{query_to_search}': {suggested_cats}")
            
            st.session_state.suggested_categories = suggested_cats
            # Resetar seleção e detalhes anteriores ao buscar novas sugestões
            st.session_state.selected_category_id = None 
            st.session_state.selected_category_details = None
            st.session_state.selected_category_max_title_len = None
            
            if not suggested_cats:
                st.warning(f"Nenhuma categoria encontrada para '{query_to_search}'.")
                
        except Exception as e: 
             logger.error(f"Erro ao executar busca de categorias (trigger): {e}", exc_info=True)
             st.error(f"Erro ao buscar categorias para '{query_to_search}': {e}")
             st.session_state.suggested_categories = []
             
    # Desativa o gatilho após a busca ser concluída (ou falhar)
    st.session_state.trigger_category_search = False 
    logger.info(f"Gatilho de busca desativado após processar '{query_to_search}'.")

# Selectbox para escolher a categoria (só aparece se houver sugestões)
if st.session_state.suggested_categories:
    # Mapear ID para o dicionário completo retornado por search_ml_categories_cached
    cat_options = {cat['id']: cat for cat in st.session_state.suggested_categories}
    
    # Usar o ID da sessão se já existir E estiver nas opções atuais, senão default (None/primeiro)
    current_selection_id = st.session_state.selected_category_id
    if current_selection_id not in cat_options:
        current_selection_id = None 
        st.session_state.selected_category_id = current_selection_id
        st.session_state.selected_category_details = None 
        st.session_state.selected_category_max_title_len = None # <<< Limpa limite antigo
        
    selected_id = st.selectbox(
        f"Selecione a categoria sugerida ({len(st.session_state.suggested_categories)} encontradas):", 
        options=[None] + list(cat_options.keys()), # Adiciona opção None no início
        format_func=lambda key: "-- Selecione --" if key is None else f"{cat_options[key].get('path_string', 'Caminho N/A')} ({key})",
        key="category_selectbox",
        index=0 if current_selection_id is None else list(cat_options.keys()).index(current_selection_id) + 1 # Ajusta índice por causa do None
    )
    
    # Se a seleção mudou, atualiza o estado e busca detalhes
    if selected_id != st.session_state.selected_category_id:
        st.session_state.selected_category_id = selected_id
        st.session_state.selected_category_details = None # Limpa detalhes antigos
        st.session_state.selected_category_max_title_len = None # <<< Limpa limite antigo
        if selected_id:
             with st.spinner(f"Confirmando detalhes da categoria {selected_id}..."): 
                 try:
                     details = get_category_details_cached(selected_id)
                     st.session_state.selected_category_details = details
                     
                     # <<< Tentar extrair max_title_length >>>
                     if details:
                          settings = details.get('settings', {})
                          max_len = settings.get('max_title_length')
                          if isinstance(max_len, int):
                               st.session_state.selected_category_max_title_len = max_len
                               logger.info(f"Limite de caracteres do título para {selected_id} encontrado: {max_len}")
                               logger.warning(f"!!!!! IMPORTANTE !!!!! Limite máximo para títulos na categoria {selected_id}: {max_len} caracteres. Os títulos devem usar pelo menos 95% desse espaço.")
                          else:
                               logger.warning(f"Campo 'max_title_length' não encontrado ou inválido nas configurações da categoria {selected_id}.")
                               st.session_state.selected_category_max_title_len = 60 # Usar fallback
                          path_confirm = " > ".join([p.get('name', '?') for p in details.get('path_from_root', [])])
                          logger.info(f"Confirmação do caminho para {selected_id}: {path_confirm}")
                     else:
                          logger.warning(f"Não foi possível obter detalhes para {selected_id} após seleção.")
                          st.session_state.selected_category_max_title_len = 60 # Usar fallback
                          
                 except Exception as e:
                      logger.error(f"Erro ao buscar/confirmar detalhes da categoria {selected_id} (interface): {e}", exc_info=True)
                      st.error(f"Erro ao buscar detalhes da categoria: {e}")
                      st.session_state.selected_category_max_title_len = 60 # Usar fallback em caso de erro

# Exibir caminho da categoria selecionada (e talvez o limite encontrado)
selected_category_display = "Nenhuma categoria selecionada"
if st.session_state.selected_category_id and st.session_state.selected_category_details:
    cat_details = st.session_state.selected_category_details
    path = cat_details.get('path_from_root', [])
    path_str = " > ".join([p.get('name', '?') for p in path])
    # Usar o nome da categoria dos detalhes buscados para consistência
    cat_name_final = cat_details.get('name','?') 
    selected_category_display = f"{cat_name_final} ({st.session_state.selected_category_id})"
    st.success(f"**Caminho Confirmado:** {path_str}") 
    # Opcional: Exibir o limite encontrado
    if st.session_state.selected_category_max_title_len:
         st.caption(f"(Limite de caracteres para título nesta categoria: {st.session_state.selected_category_max_title_len})")
elif st.session_state.selected_category_id:
     # Se selecionou mas detalhes ainda não carregaram ou falharam
     selected_category_display = f"ID: {st.session_state.selected_category_id} (Confirmando detalhes...)"
     st.warning("Buscando ou não foi possível obter/confirmar o caminho completo da categoria.")

st.info(f"**Categoria Selecionada (Final):** {selected_category_display}")
st.markdown("---Produto Base---")
product_name_base = st.text_input("**Nome Base do Produto* (Obrigatório)**", placeholder="Ex: Camiseta Algodão Lisa", help="O nome principal do seu produto.")

# Campos opcionais
col1, col2 = st.columns(2)
with col1:
    brand = st.text_input("Marca", placeholder="Ex: Nike")
    ean = st.text_input("EAN (Código de Barras)", placeholder="Ex: 7891234567890")
with col2:
    model = st.text_input("Modelo", placeholder="Ex: Air Max 90")

# <<< NOVO CAMPO: Descrição Detalhada >>>
product_description_detailed = st.text_area(
    "Descrição Detalhada do Produto (Opcional)",
    placeholder="Inclua especificações técnicas, diferenciais, conteúdo do manual, etc.",
    height=150,
    help="Forneça o máximo de detalhes relevantes sobre seu produto para enriquecer a geração de conteúdo."
)

# Entrada para URLs de concorrentes manuais
competitor_urls_text = st.text_area(
    "**URLs de Concorrentes (1 por linha)**", 
    placeholder="https://produto.mercadolivre.com.br/MLB-xxxxxxx-...\nhttps://produto.mercadolivre.com.br/MLB-yyyyyyy-...\nhttps://produto.mercadolivre.com.br/MLB-zzzzzzz-...",
    height=150,
    help="Cole as URLs completas dos anúncios concorrentes que você quer analisar."
)

submit_button = st.button("Gerar Sugestões")

# --- Processamento e Output ---
if submit_button:
    # Validação básica - Usa st.session_state.selected_category_id
    final_category_id = st.session_state.selected_category_id
    if not final_category_id or not product_name_base:
        st.error("Por favor, busque e selecione uma categoria final e preencha o Nome Base do Produto.") # Mensagem ajustada
        logger.warning("Tentativa de submissão sem categoria final selecionada ou nome base.")
    else:
        st.info("Coletando dados e iniciando processo...")
        logger.info("Botão 'Gerar Sugestões' clicado. Validando e coletando input.")

        manual_urls = clean_urls(competitor_urls_text)

        product_input = ProductInput(
            category_id=final_category_id, 
            product_name_base=product_name_base.strip(),
            brand=brand.strip() if brand else None,
            model=model.strip() if model else None,
            ean=ean.strip() if ean else None,
            manual_competitor_urls=manual_urls, 
            detailed_description=product_description_detailed.strip() if product_description_detailed else None
        )

        st.subheader("Dados Coletados (para confirmação):")
        with st.expander("Ver Dados Coletados", expanded=False):
            cat_name = "N/A"
            if st.session_state.selected_category_details:
                cat_name = st.session_state.selected_category_details.get('name', 'N/A')
            
            st.json({
                "Modelo de IA": st.session_state.selected_llm_provider,
                "Categoria Selecionada": f"{cat_name} ({final_category_id})",
                "Nome Base": product_input.product_name_base,
                "Marca": product_input.brand,
                "Modelo": product_input.model,
                "EAN": product_input.ean,
                "Descrição Detalhada": product_input.detailed_description, 
                "URLs Concorrentes Manuais": product_input.manual_competitor_urls
            })

        # --- Executar Pesquisa de Mercado ---
        auth_process = MeliAuth()
        market_research_result = None
        st.session_state.market_research_result = None 
        st.session_state.generated_ad_content = None 
        
        with st.spinner("Buscando tendências, atributos e analisando concorrentes manuais... Aguarde!"):
            try:
                market_research_result = run_async(
                    fetch_market_data(auth_process, product_input)
                )
                st.session_state.market_research_result = market_research_result
                logger.info("Pesquisa de mercado retornou dados.")
            except Exception as e:
                logger.error(f"Erro ao executar pesquisa de mercado: {e}", exc_info=True)
                st.error(f"Ocorreu um erro durante a pesquisa de mercado: {e}")
            finally:
                logger.info("Fechando cliente HTTP após pesquisa...")
                run_async(auth_process.close())
                logger.info("Cliente HTTP fechado.")

        # --- Exibir Resultados da Pesquisa de Mercado ---
        st.subheader("3. Pesquisa de Mercado")
        if market_research_result:
            st.success("Pesquisa de mercado concluída!")
            
            # Bloco de Tendências - Indentação corrigida
            st.subheader("📊 Tendências de Busca (Top 20)")
            if market_research_result.trends:
                trends_list = [t.get('keyword', 'N/A') for t in market_research_result.trends[:20]]
                st.table(trends_list)
            else:
                st.info("Nenhuma tendência de busca encontrada para esta categoria.")
            st.divider()
            
            # Bloco de Atributos - Indentação corrigida
            st.subheader("📝 Atributos da Categoria (Ficha Técnica Editável)")
            with st.expander("Ver Atributos da Categoria", expanded=False):
                if market_research_result.category_attributes:
                    editable_attributes_to_show = []
                    required_attributes_names = []
                    total_attributes = len(market_research_result.category_attributes)
                    editable_count = 0

                    for attr in market_research_result.category_attributes:
                        tags = attr.get('tags', {})
                        is_read_only = tags.get('read_only', False)
                        if is_read_only:
                            continue
                        
                        editable_count += 1
                        attr_name = attr.get('name', 'N/A')
                        attr_id = attr.get('id', 'N/A')
                        is_required = tags.get('required', False)
                        
                        display_str = f"{attr_name} ({attr_id})"
                        if is_required:
                            display_str += " [OBRIGATÓRIO]"
                            required_attributes_names.append(attr_name)
                        
                        editable_attributes_to_show.append(display_str)

                    if editable_attributes_to_show:
                        st.write(editable_attributes_to_show)
                        logger.info(f"Exibindo {editable_count} atributos editáveis de um total de {total_attributes}.")
                        if required_attributes_names:
                            st.info(f"**Atributos Obrigatórios:** {', '.join(required_attributes_names)}")
                    else:
                        st.info("Nenhum atributo editável encontrado para esta categoria.")
                else:
                    st.warning("Nenhum atributo encontrado para esta categoria (verifique o ID).")
            st.divider()

            # Bloco de Concorrência - Indentação corrigida
            st.header("4. Análise de Concorrência")
            if market_research_result.competitor_analysis:
                st.success(f"Análise de {len(market_research_result.competitor_analysis)} concorrentes disponível.")
                for i, competitor in enumerate(market_research_result.competitor_analysis):
                    with st.expander(f"Concorrente {i+1}: {competitor.mlb_id} - {competitor.title[:50]}...", expanded=False):
                        st.markdown(f"**Título:** {competitor.title}")
                        st.markdown(f"**Preço:** {competitor.price if competitor.price else 'N/A'}")
                        st.markdown("**Atributos:**")
                        st.json(competitor.attributes if competitor.attributes else {})
                        st.markdown("**Descrição:**")
                        st.text(competitor.description if competitor.description else "(Sem descrição)")
            else:
                st.info("Nenhum dado de concorrente foi obtido (verifique URLs manuais).")
            st.divider()
            
            # Bloco de Geração de Conteúdo - Indentação corrigida
            # Mostrar qual provedor está selecionado
            st.info(f"Modelo selecionado para geração: {st.session_state.selected_llm_provider.upper()}")
            
            logger.info("Iniciando a geração de conteúdo com IA (LangGraph)...") 
            with st.spinner("Gerando sugestões de títulos, descrição e atributos com IA... Aguarde!"):
                try:
                    # Importar dinamicamente o módulo content_generator para acessar o PROVIDER
                    import importlib
                    content_generator = importlib.import_module("src.agents.ad_creator.content_generator")
                    
                    # Definir o provedor selecionado A PARTIR DO SESSION STATE
                    original_provider = content_generator.PROVIDER
                    content_generator.PROVIDER = st.session_state.selected_llm_provider 
                    
                    # Log de alteração do provedor
                    logger.info(f"Alterando provedor LLM de {original_provider} para {content_generator.PROVIDER}")
                    
                    # Recriar as instâncias de LLM com o novo provedor
                    api_key_env_var = f"{content_generator.PROVIDER.upper()}_API_KEY"
                    llm_api_key = os.getenv(api_key_env_var)
                    
                    if not llm_api_key:
                        raise ValueError(f"Chave API '{api_key_env_var}' não encontrada. Configure esta variável de ambiente.")
                    
                    # Recriar instâncias LLM
                    from src.utils.llm_factory import LLMFactory
                    content_generator.llm_creative = LLMFactory.create_llm(
                        provider=content_generator.PROVIDER,
                        model_name=content_generator.MODEL_NAME,
                        temperature=content_generator.TEMPERATURE_CREATIVE,
                        api_key=llm_api_key
                    )
                    
                    content_generator.llm_analytical = LLMFactory.create_llm(
                        provider=content_generator.PROVIDER,
                        model_name=content_generator.MODEL_NAME,
                        temperature=content_generator.TEMPERATURE_ANALYTICAL,
                        api_key=llm_api_key
                    )
                    
                    # Continuar com a geração
                    retrieved_max_len = st.session_state.get('selected_category_max_title_len')
                    max_len_to_pass = retrieved_max_len if retrieved_max_len is not None else 60
                    logger.info(f"Valor final de max_len_to_pass a ser enviado para o grafo: {max_len_to_pass}")

                    generated_content = generate_ad_content_graph(product_input, market_research_result, max_len_to_pass)
                    
                    st.session_state.generated_ad_content = generated_content
                    if generated_content:
                        logger.info("Geração de conteúdo (grafo) concluída com sucesso.")
                except Exception as e:
                    logger.error(f"Erro ao executar geração de conteúdo (grafo): {e}", exc_info=True)
                    st.error(f"Ocorreu um erro durante a geração de conteúdo: {e}")
                    st.session_state.generated_ad_content = None
        
        # --- Fim do Bloco if market_research_result --- 
        else: 
            st.error("Não foi possível realizar a pesquisa de mercado. Geração de conteúdo abortada.")

    # --- Fim do Bloco else (após validação inicial do botão submit) --- 

# --- Exibir Conteúdo Gerado (Fora do if submit_button, mas usa session_state) --- 
# Este bloco é exibido SEMPRE que a página recarrega se houver conteúdo gerado no estado
st.subheader("5. Sugestões Geradas pela IA")
generated_output = st.session_state.get('generated_ad_content') # Usar .get() para segurança
if generated_output:
    st.success("Sugestões geradas!")
    
    # Exibir Títulos
    st.markdown("**Títulos Sugeridos:**")
    if generated_output.suggested_titles:
        for i, title in enumerate(generated_output.suggested_titles):
            char_count = len(title)
            max_title_len = st.session_state.get('selected_category_max_title_len', 60)
            percentage = (char_count / max_title_len) * 100 if max_title_len > 0 else 0
            
            # Determinar cor de exibição com base na % do limite utilizado
            color = "green" if percentage >= 90 else "orange" if percentage >= 75 else "red"
            
            st.write(f"{i+1}. {title}")
            st.caption(f"<span style='color:{color}'>Caracteres: {char_count}/{max_title_len} ({percentage:.1f}%)</span>", unsafe_allow_html=True)
    else:
        st.write("Nenhuma sugestão de título gerada.")
    st.divider()
    
    # Exibir Descrição
    st.markdown("**Descrição Sugerida:**")
    if generated_output.suggested_description:
        st.text_area(
            label="Descrição Sugerida (Conteúdo)", 
            value=generated_output.suggested_description, 
            height=300, 
            key="suggested_desc_output", 
            label_visibility="collapsed"
        )
    else:
        st.write("Nenhuma sugestão de descrição gerada.")
    st.divider()

    # Exibir Atributos
    st.markdown("**Sugestões para Atributos Chave:**")
    
    # <<< Log de Verificação >>>
    attributes_data = getattr(generated_output, 'attributes', None) 
    logger.debug(f"Verificando 'attributes' para exibição. Tipo: {type(attributes_data)}, Conteúdo: {attributes_data}")
    
    # Usar a variável attributes_data na verificação e exibição
    if attributes_data: 
        try:
            # Tentar converter para DataFrame
            attr_data = [{'Atributo': k, 'Sugestão': v if v else "(vazio)"} 
                         for k, v in attributes_data.items()] # Usar attributes_data
            st.dataframe(attr_data, use_container_width=True)
        except Exception as df_e:
            logger.error(f"Erro ao tentar exibir atributos como DataFrame: {df_e}")
            st.warning("Não foi possível exibir sugestões de atributos como tabela. Exibindo como JSON:")
            st.json(attributes_data) # Fallback para JSON usando attributes_data
    else:
        st.write("Nenhuma sugestão para atributos chave gerada.")
    st.divider()

    # Exibir NCM Sugerido
    st.subheader("🔢 Sugestões de NCM (Rankeadas por Confiança)")
    ncm_suggestions = generated_output.ncm_suggestions

    if not ncm_suggestions:
        st.info("Nenhuma sugestão de NCM foi gerada.")
    else:
        for i, suggestion in enumerate(ncm_suggestions):
            rank = i + 1
            confidence_emoji = {"Alta": "🟢", "Média": "🟡", "Baixa": "🔴"}.get(suggestion.confidence, "⚪️")
            
            # Exibir NCM, Confiança e Explicação
            col1, col2 = st.columns([1, 4])
            with col1:
                 st.metric(label=f"#{rank} NCM Sugerido", value=suggestion.ncm_code) 
            with col2:
                 st.write(f"**Confiança:** {confidence_emoji} {suggestion.confidence}")
                 st.caption(f"**Justificativa:** {suggestion.explanation}")
                 
            # Separador entre sugestões, exceto a última
            if rank < len(ncm_suggestions):
                 st.divider()
            

    st.divider()

    # Adicionar um rodapé ou informações extras se desejar
    st.divider()
    st.caption("Desenvolvido com a ajuda de um assistente IA.") 