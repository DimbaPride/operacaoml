# -*- coding: utf-8 -*-
# FINALIDADE: Realiza pesquisas na API do Mercado Livre (tendências, atributos, concorrentes).
# Módulo responsável pela pesquisa de mercado (Tendências, Atributos, Concorrentes) 

import asyncio
import logging
from typing import Optional, List, Dict, Any
import httpx # Para error handling
import urllib.parse

# Importar de locais relativos dentro do projeto
try:
    from src.api.auth import MeliAuth
    from .data_models import ProductInput, MarketResearchOutput, CompetitorInfo
    from src.utils.scraper import scrape_product_page, extract_mlb_id_from_url # Novo import
except ImportError as e:
    # Logar o erro aqui é dificil pois o logging pode não estar configurado ainda
    # quando este módulo for importado. Lançar o erro é mais robusto.
    print(f"Erro fatal ao importar módulos em market_research.py: {e}")
    raise e

# Obter um logger para este módulo
logger = logging.getLogger(__name__)

# Constante para o Site ID do Brasil
SITE_ID = "MLB"
# Limite padrão para busca de concorrentes
DEFAULT_COMPETITOR_LIMIT = 5

async def fetch_category_trends(auth: MeliAuth, category_id: str) -> List[Dict[str, Any]]:
    """Busca as tendências para uma categoria específica."""
    url = f"https://api.mercadolibre.com/trends/{SITE_ID}/{category_id}"
    logger.info(f"Buscando tendências para Categoria ID: {category_id} na URL: {url}")

    try:
        access_token = await auth.get_access_token()
        if not access_token:
            logger.error("Não foi possível obter access token para buscar tendências.")
            return [] # Retorna lista vazia em caso de falha de auth

        headers = {"Authorization": f"Bearer {access_token}"}
        response = await auth.client.get(url, headers=headers)

        if response.status_code == 200:
            logger.info(f"Tendências para {category_id} obtidas com sucesso.")
            trends_data = response.json()
            return trends_data if isinstance(trends_data, list) else []
        elif response.status_code == 404:
            logger.warning(f"Categoria {category_id} não encontrada para tendências (404). {response.text}")
            return []
        elif response.status_code == 403:
            logger.error(f"Não autorizado (403) para buscar tendências de {category_id}. {response.text}")
            return []
        else:
            logger.error(f"Erro ao buscar tendências para {category_id}: Status {response.status_code}. {response.text}")
            return []
    except httpx.RequestError as e:
        logger.error(f"Erro de conexão ao buscar tendências para {category_id}: {e}")
        return []
    except Exception as e:
        logger.exception(f"Erro inesperado ao buscar tendências para {category_id}: {e}")
        return []

async def fetch_category_attributes(auth: MeliAuth, category_id: str) -> List[Dict[str, Any]]:
    """Busca os atributos (ficha técnica) de uma categoria específica."""
    url = f"https://api.mercadolibre.com/categories/{category_id}/attributes"
    logger.info(f"Buscando atributos para Categoria ID: {category_id} na URL: {url}")

    try:
        # Geralmente este endpoint não requer autenticação, mas usá-la não prejudica
        # e garante consistência caso mude no futuro.
        access_token = await auth.get_access_token()
        if not access_token:
            logger.warning("Não foi possível obter access token para buscar atributos (continuando sem auth).")
            headers = {}
        else:
            headers = {"Authorization": f"Bearer {access_token}"}

        response = await auth.client.get(url, headers=headers)

        if response.status_code == 200:
            logger.info(f"Atributos para {category_id} obtidos com sucesso.")
            attributes_data = response.json()
            return attributes_data if isinstance(attributes_data, list) else []
        elif response.status_code == 404:
            logger.warning(f"Categoria {category_id} não encontrada para atributos (404). {response.text}")
            return []
        else:
            logger.error(f"Erro ao buscar atributos para {category_id}: Status {response.status_code}. {response.text}")
            return []
    except httpx.RequestError as e:
        logger.error(f"Erro de conexão ao buscar atributos para {category_id}: {e}")
        return []
    except Exception as e:
        logger.exception(f"Erro inesperado ao buscar atributos para {category_id}: {e}")
        return []

async def fetch_market_data(auth: MeliAuth, product_input: ProductInput) -> MarketResearchOutput:
    """Função principal para orquestrar a pesquisa de mercado (tendências, atributos, concorrentes MANUAIS)."""
    logger.info(f"Iniciando pesquisa de mercado para Categoria: {product_input.category_id}, Produto Base: '{product_input.product_name_base}'")

    # --- Obter URLs manuais --- 
    manual_urls = product_input.manual_competitor_urls
    logger.info(f"URLs de concorrentes manuais recebidas: {len(manual_urls)} URLs {manual_urls if manual_urls else 'Nenhuma'}")

    # <<< USAR APENAS URLs MANUAIS >>>
    urls_to_scrape = manual_urls

    if not urls_to_scrape:
        logger.warning("Nenhuma URL de concorrente manual fornecida para analisar.") # <<< Mensagem ajustada

    # --- Preparar todas as tarefas assíncronas --- 
    tasks = {
        "trends": fetch_category_trends(auth, product_input.category_id),
        "attributes": fetch_category_attributes(auth, product_input.category_id),
        "scraped_competitors": [] 
    }

    # Criar tarefas para fazer scraping de cada URL MANUAL
    tasks["scraped_competitors"] = [
        scrape_product_page(auth.client, url) for url in urls_to_scrape 
    ] 

    # --- Executar tarefas em paralelo --- 
    logger.info(f"Iniciando busca de tendências, atributos e scraping de {len(urls_to_scrape)} concorrentes manuais...") # <<< Log ajustado
    trends_results, attributes_results, *scraped_data_list = await asyncio.gather(
        tasks["trends"],
        tasks["attributes"],
        *tasks["scraped_competitors"], 
        return_exceptions=True 
    )
    
    # Verificar erros nas tarefas principais (trends, attributes)
    if isinstance(trends_results, Exception):
        logger.error(f"Erro ao buscar tendências durante gather: {trends_results}")
        trends_results = [] # Fallback para lista vazia
    if isinstance(attributes_results, Exception):
        logger.error(f"Erro ao buscar atributos durante gather: {attributes_results}")
        attributes_results = [] # Fallback para lista vazia

    # Mapear resultados do scraping de volta às URLs MANUAIS
    final_competitor_analysis: List[CompetitorInfo] = []
    for i, url in enumerate(urls_to_scrape): # Iterar sobre a lista manual
        scraped_data = scraped_data_list[i]
        
        if isinstance(scraped_data, dict): 
            mlb_id = extract_mlb_id_from_url(url) 
            if not mlb_id:
                mlb_id = f"SCRAPED_{i}"
                logger.warning(f"Não foi possível extrair MLB ID da URL {url}, usando ID fallback: {mlb_id}")

            # Usar as chaves CORRETAS retornadas pelo scraper
            scraped_attributes = scraped_data.get('attributes', {})
            logger.debug(f"Atributos encontrados pelo scraper para {url}: {scraped_attributes}")
            
            competitor = CompetitorInfo(
                mlb_id=mlb_id,
                title=scraped_data.get('title', 'N/A'),
                price=scraped_data.get('price'),
                attributes=scraped_attributes,
                description=scraped_data.get('description', ''),
                listing_type_id=None, 
                shipping_mode=None,
                free_shipping=False
            )
            final_competitor_analysis.append(competitor)
            logger.debug(f"Dados do scraping da URL {url} processados para o concorrente {mlb_id}.")

        elif isinstance(scraped_data, Exception):
            logger.error(f"Erro inesperado na tarefa de scraping para a URL {url}: {scraped_data}")
        else: 
             logger.warning(f"Falha no scraping da URL {url}. Será omitido da análise final.")
            
    logger.info(f"Scraping e processamento de concorrentes manuais concluído. {len(final_competitor_analysis)} concorrentes adicionados.") # <<< Log ajustado

    # --- Criar o objeto de output --- 
    market_data = MarketResearchOutput(
        trends=trends_results if isinstance(trends_results, list) else [],
        category_attributes=attributes_results if isinstance(attributes_results, list) else [],
        competitor_analysis=final_competitor_analysis # Usa a lista populada pelo scraper
    )

    logger.info("Pesquisa de mercado completa.")
    return market_data

# Exemplo de como usar (será chamado a partir do main_flow.py ou app_visual.py)
# async def main_test():
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     logger.info("Iniciando teste de market_research...")
#     auth = MeliAuth()
#     # Simular um input
#     test_input = ProductInput(
#         category_id="MLB6284", # Perfumes
#         product_name_base="Teste Perfume",
#         brand="TestBrand"
#     )
#     try:
#         market_info = await fetch_market_data(auth, test_input)
#         print("\n--- Resultados da Pesquisa ---")
#         print(f"Tendências ({len(market_info.trends)}): {market_info.trends[:5]}...") # Mostrar as 5 primeiras
#         print(f"Atributos ({len(market_info.category_attributes)}): {market_info.category_attributes[:3]}...") # Mostrar os 3 primeiros
#     finally:
#         await auth.close()
#         logger.info("Cliente HTTP fechado.")
#
# if __name__ == "__main__":
#     asyncio.run(main_test()) 