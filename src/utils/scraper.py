import httpx
from bs4 import BeautifulSoup
import json
import re
from typing import Optional, Dict, Any
import logging
import time
import random
import asyncio

logger = logging.getLogger(__name__)

# Definir seletores para encontrar as especificações/características
SPEC_SELECTORS = [
    "div.ui-pdp-specs__table", # <<< Layout principal comum
    "div.ui-vpp-striped-specs__table", # <<< Layout de Variações?
    "div.ui-vpp-highlighted-specs__attribute-columns", # <<< NOVO SELETOR para características destacadas
    # Adicionar outros seletores DIV se identificados
]
TABLE_SELECTORS = [
    "table.andes-table", # <<< Tabelas padrão
    # Adicionar outros seletores TABLE se identificados
]

async def _extract_attributes_directly(soup: BeautifulSoup, url: str) -> Dict[str, str]:
    """Tenta extrair atributos da tabela de especificações e itens destacados, 
       buscando tabelas dentro de contêineres de especificações conhecidos."""
    attributes = {}
    processed_tables = set() # Para evitar processar a mesma tabela múltiplas vezes

    # Abordagem 1: Itens destacados (ui-pdp-highlights__item) - Mantida, parece capturar "Características Principais"
    logger.debug(f"[{url}] Iniciando Etapa 1: Busca por Itens Destacados (ui-pdp-highlights__item).")
    highlighted_items = soup.find_all('li', class_='ui-pdp-highlights__item')
    if highlighted_items:
        logger.debug(f"[{url}] Etapa 1: Encontrados {len(highlighted_items)} itens destacados.")
        for item in highlighted_items:
            # Extrair key/value dos spans internos (lógica pode precisar de ajuste fino se variar)
            spans = item.find_all('span')
            if len(spans) >= 2:
                 key = spans[0].get_text(strip=True).rstrip(':').strip()
                 value = spans[1].get_text(strip=True)
                 if key and value and key not in attributes:
                      attributes[key] = value
                      logger.debug(f"[{url}] Etapa 1: Atributo '{key}' = '{value}' adicionado (item destacado).")
    else:
        logger.debug(f"[{url}] Etapa 1: Itens destacados (ui-pdp-highlights__item) NÃO encontrados.")

    # Abordagem 2: Buscar tabelas DENTRO de contêineres de especificações conhecidos
    logger.debug(f"[{url}] Iniciando Etapa 2: Busca por tabelas dentro de contêineres de specs {SPEC_SELECTORS}.")
    spec_containers_found = 0
    for spec_selector in SPEC_SELECTORS:
        containers = soup.select(spec_selector)
        if containers:
            logger.debug(f"[{url}] Etapa 2: Encontrados {len(containers)} contêiner(es) com seletor '{spec_selector}'.")
            spec_containers_found += len(containers)
            for container in containers:
                 # Buscar tabelas (priorizar andes-table, mas aceitar qualquer table como fallback dentro do container)
                 tables_in_container = container.find_all('table', class_='andes-table')
                 if not tables_in_container:
                      tables_in_container = container.find_all('table') 
                      if tables_in_container:
                           logger.debug(f"[{url}] Etapa 2: Nenhuma 'andes-table' encontrada em '{spec_selector}', buscando qualquer 'table'...")
                 
                 logger.debug(f"[{url}] Etapa 2: Encontradas {len(tables_in_container)} tabela(s) dentro de '{spec_selector}'.")
                 for table in tables_in_container:
                     # Verificar se já processamos esta tabela para evitar duplicatas
                     table_tuple = tuple(str(row) for row in table.find_all('tr')) # Forma de identificar a tabela
                     if table_tuple in processed_tables:
                         logger.debug(f"[{url}] Etapa 2: Pulando tabela já processada dentro de '{spec_selector}'.")
                         continue
                     processed_tables.add(table_tuple)
                     
                     rows = table.find_all('tr')
                     logger.debug(f"[{url}] Etapa 2: Processando tabela com {len(rows)} linhas dentro de '{spec_selector}'.")
                     for row in rows:
                         cells = row.find_all(['th', 'td'])
                         if len(cells) >= 2:
                             key_tag = cells[0]
                             value_tag = cells[1]
                             key = key_tag.get_text(strip=True)
                             # Tentar pegar valor de span específico ou do td inteiro
                             value_span = value_tag.find('span', class_='andes-table__column--value') 
                             value = value_span.get_text(strip=True) if value_span else value_tag.get_text(strip=True)
                             if key and value and key not in attributes: # Adicionar APENAS se a chave não existe
                                 attributes[key] = value
                                 logger.debug(f"[{url}] Etapa 2: Atributo '{key}' = '{value}' adicionado (tabela em '{spec_selector}').")
        else:
             logger.debug(f"[{url}] Etapa 2: Nenhum contêiner encontrado com seletor '{spec_selector}'.")
             
    if spec_containers_found == 0:
        logger.warning(f"[{url}] Etapa 2: Nenhum contêiner de especificações conhecido ({SPEC_SELECTORS}) encontrado na página.")

    # Log final
    logger.debug(f"[{url}] Extração de atributos (Direta) concluída. Total final: {len(attributes)} atributos. Resultado: {attributes if attributes else '{}'}")
    return attributes

async def scrape_product_page(session: httpx.AsyncClient, url: str, max_retries: int = 1, initial_delay: float = 0.5) -> Dict[str, Any]:
    """Faz o scraping da página de um produto no Mercado Livre com retries e delays."""
    product_data = {
        "mlb_id": extract_mlb_id_from_url(url),
        "title": "",
        "price": None,
        "attributes": {},
        "description": ""
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for attempt in range(max_retries + 1):
        try:
            # Adiciona um delay crescente e aleatório antes de cada tentativa (exceto a primeira talvez)
            if attempt > 0:
                delay = initial_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                logger.warning(f"[{url}] Tentativa {attempt + 1}/{max_retries + 1} de scraping após falha anterior. Aguardando {delay:.2f}s...")
                await asyncio.sleep(delay)
            else: 
                # Pequeno delay aleatório inicial mesmo na primeira tentativa?
                await asyncio.sleep(random.uniform(0.2, initial_delay))
                
            response = await session.get(url, headers=headers, timeout=20.0) # Aumentar timeout?
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extrair título
            title_tag = soup.find('h1', class_='ui-pdp-title')
            product_data["title"] = title_tag.text.strip() if title_tag else "Título não encontrado"

            # Extrair preço
            price_tag = soup.find('span', class_='andes-money-amount__fraction')
            cents_tag = soup.find('span', class_='andes-money-amount__cents')
            if price_tag:
                price_str = price_tag.text.strip().replace('.', '')
                if cents_tag:
                    price_str += '.' + cents_tag.text.strip()
                try:
                    product_data["price"] = float(price_str)
                except ValueError:
                    logger.warning(f"[{url}] Não foi possível converter o preço '{price_str}' para float.")
                    product_data["price"] = None
            else:
                 product_data["price"] = None

            # Extrair descrição
            desc_tag = soup.find('p', class_='ui-pdp-description__content')
            product_data["description"] = desc_tag.text.strip() if desc_tag else ""

            # Extrair atributos diretamente
            logger.info(f"[{url}] Iniciando extração de atributos (Abordagem Direta - Tentativa {attempt + 1})...")
            product_data["attributes"] = await _extract_attributes_directly(soup, url)
            
            # Se conseguiu atributos OU se foi a última tentativa, retorna o que tem
            if product_data["attributes"] or attempt == max_retries:
                if not product_data["attributes"] and attempt == max_retries:
                     logger.error(f"[{url}] ATENÇÃO: Falha em extrair QUALQUER atributo após {max_retries + 1} tentativas.")
                break # Sai do loop de retries

        except httpx.HTTPStatusError as e:
            logger.error(f"[{url}] Erro HTTP {e.response.status_code} ao acessar a URL na tentativa {attempt + 1}.", exc_info=True)
            if attempt == max_retries: # Se erro na última tentativa, desiste
                logger.error(f"[{url}] Desistindo após {max_retries + 1} tentativas com erro HTTP.")
                break 
            # Continuar para a próxima tentativa
        except httpx.RequestError as e:
            logger.error(f"[{url}] Erro de requisição ao acessar a URL na tentativa {attempt + 1}: {e}", exc_info=True)
            if attempt == max_retries: # Se erro na última tentativa, desiste
                logger.error(f"[{url}] Desistindo após {max_retries + 1} tentativas com erro de requisição.")
                break
            # Continuar para a próxima tentativa
        except Exception as e:
            logger.error(f"[{url}] Erro inesperado durante o scraping na tentativa {attempt + 1}: {e}", exc_info=True)
            if attempt == max_retries: # Se erro na última tentativa, desiste
                 logger.error(f"[{url}] Desistindo após {max_retries + 1} tentativas com erro inesperado.")
                 break
            # Continuar para a próxima tentativa
            
    logger.info(f"Scraping da URL {url} concluído. Título: '{product_data['title'][:30]}...', Preço: {product_data['price']}")
    return product_data

def extract_mlb_id_from_url(url: str) -> Optional[str]:
    """Extrai o ID do anúncio (MLBxxxx) de uma URL do Mercado Livre."""
    # Padrão para MLB seguido por números
    match = re.search(r'(MLB-?\d+)', url, re.IGNORECASE)
    if match:
        # Remove o hífen se presente para padronizar com IDs da API
        return match.group(1).replace('-', '')
    logger.warning(f"Não foi possível extrair o MLB ID da URL: {url}")
    return None 