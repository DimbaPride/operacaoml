import os
import sys
import csv
import argparse
import asyncio
import time
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

# Adicionar o diretório raiz do projeto ao PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.auth import MeliAuth
from src.api.products import ProductsAPI
from src.api.promotions import PromotionsAPI
from src.utils.llm_factory import LLMFactory, LLMProvider

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()

async def check_item_price_api(product_api, item):
    """
    Verifica se um item tem preços promocionais usando a API de preços.
    
    Returns:
        tuple: (item_details, is_promoted, promotion_flags)
    """
    item_id = item.get('id')
    if not item_id:
        return None, False, []
        
    try:
        # Obter detalhes completos do item
        item_details = await product_api.get_item_details(item_id)
        
        # Verificar os preços do item usando a API específica de preços
        access_token = await product_api.auth.get_access_token()
        url = f"https://api.mercadolibre.com/items/{item_id}/prices"
        headers = {"Authorization": f"Bearer {access_token}"}
        
        response = await product_api.auth.client.get(url, headers=headers)
        
        if response.status_code != 200:
            logger.warning(f"Erro ao obter preços para o item {item_id}: Status {response.status_code}")
            return item_details, False, []
            
        price_data = response.json()
        price_list = price_data.get('prices', [])
        
        # Verificar se há algum preço do tipo "promotion"
        has_promotion = any(price.get('type') == 'promotion' for price in price_list)
        
        promotion_flags = []
        if has_promotion:
            # Item está em promoção
            promotion_flags = ["price_api_promotion_type"]
            
            # Extrair detalhes da promoção (para logging)
            promo_prices = [p for p in price_list if p.get('type') == 'promotion']
            if promo_prices:
                first_promo = promo_prices[0]
                amount = first_promo.get('amount')
                regular_amount = first_promo.get('regular_amount')
                
                if amount and regular_amount:
                    discount_percentage = ((regular_amount - amount) / regular_amount) * 100
                    promotion_flags.append(f"discount_{discount_percentage:.1f}%")
        
        return item_details, has_promotion, promotion_flags
    except Exception as e:
        logger.error(f"Erro ao verificar preços para o item {item_id}: {str(e)}")
        return None, False, []

async def _identify_non_promoted_items_async(products_api, promotions_api):
    """
    Identifica itens ativos sem promoções, utilizando a API específica de preços
    do Mercado Livre para garantir alta precisão na detecção.
    
    Um produto sem promoção é aquele que possui apenas preços do tipo 'standard'
    e nenhum preço do tipo 'promotion'.
    """
    logger.info("Iniciando análise para identificar produtos sem promoções...")
    
    # Obter todos os itens ativos
    logger.info("Obtendo itens ativos...")
    active_items = await products_api.get_all_items(only_active=True, include_catalog=False)
    
    if not active_items:
        logger.warning("Nenhum item ativo encontrado.")
        return None
    
    # Filtrar apenas itens realmente ativos
    logger.info(f"Processando {len(active_items)} itens ativos...")
    active_listings = []
    
    for item in active_items:
        if item.get('status') == 'active':
            active_listings.append(item)
    
    if not active_listings:
        logger.warning("Nenhum item com status 'active' encontrado.")
        return None
    
    logger.info(f"Total de listagens realmente ativas: {len(active_listings)} produtos")
    
    # Inicializar listas para armazenar resultados
    promoted_items = []
    non_promoted_items = []
    
    # Processar itens em paralelo em lotes para melhor desempenho
    # mas evitando sobrecarregar a API
    batch_size = 10  # Ajuste conforme necessário
    logger.info(f"Verificando produtos em lotes de {batch_size}...")
    
    for i in range(0, len(active_listings), batch_size):
        batch = active_listings[i:i+batch_size]
        
        # Criar tarefas para verificar cada item no lote em paralelo
        tasks = [check_item_price_api(products_api, item) for item in batch]
        results = await asyncio.gather(*tasks)
        
        # Processar resultados do lote
        for item, (item_details, is_promoted, promotion_flags) in zip(batch, results):
            if not item_details:
                continue  # Pular itens que falharam na verificação
                
            if is_promoted:
                # Item está em promoção
                item_with_status = {
                    'item': item_details,
                    'promotion_status': True,
                    'promotion_flags': promotion_flags
                }
                promoted_items.append(item_with_status)
                
                logger.info(f"Produto em promoção: {item_details.get('title')} ({item.get('id')}) - "
                           f"Flags: {promotion_flags}")
            else:
                # Item não está em promoção
                item_with_status = {
                    'item': item_details,
                    'promotion_status': False
                }
                non_promoted_items.append(item_with_status)
                
                logger.info(f"Produto sem promoção: {item_details.get('title')} ({item.get('id')})")
    
    logger.info(f"Total de produtos ativos analisados: {len(active_listings)}")
    logger.info(f"Produtos em promoção pela API de preços: {len(promoted_items)}")
    logger.info(f"Produtos sem promoção pela API de preços: {len(non_promoted_items)}")
    
    # Dupla verificação usando o endpoint /seller-promotions/items/{item_id}
    # para itens que não foram identificados como promovidos pela API de preços
    logger.info("Realizando dupla verificação dos produtos sem promoção...")
    
    items_to_recategorize = []
    
    # Processar verificação em lotes para melhor desempenho
    for i in range(0, len(non_promoted_items), batch_size):
        batch = non_promoted_items[i:i+batch_size]
        verification_tasks = []
        
        for item_data in batch:
            item_id = item_data['item']['id']
            # Criar tarefa para verificar promoções
            verification_tasks.append(promotions_api.check_item_promotions(item_id))
        
        # Executar tarefas em paralelo
        promo_results = await asyncio.gather(*verification_tasks)
        
        # Analisar resultados
        for idx, (item_data, promotions) in enumerate(zip(batch, promo_results)):
            item_id = item_data['item']['id']
            try:
                # Verificar se há promoções REALMENTE ativas no momento
                has_active_promotion = False
                
                if promotions:
                    for promo in promotions:
                        # Verificar se a promoção está em estado ativo
                        if promo.get('status') not in ['started', 'active']:
                            continue
                        
                        # Verificar se a promoção está dentro da data válida
                        now = datetime.now().timestamp() * 1000  # Timestamp atual em milissegundos
                        date_from = promo.get('date_from', 0)
                        date_to = promo.get('date_to', float('inf'))
                        
                        if now < date_from or now > date_to:
                            continue
                            
                        # Verificar se a promoção tem um desconto real
                        price_details = promo.get('price_point', {})
                        if price_details:
                            original = price_details.get('original_price', 0)
                            promoted = price_details.get('promotional_price', 0)
                            
                            # Se houver um desconto real de pelo menos 1%
                            if original > 0 and promoted > 0 and original > promoted:
                                discount_pct = ((original - promoted) / original) * 100
                                if discount_pct >= 1.0:  # Pelo menos 1% de desconto
                                    has_active_promotion = True
                                    logger.info(f"Item {item_id} tem promoção ativa com {discount_pct:.1f}% de desconto")
                                    break
                        
                        # Se não encontrou detalhes de preço mas a promoção está ativa
                        # Verificar se a promoção é limitada por quantidade
                        quantity_info = promo.get('quantity_info', {})
                        if quantity_info:
                            available = quantity_info.get('available', 0)
                            if available > 0:  # Se ainda tem disponibilidade
                                has_active_promotion = True
                                logger.info(f"Item {item_id} tem promoção ativa com {available} itens disponíveis")
                                break
                
                if has_active_promotion:
                    logger.warning(f"Item {item_id} detectado como não promovido pela API de preços, "
                                 f"mas tem promoções ativas segundo a API de promoções. Recategorizando.")
                    
                    # Adicionar à lista de itens para recategorizar
                    items_to_recategorize.append((i + idx, item_data))
                else:
                    logger.info(f"Item {item_id} não tem promoções realmente ativas no momento")
            except Exception as e:
                logger.error(f"Erro na verificação para o item {item_id}: {str(e)}")
    
    # Recategorizar itens de trás para frente para não afetar índices
    for idx, item_data in sorted(items_to_recategorize, key=lambda x: x[0], reverse=True):
        try:
            # Remover da lista de não promovidos
            non_promoted_items.pop(idx)
            
            # Adicionar à lista de promovidos com a flag apropriada
            item_data['promotion_status'] = True
            item_data['promotion_flags'] = ["seller_promotions_api"]
            promoted_items.append(item_data)
        except Exception as e:
            logger.error(f"Erro ao recategorizar item: {str(e)}")
    
    logger.info(f"Após dupla verificação: {len(promoted_items)} produtos em promoção, {len(non_promoted_items)} sem promoção")
    
    return {
        'total_items': len(active_listings),
        'promoted_items': promoted_items,
        'non_promoted_items': non_promoted_items
    }

async def run_non_promotion_analysis():
    """Executa análise de produtos sem promoção e lista todos eles."""
    logger.info("Iniciando análise de produtos sem promoção...")
    
    try:
        start_time = time.time()
        
        # Inicializar APIs
        auth = MeliAuth()
        products_api = ProductsAPI(auth)
        promotions_api = PromotionsAPI(auth)
        
        # Identificar produtos sem promoção
        results = await _identify_non_promoted_items_async(products_api, promotions_api)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if not results:
            logger.error("Não foi possível obter resultados da análise.")
            return
        
        total_items = results['total_items']
        promoted_items = results['promoted_items']
        non_promoted_items = results['non_promoted_items']
        
        # Exibir estatísticas
        print("\n" + "="*50)
        print(f"Análise de Produtos sem Promoção")
        print("="*50)
        print(f"Total de produtos ativos: {total_items}")
        print(f"Produtos em promoção: {len(promoted_items)}")
        print(f"Produtos sem promoção: {len(non_promoted_items)}")
        print(f"Tempo de execução: {execution_time:.2f} segundos")
        print("="*50)
        
        # Exibir todos os produtos sem promoção
        if non_promoted_items:
            print("\nProdutos SEM promoção:")
            print(f"{'#':3} {'MLB':14} {'Preço':10} {'Título':50}")
            print("-"*80)
            
            for i, item in enumerate(non_promoted_items, 1):
                mlb = item['item']['id']
                title = item['item']['title']
                price = f"R$ {item['item']['price']}"
                print(f"{i:3d} {mlb:14} {price:10} {title[:50]}")
                
            # Exportar para CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"produtos_sem_promocao_{timestamp}.csv"
            
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['MLB', 'Título', 'Preço', 'URL']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for item in non_promoted_items:
                    writer.writerow({
                        'MLB': item['item']['id'],
                        'Título': item['item']['title'],
                        'Preço': item['item']['price'],
                        'URL': item['item'].get('permalink', '')
                    })
                print(f"\nResultados exportados para {csv_filename}")
        else:
            print("Nenhum produto sem promoção encontrado.")
            
        # Exibir amostra de produtos com promoção para verificação
        if promoted_items:
            print("\nAmostra de produtos COM promoção (para verificação):")
            print(f"{'#':3} {'MLB':14} {'Preço Atual':10} {'Preço Original':10} {'Desconto':10} {'Título':50}")
            print("-"*100)
            
            for i, item in enumerate(promoted_items[:5], 1):  # Mostrar até 5 exemplos
                product = item['item']
                mlb = product['id']
                title = product['title']
                current_price = product.get('price', 0)
                original_price = product.get('original_price', current_price)
                
                if original_price and original_price > current_price:
                    discount = ((original_price - current_price) / original_price) * 100
                    discount_str = f"{discount:.1f}%"
                else:
                    discount_str = "N/A"
                
                price_str = f"R$ {current_price}"
                original_price_str = f"R$ {original_price}" if original_price else "N/A"
                
                print(f"{i:3d} {mlb:14} {price_str:10} {original_price_str:10} {discount_str:10} {title[:50]}")
                
            print(f"\nFlags de promoção encontradas em produtos:")
            all_flags = set()
            for item in promoted_items:
                for flag in item.get('promotion_flags', []):
                    all_flags.add(flag)
            
            for flag in sorted(all_flags):
                print(f"- {flag}")
                
    except Exception as e:
        logger.error(f"Erro durante a análise de produtos: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Executa análise de produtos sem promoção')
    args = parser.parse_args()
    # Executar a função assíncrona
    asyncio.run(run_non_promotion_analysis())
    
if __name__ == "__main__":
    main()
