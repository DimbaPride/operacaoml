from .auth import MeliAuth
import logging

class PromotionsAPI:
    """API para gerenciar promoções no Mercado Livre"""
    
    def __init__(self, auth: MeliAuth):
        self.auth = auth
        self.logger = logging.getLogger(__name__)
    
    async def get_active_promotions(self, user_id=None):
        """
        Obtém todas as promoções ativas do vendedor
        """
        try:
            if not user_id:
                self.logger.error("ID do usuário não especificado para get_active_promotions")
                return []
                
            access_token = await self.auth.get_access_token()
            
            # Endpoint de promoções do vendedor
            url = f"https://api.mercadolibre.com/users/{user_id}/promotions"
            headers = {"Authorization": f"Bearer {access_token}"}
            
            self.logger.info(f"Consultando promoções ativas para o usuário {user_id}")
            
            response = await self.auth.client.get(url, headers=headers)
            
            if response.status_code != 200:
                self.logger.error(f"Erro ao consultar promoções: Status {response.status_code}")
                self.logger.error(f"Resposta: {response.text}")
                return []
                
            data = response.json()
            active_promotions = []
            
            # Filtrar apenas promoções ativas
            if isinstance(data, list):
                for promo in data:
                    if promo.get("status") in ["active", "started", "approved"]:
                        active_promotions.append(promo)
            
            self.logger.info(f"Encontradas {len(active_promotions)} promoções ativas")
            return active_promotions
        except Exception as e:
            self.logger.error(f"Erro ao obter promoções ativas: {str(e)}")
            return []
    
    async def get_items_in_promotion(self, promotion_id, promotion_type="DEAL"):
        """
        Obtém todos os itens de uma promoção específica.
        
        Args:
            promotion_id: ID da promoção
            promotion_type: Tipo da promoção (padrão DEAL)
            
        Returns:
            Lista de IDs de itens na promoção
        """
        try:
            if not promotion_id:
                self.logger.error("ID da promoção não especificado para get_items_in_promotion")
                return []
                
            access_token = await self.auth.get_access_token()
            
            # Endpoint para obter itens de uma promoção específica
            url = f"https://api.mercadolibre.com/seller-promotions/promotions/{promotion_id}/items"
            params = {
                "promotion_type": promotion_type,
                "app_version": "v2"
            }
            headers = {"Authorization": f"Bearer {access_token}"}
            
            self.logger.info(f"Consultando itens da promoção {promotion_id} (tipo: {promotion_type})")
            
            response = await self.auth.client.get(url, headers=headers, params=params)
            
            if response.status_code != 200:
                if response.status_code == 404:
                    self.logger.warning(f"Promoção {promotion_id} não encontrada ou sem itens")
                    return []
                    
                self.logger.error(f"Erro ao consultar itens da promoção: Status {response.status_code}")
                self.logger.error(f"Resposta: {response.text}")
                return []
                
            data = response.json()
            items = data.get("results", [])
            
            # Extrair apenas os IDs dos itens
            item_ids = [item.get("id") for item in items if item.get("id")]
            
            self.logger.info(f"Encontrados {len(item_ids)} itens na promoção {promotion_id}")
            return item_ids
        except Exception as e:
            self.logger.error(f"Erro ao obter itens da promoção {promotion_id}: {str(e)}")
            return []
    
    async def get_promotion_items(self, user_id=None):
        """
        Obter todos os itens em promoção para um vendedor
        """
        try:
            if not user_id:
                self.logger.error("ID do usuário não especificado para get_promotion_items")
                return []
                
            # Conjunto para armazenar os IDs únicos de itens em promoção
            promotion_item_ids = set()
            
            # Método 1: Obter promoções ativas e extrair itens
            active_promotions = await self.get_active_promotions(user_id)
            for promotion in active_promotions:
                items_in_promotion = await self.get_items_in_promotion(promotion.get('id'))
                promotion_item_ids.update(items_in_promotion)
            
            # Método 2: Tentar endpoints alternativos
            await self._try_alternative_promotion_endpoint(user_id, promotion_item_ids)
            
            self.logger.info(f"Total de {len(promotion_item_ids)} itens em promoção encontrados")
            return list(promotion_item_ids)
            
        except Exception as e:
            self.logger.error(f"Erro ao obter itens em promoção: {str(e)}")
            return []
    
    async def check_item_promotions(self, item_id):
        """
        Verifica se um item específico tem promoções ativas usando 
        o endpoint /seller-promotions/items/{ITEM_ID}
        
        Returns:
            Uma lista de promoções associadas ao item, ou lista vazia se não houver
        """
        try:
            if not item_id:
                self.logger.error("ID do item não especificado para check_item_promotions")
                return []
                
            access_token = await self.auth.get_access_token()
            
            # Endpoint específico para verificar promoções de um item
            url = f"https://api.mercadolibre.com/seller-promotions/items/{item_id}?app_version=v2"
            headers = {"Authorization": f"Bearer {access_token}"}
            
            self.logger.info(f"Verificando promoções para o item {item_id}")
            
            response = await self.auth.client.get(url, headers=headers)
            
            if response.status_code != 200:
                if response.status_code == 404:
                    # Item não encontrado ou sem promoções
                    return []
                    
                self.logger.warning(f"Erro ao verificar promoções para item {item_id}: Status {response.status_code}")
                self.logger.warning(f"Resposta: {response.text}")
                return []
                
            # Resposta contém uma lista de promoções do item
            promotions = response.json()
            self.logger.info(f"Item {item_id} tem {len(promotions)} promoções configuradas")
            
            return promotions
            
        except Exception as e:
            self.logger.error(f"Erro ao verificar promoções para item {item_id}: {str(e)}")
            return []
    
    async def _get_promotion_items_detail(self, promotion_id, promotion_type):
        """
        Obtém detalhes de itens em uma promoção específica
        """
        try:
            access_token = await self.auth.get_access_token()
            
            # URL varia dependendo do tipo de promoção
            url = f"https://api.mercadolibre.com/seller-promotions/{promotion_id}"
            
            if promotion_type == "deal":
                url = f"https://api.mercadolibre.com/deals/{promotion_id}"
            elif promotion_type == "campaign":
                url = f"https://api.mercadolibre.com/campaigns/{promotion_id}"
            
            headers = {"Authorization": f"Bearer {access_token}"}
            
            response = await self.auth.client.get(url, headers=headers)
            
            if response.status_code != 200:
                self.logger.error(f"Erro ao obter detalhes da promoção {promotion_id}: Status {response.status_code}")
                self.logger.error(f"Resposta: {response.text}")
                return {}
                
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Erro ao obter detalhes da promoção {promotion_id}: {str(e)}")
            return {}
    
    async def _try_alternative_promotion_endpoint(self, user_id, result_set):
        """
        Tenta um endpoint alternativo para obter itens em promoção
        e adiciona os resultados ao conjunto fornecido
        """
        try:
            access_token = await self.auth.get_access_token()
            
            # Endpoint alternativo para obter itens em promoção
            url = f"https://api.mercadolibre.com/users/{user_id}/items/promotions"
            headers = {"Authorization": f"Bearer {access_token}"}
            
            response = await self.auth.client.get(url, headers=headers)
            
            if response.status_code != 200:
                self.logger.warning(f"Endpoint alternativo retornou status {response.status_code}")
                return
                
            data = response.json()
            
            # O formato da resposta pode variar, extrair IDs de itens
            item_ids = self._extract_item_ids(data)
            
            self.logger.info(f"Endpoint alternativo: {len(item_ids)} itens encontrados")
            result_set.update(item_ids)
            
        except Exception as e:
            self.logger.error(f"Erro ao usar endpoint alternativo: {str(e)}")
    
    def _extract_item_ids(self, data):
        """
        Extrai todos os IDs de itens de uma resposta de API de promoção,
        independentemente do formato da resposta
        """
        item_ids = set()
        
        # Função recursiva para extrair IDs de qualquer estrutura de dados
        def extract_ids(obj):
            if isinstance(obj, dict):
                # Verificar se este dicionário contém um ID de item
                if "id" in obj:
                    item_id = obj["id"]
                    # Normalizar o ID (garantir que começa com MLB)
                    if isinstance(item_id, str):
                        if not item_id.startswith("MLB"):
                            item_id = f"MLB{item_id}"
                        item_ids.add(item_id)
                
                # Processar valores recursivamente
                for value in obj.values():
                    extract_ids(value)
                    
            elif isinstance(obj, list):
                # Processar cada item da lista
                for item in obj:
                    extract_ids(item)
        
        # Iniciar a extração recursiva
        extract_ids(data)
        return item_ids
