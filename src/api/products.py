from .auth import MeliAuth

class ProductsAPI:
    """API para gerenciar produtos no Mercado Livre"""
    
    def __init__(self, auth: MeliAuth):
        self.auth = auth
        self.user_id = None
    
    async def get_user_id(self):
        """Obtém o ID do usuário atual"""
        if self.user_id:
            return self.user_id
            
        try:
            access_token = await self.auth.get_access_token()
            
            url = "https://api.mercadolibre.com/users/me"
            headers = {"Authorization": f"Bearer {access_token}"}
            
            response = await self.auth.client.get(url, headers=headers)
            
            if response.status_code != 200:
                print(f"Erro ao obter informações do usuário: Status {response.status_code}")
                print(f"Resposta: {response.text}")
                return None
                
            data = response.json()
            self.user_id = data.get("id")
            print(f"ID do usuário obtido com sucesso: {self.user_id}")
            return self.user_id
        except Exception as e:
            print(f"Erro ao obter ID do usuário: {str(e)}")
            return None
    
    async def get_all_items(self, only_active=True, include_catalog=False):
        """
        Obtém todos os itens/anúncios do vendedor
        
        Args:
            only_active (bool): Se True, retorna apenas os anúncios ativos.
                               Se False, retorna todos os anúncios.
            include_catalog (bool): Se True, inclui anúncios de catálogo.
                                  Se False, filtra anúncios de catálogo.
        """
        try:
            # Obter o ID do usuário
            user_id = await self.get_user_id()
            if not user_id:
                print("Não foi possível obter o ID do usuário")
                return []
                
            access_token = await self.auth.get_access_token()
            
            # Usando o endpoint correto com o ID do usuário
            url = f"https://api.mercadolibre.com/users/{user_id}/items/search"
            headers = {"Authorization": f"Bearer {access_token}"}
            
            # Adicionar o filtro de status se solicitado
            params = {
                "limit": 100, 
                "offset": 0
            }
            
            if only_active:
                params["status"] = "active"
                print("Filtrando apenas anúncios ativos")
            
            item_search_results = []
            while True:
                try:
                    response = await self.auth.client.get(url, headers=headers, params=params)
                    
                    # Verificar se a resposta foi bem-sucedida
                    if response.status_code != 200:
                        print(f"Erro na API do Mercado Livre: Status {response.status_code}")
                        print(f"Resposta: {response.text}")
                        break
                    
                    data = response.json()
                    items = data.get("results", [])
                    if not items:
                        break
                        
                    item_search_results.extend(items)
                    params["offset"] += params["limit"]
                    if params["offset"] >= data.get("paging", {}).get("total", 0):
                        break
                except Exception as e:
                    print(f"Erro ao buscar página de itens: {str(e)}")
                    break
            
            # Obter detalhes completos dos itens para verificar se são de catálogo
            all_items_with_details = []
            
            # Criar batches para processamento mais rápido
            batch_size = 20
            total_items = len(item_search_results)
            
            print(f"Obtendo detalhes de {total_items} produtos...")
            
            for i in range(0, total_items, batch_size):
                batch = item_search_results[i:i+batch_size]
                batch_results = []
                
                for item_id in batch:
                    try:
                        # Obter detalhes completos do item
                        item_details = await self.get_item_details(item_id)
                        
                        # Verificar se é um item de catálogo
                        is_catalog = item_details.get("catalog_listing", False)
                        is_synchronized = False
                        synchronized_with = None
                        
                        # Verificar sincronização
                        if "attributes" in item_details:
                            for attr in item_details.get("attributes", []):
                                if attr.get("id") == "SYNCHRONIZED_WITH":
                                    is_synchronized = True
                                    synchronized_with = attr.get("value_name")
                        
                        # Decidir se incluir o item com base nos filtros
                        if include_catalog or not is_catalog:
                            # Adicionar informações extras para ajudar no processamento posterior
                            item_details["_is_catalog"] = is_catalog
                            item_details["_is_synchronized"] = is_synchronized
                            item_details["_synchronized_with"] = synchronized_with
                            
                            batch_results.append(item_details)
                            
                    except Exception as e:
                        print(f"Erro ao processar item {item_id}: {str(e)}")
                
                # Adicionar os resultados do batch à lista principal
                all_items_with_details.extend(batch_results)
                print(f"Processados {min(i+batch_size, total_items)}/{total_items} produtos...")
            
            # Contar tipos de itens para debug
            catalog_count = sum(1 for item in all_items_with_details if item.get("_is_catalog", False))
            synchronized_count = sum(1 for item in all_items_with_details if item.get("_is_synchronized", False))
            
            print(f"Total de itens obtidos: {len(all_items_with_details)}")
            print(f"Itens de catálogo: {catalog_count}")
            print(f"Itens sincronizados: {synchronized_count}")
            
            return all_items_with_details
        except Exception as e:
            print(f"Erro ao buscar todos os itens: {str(e)}")
            return []
    
    async def get_item_details(self, item_id):
        """Obtém detalhes de um item específico"""
        try:
            access_token = await self.auth.get_access_token()
            
            url = f"https://api.mercadolibre.com/items/{item_id}"
            headers = {"Authorization": f"Bearer {access_token}"}
            
            response = await self.auth.client.get(url, headers=headers)
            
            if response.status_code != 200:
                print(f"Erro ao obter detalhes do item {item_id}: Status {response.status_code}")
                return {"id": item_id, "error": response.text}
                
            return response.json()
        except Exception as e:
            print(f"Erro ao obter detalhes do item {item_id}: {str(e)}")
            return {"id": item_id, "error": str(e)}
    
    async def check_item_promotion_status(self, item_id):
        """
        Verifica o status de promoção de um item específico usando
        o endpoint direto de promoções
        """
        try:
            access_token = await self.auth.get_access_token()
            
            # Verificar status de promoção diretamente
            url = f"https://api.mercadolibre.com/items/{item_id}/promotion_status"
            headers = {"Authorization": f"Bearer {access_token}"}
            
            response = await self.auth.client.get(url, headers=headers)
            
            # Nem todos os itens têm este endpoint disponível, então não considerar erros como falha
            if response.status_code != 200:
                return {"id": item_id, "has_promotion": False, "reason": f"Status {response.status_code}"}
            
            data = response.json()
            
            # Verificar se o item tem alguma promoção ativa
            has_promotion = False
            promotion_type = None
            promotion_details = {}
            
            if "actions" in data and isinstance(data["actions"], list):
                for action in data["actions"]:
                    if action.get("status") == "active":
                        has_promotion = True
                        promotion_type = action.get("type")
                        promotion_details = action
                        break
            
            # Verificar também outros campos que indicam promoção
            if "has_discount" in data:
                has_discount = data.get("has_discount", False)
                if has_discount:
                    has_promotion = True
                    
            # Verificar se o item está em oferta
            if "offer" in data:
                offer_data = data.get("offer", {})
                if offer_data and offer_data.get("status") == "active":
                    has_promotion = True
            
            return {
                "id": item_id,
                "has_promotion": has_promotion,
                "promotion_type": promotion_type,
                "promotion_details": promotion_details,
                "raw_status": data
            }
        except Exception as e:
            print(f"Erro ao verificar status de promoção do item {item_id}: {str(e)}")
            return {"id": item_id, "has_promotion": False, "error": str(e)}
