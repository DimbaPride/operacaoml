import os
import httpx
import time
import logging
from dotenv import load_dotenv, set_key
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class MeliAuth:
    """Classe para gerenciar autenticação com o Mercado Livre"""
    
    def __init__(self):
        # Carregar variáveis de ambiente AQUI, dentro do init
        # Adicionar override=True para garantir que valores do .env sobreponham os do sistema
        load_dotenv(override=True) 
        logger.info("Tentando carregar credenciais do .env dentro de MeliAuth.__init__")
        
        self.client_id = os.getenv("MELI_CLIENT_ID")
        self.client_secret = os.getenv("MELI_CLIENT_SECRET")
        self.refresh_token = os.getenv("MELI_REFRESH_TOKEN")
        self.redirect_uri = os.getenv("MELI_REDIRECT_URI")
        self.access_token = None # Iniciar como None
        self.client = httpx.AsyncClient(timeout=30.0)
        self.last_refresh_time = 0
        
        # Logar os valores lidos (com mais detalhes, mas ainda seguro)
        logger.info(f"MELI_APP_ID: {'Presente (' + self.client_id[:4] + '...)' if self.client_id else 'Ausente'}")
        logger.info(f"MELI_CLIENT_SECRET: {'Presente' if self.client_secret else 'Ausente'}") # Não logar valor
        logger.info(f"MELI_REFRESH_TOKEN: {'Presente (...' + self.refresh_token[-6:] + ')' if self.refresh_token else 'Ausente'}")
        logger.info(f"MELI_REDIRECT_URI: {'Presente (' + self.redirect_uri[:10] + '...)' if self.redirect_uri else 'Ausente'}")

        # Verificar se as credenciais estão presentes (usando self.)
        if not all([self.client_id, self.client_secret, self.redirect_uri, self.refresh_token]):
            # Mudar para ERROR ou WARNING em vez de CRITICAL se a app pode continuar sem auth
            logger.error("Uma ou mais credenciais da API do Mercado Livre (APP_ID, CLIENT_SECRET, REDIRECT_URI, REFRESH_TOKEN) não foram encontradas no .env.")
        else:
            logger.info("Credenciais de autenticação carregadas com sucesso.")

    async def close(self):
        """Fecha a conexão do cliente HTTP"""
        logger.info("Fechando cliente HTTP da MeliAuth...")
        await self.client.aclose()
        logger.info("Cliente HTTP da MeliAuth fechado.")

    def is_token_expired(self) -> bool:
        # Tokens do ML geralmente duram 6 horas (21600 segundos)
        # Verificar se passaram, por exemplo, 5.5 horas
        return (time.time() - self.last_refresh_time) > 19800 # 5.5 * 3600

    # Esta é a versão CORRETA de get_access_token
    async def get_access_token(self) -> Optional[str]: 
        if not self.access_token or self.is_token_expired():
            logger.info("Access token ausente ou expirado. Tentando obter novo token com refresh token...")
            # Chama a versão correta de refresh_access_token
            refreshed_token = await self.refresh_access_token()
            if refreshed_token:
                self.access_token = refreshed_token
                self.last_refresh_time = time.time()
                logger.info("Novo access token obtido e armazenado na instância.")
            else:
                logger.error("Falha ao obter novo access token usando refresh token.")
                self.access_token = None # Garante que não use o token antigo
                return None # Retorna None em caso de falha no refresh
        # else: # Log opcional para token válido
        #     logger.debug("Usando access token existente da instância.")
        return self.access_token

    # Esta é a versão CORRETA de refresh_access_token
    async def refresh_access_token(self) -> Optional[str]: 
        if not self.refresh_token:
            logger.error("Refresh token não está disponível para renovar o access token.")
            return None

        url = "https://api.mercadolibre.com/oauth/token"
        payload = {
            'grant_type': 'refresh_token',
            'client_id': self.client_id, 
            'client_secret': self.client_secret, 
            'refresh_token': self.refresh_token
        }
        # Comentar log do payload
        # logger.info(f"Payload a ser enviado para /oauth/token: {payload}")
        logger.info(f"Tentando renovar access token usando refresh token: ...{self.refresh_token[-6:]}")

        try:
            response = await self.client.post(url, data=payload)

            if response.status_code == 200:
                data = response.json()
                new_access_token = data.get('access_token')
                new_refresh_token = data.get('refresh_token') # Obter o NOVO refresh token
                expires_in = data.get('expires_in')
                logger.info(f"Token atualizado com sucesso. Expira em {expires_in} segundos.")

                # --- SALVAR O NOVO REFRESH TOKEN DE VOLTA NO .ENV --- 
                if new_refresh_token and new_refresh_token != self.refresh_token:
                    logger.info(f"Novo refresh token recebido: ...{new_refresh_token[-6:]}")
                    env_path = '.env' # Assume que o .env está na raiz do projeto
                    # Usar set_key da biblioteca python-dotenv
                    if set_key(env_path, 'MELI_REFRESH_TOKEN', new_refresh_token):
                        self.refresh_token = new_refresh_token # Atualiza o refresh token na instância
                        logger.info("Novo refresh token salvo com sucesso no arquivo .env.")
                    else:
                        # set_key retorna False se não conseguir encontrar/escrever no arquivo
                        logger.error(f"Falha ao salvar o novo refresh token no arquivo {env_path} usando set_key! Verifique permissões e se o arquivo existe.")
                elif not new_refresh_token:
                     logger.warning("API não retornou um novo refresh token na resposta.")
                # else: # Não precisa logar se for o mesmo
                     # logger.info("Refresh token retornado pela API é o mesmo que já tínhamos.")
                # -------------------------------------------------------
                
                return new_access_token
            else:
                error_details = ""
                try:
                    error_data = response.json()
                    error_type = error_data.get('error', '')
                    # Usar 'message' que parece ser o padrão na resposta de erro do ML
                    error_msg = error_data.get('message', response.text) # Fallback para texto completo
                    if error_type:
                        error_details = f": {error_type}{' - ' + error_msg if error_msg else ''}"
                    elif error_msg:
                         error_details = f": {error_msg}"
                except Exception:
                    error_details = f": {response.text}" # Fallback se JSON falhar
                
                logger.error(f"Falha ao renovar token (HTTP {response.status_code}){error_details}") 
                # Não logar payload completo por segurança, apenas que a tentativa foi feita
                # logger.error(f"Falha ao renovar token (HTTP {response.status_code}){error_details}. Payload enviado: {payload}. Resposta: {response.text}") 
                return None # Retorna None para indicar falha

        except httpx.RequestError as e:
            logger.error(f"Erro de conexão ao tentar renovar token: {e}")
            return None
        except Exception as e:
            logger.exception(f"Erro inesperado ao renovar token: {e}")
            return None

    # Método para obter código de autorização não é usado no fluxo de refresh, mantido para referência
    # def get_authorization_url(self):
    #     params = {
    #         'response_type': 'code',
    #         'client_id': self.app_id,
    #         'redirect_uri': self.redirect_uri,
    #     }
    #     auth_url = f"https://auth.mercadolivre.com.br/authorization?{httpx.QueryParams(params)}"
    #     return auth_url

    # Método para obter token inicial com código não é usado no fluxo de refresh, mantido para referência
    # async def fetch_initial_token(self, code):
    #     ...
