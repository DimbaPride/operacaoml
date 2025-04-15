import os
from typing import Literal, Optional
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic

# Carregamento automático das variáveis de ambiente
load_dotenv()

LLMProvider = Literal["openai", "groq", "anthropic"]

class LLMFactory:
    """Fábrica para criar instâncias de diferentes LLMs"""
    
    @staticmethod
    def create_llm(
        provider: LLMProvider = "openai", 
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        api_key: Optional[str] = None
    ) -> BaseChatModel:
        """
        Cria uma instância de LLM com base no provedor especificado.
        
        Args:
            provider: O provedor LLM ("openai", "groq", "anthropic")
            model_name: Nome específico do modelo (opcional)
            temperature: Temperatura para geração de texto
            api_key: Chave API opcional para autenticação
            
        Returns:
            Uma instância do modelo LLM solicitado
        """
        if provider == "openai":
            key_to_use = api_key or os.getenv("OPENAI_API_KEY")
            if not key_to_use:
                raise ValueError(f"Chave API para {provider} não fornecida nem encontrada no ambiente.")
            default_model = "gpt-4.1"
            return ChatOpenAI(
                api_key=key_to_use,
                model_name=model_name or default_model,
                temperature=temperature
            )
        
        elif provider == "groq":
            key_to_use = api_key or os.getenv("GROQ_API_KEY")
            if not key_to_use:
                raise ValueError(f"Chave API para {provider} não fornecida nem encontrada no ambiente.")
            default_model = "llama-3.3-70b-versatile"
            return ChatGroq(
                api_key=key_to_use,
                model_name=model_name or default_model,
                temperature=temperature
            )
        
        elif provider == "anthropic":
            key_to_use = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key_to_use:
                raise ValueError(f"Chave API para {provider} não fornecida nem encontrada no ambiente.")
            default_model = "claude-3-5-haiku-latest"
            return ChatAnthropic(
                api_key=key_to_use,
                model_name=model_name or default_model,
                temperature=temperature
            )
        
        else:
            raise ValueError(f"Provedor LLM não suportado: {provider}")
