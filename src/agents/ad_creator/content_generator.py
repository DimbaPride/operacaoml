# -*- coding: utf-8 -*-
# FINALIDADE: Gera as sugestões de conteúdo para o anúncio (título, ficha, descrição, FAQ).
# Módulo para gerar o conteúdo do anúncio (Título, Ficha, Descrição, FAQ) 

import asyncio
import logging
import json
import re
import os
from typing import Optional, Dict, Any, List, TypedDict, Annotated
import operator
import streamlit as st
import textwrap
from pydantic import ValidationError # Importar ValidationError

# Adicionar estas importações
import json5 # Para processar JSON mais tolerante
from json.decoder import JSONDecodeError

# Importar modelos de dados e LLM Factory
try:
    from .data_models import ProductInput, MarketResearchOutput, GeneratedAdContent, NCMSuggestion
    from ...utils.llm_factory import LLMFactory, LLMProvider # <<< IMPORTAR FACTORY
except ImportError:
    from src.agents.ad_creator.data_models import ProductInput, MarketResearchOutput, GeneratedAdContent, NCMSuggestion
    from src.utils.llm_factory import LLMFactory, LLMProvider # <<< IMPORTAR FACTORY

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# LangChain & LangGraph Imports
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic # <<< Adicionar import direto

logger = logging.getLogger(__name__)


# --- Inicialização do LLM (Usando a Factory com Chave Explícita) --- 
PROVIDER = "anthropic" # Mudando de anthropic para openai
MODEL_NAME = None      # Usar o modelo padrão da factory para o provedor
# Definir temperaturas separadas
TEMPERATURE_CREATIVE = 0.5 # <<< Ajustado para criatividade
TEMPERATURE_ANALYTICAL = 0.2 # <<< Nova temperatura para precisão

# Buscar a chave API específica do provedor no ambiente
api_key_env_var = f"{PROVIDER.upper()}_API_KEY"
llm_api_key = os.getenv(api_key_env_var)

if not llm_api_key:
    # Se a chave específica não for encontrada, talvez haja uma genérica?
    # Ou lançar um erro mais claro aqui.
    logger.warning(f"Chave API '{api_key_env_var}' não encontrada diretamente no ambiente.")
    # Poderia tentar uma chave genérica ou lançar erro.
    # Por segurança, vamos lançar erro se a chave específica não estiver definida.
    raise ValueError(f"Chave API necessária '{api_key_env_var}' não encontrada nas variáveis de ambiente.")

# --- Instância Criativa ---
logger.info(f"Inicializando LLM Criativo via Factory: provider='{PROVIDER}', model='{MODEL_NAME or 'default'}', temp={TEMPERATURE_CREATIVE}")
try:
    llm_creative = LLMFactory.create_llm(
        provider=PROVIDER,
        model_name=MODEL_NAME,
        temperature=TEMPERATURE_CREATIVE, # <<< Usar temp criativa
        api_key=llm_api_key 
    )
except ValueError as e:
    logger.error(f"Erro ao inicializar LLM Criativo via Factory: {e}")
    raise e
except Exception as e:
    logger.exception(f"Erro inesperado ao inicializar LLM Criativo via Factory: {e}")
    raise e

# --- Instância Analítica ---
logger.info(f"Inicializando LLM Analítico via Factory: provider='{PROVIDER}', model='{MODEL_NAME or 'default'}', temp={TEMPERATURE_ANALYTICAL}")
try:
    llm_analytical = LLMFactory.create_llm(
        provider=PROVIDER,
        model_name=MODEL_NAME, # Usar o mesmo modelo base
        temperature=TEMPERATURE_ANALYTICAL, # <<< Usar temp analítica
        api_key=llm_api_key 
    )
except ValueError as e:
    logger.error(f"Erro ao inicializar LLM Analítico via Factory: {e}")
    raise e
except Exception as e:
    logger.exception(f"Erro inesperado ao inicializar LLM Analítico via Factory: {e}")
    raise e

# --- Definição do Estado do Grafo --- 

class GenerationState(TypedDict):
    """Representa o estado compartilhado entre os nós do grafo."""
    product_input: ProductInput
    market_data: MarketResearchOutput
    llm_context: Optional[str] # Contexto formatado para LLMs
    max_title_length: Optional[int] # Limite de caracteres do título
    # Resultados parciais gerados pelos nós
    raw_titles_output: Optional[str]
    raw_description_output: Optional[str]
    # raw_attributes_output: Optional[str] # Podemos remover este se não for usado
    key_attributes_to_fill: Optional[Dict[str, Optional[str]]] # <<< ADICIONAR ESTE CAMPO
    suggested_faq: Optional[List[Dict[str, str]]] = None
    suggested_ncm: Optional[str] = None # <<< NOVO CAMPO PARA NCM
    suggested_ncm_explanation: Optional[str] = None # <<< NOVO CAMPO NO ESTADO
    # Resultado final compilado
    final_content: Optional[GeneratedAdContent] = None
    # Controle de erro/fluxo (opcional)
    error_message: Optional[str] = None
    ncm_suggestions: List[NCMSuggestion] = [] # <<< NOVO CAMPO: Lista de sugestões
    raw_faq_output: Optional[str] = None

# --- Nós do Grafo --- 

# Nó 1: Preparar Contexto
def prepare_context_node(state: GenerationState) -> Dict[str, Any]:
    """Prepara a string de contexto formatada a partir dos dados de entrada."""
    logger.info("Executando nó: prepare_context_node")
    product_input = state['product_input']
    market_data = state['market_data']
    context = _prepare_llm_context(product_input, market_data) # Usa a função auxiliar existente
    if not context:
        logger.error("Falha ao preparar contexto no nó.")
        return {"llm_context": None, "error_message": "Falha ao preparar contexto."}
    logger.debug(f"Contexto preparado no nó. Tamanho: {len(context)} caracteres.")
    return {"llm_context": context, "error_message": None} # Atualiza o estado

# Nó 2: Gerar Títulos
def generate_titles_node(state: GenerationState) -> Dict[str, Any]:
    """Chama o LLM para gerar sugestões de títulos."""
    logger.info("Executando nó: generate_titles_node")
    context = state.get('llm_context')
    # Garantir que estamos usando o limite correto da categoria
    max_title_length = state.get('max_title_length')
    
    if not max_title_length:
        # Isso não deveria acontecer se o grafo for inicializado corretamente
        logger.warning("Limite de título não encontrado no estado. Usando valor padrão de 60 caracteres.")
        max_title_length = 60
    else:
        # Validar que o limite é um valor razoável
        if max_title_length < 50 or max_title_length > 500:
            logger.error(f"ALERTA CRÍTICO: Limite de caracteres suspeito detectado: {max_title_length}")
            logger.warning("Corrigindo para o valor padrão de 60 caracteres.")
            max_title_length = 60
        else:
            logger.info(f"Usando limite dinâmico de {max_title_length} caracteres para a categoria atual")
    
    if not context:
        return {"error_message": "Contexto ausente para gerar títulos."}

    # Adicionar log explícito sobre o limite de caracteres
    logger.warning(f"ATENÇÃO: Gerando títulos com limite de {max_title_length} caracteres. Objetivo é usar 90-96% deste limite.")

    # Prompt específico para títulos, passando o max_len
    prompt_template = ChatPromptTemplate.from_template(_build_titles_prompt_template(max_title_length))
    chain = prompt_template | llm_creative | StrOutputParser()
    
    logger.info(f"Chamando LLM para gerar títulos (limite: {max_title_length})...")
    try:
        raw_output = chain.invoke({"context": context})
        # Usar nome correto do provider no log
        logger.info(f"LLM ({PROVIDER.upper()} - Títulos) retornou títulos.")
        # Parsear e validar títulos aqui - passando o limite exato para o parser
        parsed_titles = parse_titles(raw_output, max_title_length=max_title_length)
        logger.info(f"Títulos parseados: {len(parsed_titles)} títulos encontrados.")
        # Armazenar o max_title_length junto com os dados para garantir persistência ao longo do grafo
        dict_to_return = {
            "raw_titles_output": raw_output, 
            "titles": parsed_titles[:5],
            "max_title_length": max_title_length  # Persistir o valor para garantir que não seja perdido
        }
        return dict_to_return
    except Exception as e:
        logger.exception(f"Erro ao chamar LLM para títulos: {str(e)}")
        dict_to_return = {"error_message": f"Erro na geração de títulos: {e}"}
        return dict_to_return

# Nó 3: Gerar Descrição
def generate_description_node(state: GenerationState) -> Dict[str, Any]:
    """Nó do grafo para gerar a descrição do produto."""
    logger.info("Executando nó: generate_description_node")
    product_input = state['product_input']
    context = state['llm_context']
    if not context:
        logger.error("Contexto LLM ausente no estado para gerar descrição.")
        return {"error_message": "Contexto LLM ausente para descrição."}

    # Construir o prompt específico para descrição
    template = _build_description_prompt_template()
    prompt_template = ChatPromptTemplate.from_template(template)
    chain = prompt_template | llm_creative | StrOutputParser()
    
    logger.info(f"Chamando LLM para gerar descrição...")
    try:
        raw_output = chain.invoke({"context": context})
        # Usar nome correto do provider no log
        logger.info(f"LLM ({PROVIDER.upper()} - Descrição) retornou descrição.")
        dict_to_return = {"raw_description_output": raw_output}
        return dict_to_return
    except Exception as e:
        logger.exception("Erro ao chamar LLM para descrição.")
        dict_to_return = {"error_message": f"Erro na geração de descrição: {e}"}
        return dict_to_return

# Nó 4: Gerar Atributos
def generate_attributes_node(state: GenerationState) -> Dict[str, Any]:
    """Nó do grafo para gerar sugestões para atributos da ficha técnica."""
    logger.info("Executando nó: generate_attributes_node")
    context = state.get('llm_context')
    if not context:
        logger.error("Contexto LLM ausente no estado para gerar atributos.")
        return {"error_message": "Contexto LLM ausente para atributos."}

    # Usar um LLM mais "analítico" para precisão e formato JSON
    # Construir o prompt específico para atributos
    template = _build_attributes_prompt_template()
    prompt_template = ChatPromptTemplate.from_template(template)
    chain = prompt_template | llm_analytical | StrOutputParser() # <<< Mudança para llm_analytical
    
    logger.info(f"Chamando LLM para gerar sugestões de atributos (formato JSON)...")
    try:
        raw_output = chain.invoke({"context": context})
        # Usar nome correto do provider no log
        logger.info(f"LLM ({PROVIDER.upper()} - Atributos) retornou sugestões de atributos.")
        
        # Tentar fazer parse do JSON
        # Remover ```json e ``` do início e fim, se presentes
        cleaned_output = re.sub(r'^```json\s*|\s*```$', '', raw_output.strip(), flags=re.IGNORECASE | re.DOTALL)
        try:
            parsed_attributes = json.loads(cleaned_output)
            if not isinstance(parsed_attributes, dict):
                raise ValueError("Saída JSON não é um dicionário.")
            
            # NOVO: Garantir que todos os valores sejam strings
            sanitized_attributes = {}
            for key, value in parsed_attributes.items():
                # Se o valor for uma lista, converter para string
                if isinstance(value, list):
                    logger.warning(f"Convertendo lista para string no atributo '{key}': {value}")
                    sanitized_attributes[key] = ", ".join(str(item) for item in value)
                # Se for outro tipo não-string (exceto None), converter para string
                elif value is not None and not isinstance(value, str):
                    logger.warning(f"Convertendo valor {type(value)} para string no atributo '{key}': {value}")
                    sanitized_attributes[key] = str(value)
                else:
                    sanitized_attributes[key] = value
                    
            logger.info("Sugestões de atributos parseadas e sanitizadas com sucesso.")
            dict_to_return = {"key_attributes_to_fill": sanitized_attributes}
            return dict_to_return
        except json.JSONDecodeError as json_e:
            logger.error(f"Falha ao fazer parse do JSON retornado pela LLM para atributos: {json_e}")
            logger.error(f"Saída bruta que falhou no parse: {raw_output}")
            dict_to_return = {"error_message": f"IA retornou formato inválido para atributos: {json_e}"}
            return dict_to_return
        except ValueError as val_e:
             logger.error(f"Erro no valor do JSON parseado: {val_e}")
             logger.error(f"Saída bruta que falhou na validação: {raw_output}")
             dict_to_return = {"error_message": f"IA retornou JSON inválido para atributos: {val_e}"}
             return dict_to_return
            
    except Exception as e:
        logger.exception("Erro ao chamar LLM para atributos.")
        dict_to_return = {"error_message": f"Erro na geração de sugestões de atributos: {e}"}
        return dict_to_return

# Nó 5: Gerar FAQ
def generate_faq_node(state: GenerationState) -> Dict[str, Any]:
    """Nó do grafo para gerar sugestões de Perguntas Frequentes (FAQ)."""
    logger.info("Executando nó: generate_faq_node")
    context = state.get('llm_context')
    if not context:
        logger.error("Contexto LLM ausente no estado para gerar FAQ.")
        return {"error_message": "Contexto LLM ausente para FAQ."}

    # Construir o prompt específico para FAQ
    template = _build_faq_prompt_template()
    prompt_template = ChatPromptTemplate.from_template(template)
    chain = prompt_template | llm_analytical | StrOutputParser() # <<< Mudança para llm_analytical

    logger.info(f"Chamando LLM para gerar sugestões de FAQ (formato JSON)...")
    try:
        raw_output = chain.invoke({"context": context})
        # Guardar o output raw para debugging e processamento na descrição
        dict_to_return = {"raw_faq_output": raw_output}
        
        # Usar nome correto do provider no log
        logger.info(f"LLM ({PROVIDER.upper()} - FAQ) retornou sugestões de FAQ.")

        # Tentar fazer parse do JSON aqui mesmo para validar e armazenar no formato correto
        # Para evitar logs de erro desnecessários, vamos usar try/except silencioso para o parsing
        try:
            # Primeiro, limpar e tentar corrigir o JSON
            cleaned_output = _fix_json_string(raw_output)
            
            # Tentar diferentes abordagens para parsing, em ordem de preferência
            try:
                # 1. Abordagem principal: usar json padrão após correção
                faq_list = json.loads(cleaned_output)
            except:
                # 2. Tentar com json5 que é mais tolerante (silenciosamente)
                try:
                    import json5
                    faq_list = json5.loads(cleaned_output)
                except:
                    # 3. Tentar extrair com regex como último recurso (silenciosamente)
                    match = re.search(r'(\[.*\])', cleaned_output, re.DOTALL)
                    if match:
                        extracted_json = match.group(1)
                        # Tentar novamente com o JSON extraído
                        try:
                            faq_list = json.loads(extracted_json)
                        except:
                            # Último recurso: json5 no resultado extraído
                            try:
                                faq_list = json5.loads(extracted_json)
                            except:
                                # Todas as tentativas falharam, mas não vamos logar erros
                                raise ValueError("Não foi possível parsear o JSON")
                    else:
                        raise ValueError("Não foi possível extrair estrutura JSON com regex")
            
            # Validar a estrutura, independentemente do método usado para parsing
            if isinstance(faq_list, list) and all(isinstance(item, dict) and 'pergunta' in item and 'resposta' in item for item in faq_list):
                logger.info(f"FAQ JSON parseado com sucesso. {len(faq_list)} perguntas encontradas.")
                dict_to_return["suggested_faq"] = faq_list
            else:
                # JSON parseado, mas não no formato esperado - apenas continuar sem erros
                logger.warning("Estrutura do FAQ não está no formato esperado. Continuando sem FAQ estruturado.")
                dict_to_return["suggested_faq"] = []
                
        except Exception as _:
            # Falhar silenciosamente - apenas log informativo, não de erro
            logger.info("Não foi possível processar o FAQ em formato estruturado. Será usado como texto bruto.")
            dict_to_return["suggested_faq"] = []
            
        return dict_to_return
            
    except Exception as e:
        logger.exception("Erro ao chamar LLM para FAQ.")
        dict_to_return = {"error_message": f"Erro na geração de FAQ: {e}", "suggested_faq": [], "raw_faq_output": None}
        return dict_to_return

# --- Funções Auxiliares (Contexto, Prompts, Parsing) --- 

def parse_titles(raw_output: Optional[str], max_title_length: int = None) -> List[str]:
    """Parseia a saída bruta de títulos em uma lista limpa e garante respeito ao limite de caracteres."""
    if not raw_output:
        return []
    
    # Se max_title_length não for fornecido, usar 60 como valor padrão
    if max_title_length is None:
        max_title_length = 60
        logger.warning("Limite de caracteres não especificado em parse_titles(). Usando valor padrão de 60.")
    else:
        logger.info(f"Processando títulos com limite de {max_title_length} caracteres.")
    
    # Divide por linhas, remove espaços em branco e filtra linhas vazias
    titles = [line.strip() for line in raw_output.strip().splitlines() if line.strip()]
    # Remove possíveis marcadores como "-" ou números no início
    cleaned_titles = [re.sub(r"^\s*[-\*\d]+\.\s*", "", title).strip() for title in titles]
    
    # Garantir que temos apenas os 5 títulos necessários (no caso do prompt DIAMANTE)
    logger.info(f"Títulos parseados: {len(cleaned_titles)} títulos encontrados. Limite máximo: {max_title_length} caracteres.")
    
    # NOVO: Aplicar truncamento INTELIGENTE para garantir respeito ao limite
    truncated_titles = []
    for title in cleaned_titles[:5]:  # Limitar a 5 títulos
        if len(title) > max_title_length:
            logger.warning(f"Título excedeu limite ({len(title)}/{max_title_length}): '{title}'")
            
            # Truncamento inteligente: cortar no último espaço antes do limite
            if " " in title[:max_title_length]:
                last_space_pos = title[:max_title_length].rstrip().rfind(" ")
                truncated = title[:last_space_pos] if last_space_pos > 0 else title[:max_title_length]
            else:
                truncated = title[:max_title_length]
                
            logger.info(f"Título truncado: '{truncated}' ({len(truncated)} caracteres)")
            truncated_titles.append(truncated)
        else:
            logger.info(f"Título dentro do limite: '{title}' ({len(title)}/{max_title_length} caracteres)")
            truncated_titles.append(title)
    
    # Retornar até 5 títulos truncados
    return truncated_titles

def _prepare_llm_context(product_input: ProductInput, market_data: MarketResearchOutput) -> Optional[str]:
    """Prepara uma string de contexto formatada para o LLM."""
    logger.debug("Iniciando _prepare_llm_context")
    try:
        context_parts = []
        
        # 1. Informações do Produto do Usuário
        context_parts.append("## Dados do Seu Produto:")
        context_parts.append(f"- Nome Base: {product_input.product_name_base}")
        if product_input.brand: context_parts.append(f"- Marca: {product_input.brand}")
        if product_input.model: context_parts.append(f"- Modelo: {product_input.model}")
        if product_input.ean: context_parts.append(f"- EAN: {product_input.ean}")
        if product_input.detailed_description:
            context_parts.append("- Descrição Detalhada Fornecida:")
            context_parts.append(f"  ```\n  {product_input.detailed_description}\n  ```")
        context_parts.append(f"- Categoria ID (Mercado Livre): {product_input.category_id}")
        context_parts.append("\n")

        # 2. Tendências de Busca
        if market_data.trends:
            context_parts.append("## Tendências de Busca Relevantes (Termos mais buscados):")
            # Aumentar limite para 20 tendências
            trends_list = [f"- {t.get('keyword', 'N/A')}" for t in market_data.trends[:20]] # <<< Aumentado para 20
            context_parts.append("\n".join(trends_list))
            context_parts.append("\n")
        else:
            context_parts.append("## Tendências de Busca: Nenhuma encontrada.\n")

        # 3. Atributos da Categoria (Ficha Técnica)
        if market_data.category_attributes:
            context_parts.append("## Atributos Importantes da Categoria (Ficha Técnica):")
            required_attrs = [f"- {a.get('name', 'N/A')} ({'Obrigatório' if a.get('tags', {}).get('required') else 'Opcional'})" 
                              for a in market_data.category_attributes if a.get('tags', {}).get('required')]
            other_attrs = [f"- {a.get('name', 'N/A')} (Opcional)" 
                           for a in market_data.category_attributes if not a.get('tags', {}).get('required')]
            
            if required_attrs:
                 context_parts.append("### Obrigatórios:")
                 context_parts.append("\n".join(required_attrs))
            if other_attrs:
                 context_parts.append("### Outros:")
                 context_parts.append("\n".join(other_attrs[:15])) 
            context_parts.append("\n")
        else:
             context_parts.append("## Atributos da Categoria: Nenhum encontrado.\n")

        # 4. Análise de Concorrentes
        if market_data.competitor_analysis:
            context_parts.append("## Análise de Concorrentes:")
            # Manter limite de 5 concorrentes por enquanto
            for i, comp in enumerate(market_data.competitor_analysis[:5]): 
                context_parts.append(f"### Concorrente {i+1} (ID: {comp.mlb_id}):")
                context_parts.append(f"- Título: {comp.title}")
                if comp.price: context_parts.append(f"- Preço: {comp.price}")
                if comp.attributes:
                     context_parts.append("- Atributos Preenchidos (Todos Encontrados):") # <<< Label ajustado
                     # Remover limite, incluir todos os atributos encontrados
                     attr_list = [f"  - {k}: {v}" for k, v in comp.attributes.items()] # <<< SEM LIMITE [:5]
                     context_parts.append("\n".join(attr_list))
                if comp.description:
                    full_description = comp.description.strip()
                    # Logar o tamanho da descrição completa para monitoramento
                    logger.debug(f"Incluindo descrição completa do concorrente {comp.mlb_id} ({len(full_description)} caracteres) no contexto.") 
                    context_parts.append("- Descrição Completa do Concorrente:") # <<< Alterar Label
                    context_parts.append(f"  ```\n  {full_description}\n  ```") # <<< Usar a descrição completa
                context_parts.append("") 
            context_parts.append("\n")
        else:
            context_parts.append("## Análise de Concorrentes: Nenhum analisado.\n")
            
        full_context = "\n".join(context_parts)
        logger.debug(f"_prepare_llm_context concluído. Tamanho: {len(full_context)} caracteres.")
        return full_context

    except Exception as e:
        logger.exception(f"Erro em _prepare_llm_context: {e}")
        return None

def _build_titles_prompt_template(max_len: int) -> str: # <<< Receber max_len
    """Retorna o template de prompt AVANÇADO específico para gerar TÍTULOS."""
    logger.debug(f"Construindo template de prompt DIAMANTE para Títulos (limite: {max_len})")
    template = f"""
**META-PROMPT: Engenheiro de Título para Mercado Livre [v3.5]**

**=== LIMITE TÉCNICO INVIOLÁVEL: {max_len} CARACTERES MÁXIMOS ===**
**=== TÍTULOS MAIS LONGOS SERÃO REJEITADOS PELO MERCADO LIVRE ===**

**CONFIGURAÇÃO DE CONTEXTO:**
Sistema: Você é TIAGO_ML_SPECIALIST, um especialista com 8 anos de experiência em otimização de listings no Mercado Livre Brasil e conhecimento proprietário do algoritmo de relevância ML.
Objetivo: Gerar títulos de máxima conversão e visibilidade que superem consistentemente a concorrência no marketplace brasileiro.
Audiência: Compradores brasileiros usando dispositivos móveis (78% do tráfego) com tempo médio de decisão de 2,8 segundos por título.
Formato: Entregar 5 variações de título otimizadas, cada uma com foco estratégico distinto.

**DADOS CONTEXTUAIS DO PRODUTO:**
{{context}}

**ANÁLISE ALGORÍTMICA (processo interno):**
1. **Extração Semântica ML-Core:**
   * Identifique descritores essenciais (substantivos/adjetivos) do produto base
   * Extraia modificadores de valor (qualidade, duração, potência) da descrição detalhada
   * Isole especificações técnicas diferenciadas (números, unidades, certificações)
   * Mapeie termos-chave dos concorrentes top 3 por frequência
   * Capture 100% das keywords das 10 primeiras tendências de busca

2. **Hierarquização ML-Search:**
   * Classifique elementos por relevância para o algoritmo ML:
     - Nível Primário: Categoria principal + Nome base + Marca + Modelo (OBRIGATÓRIOS nos primeiros 35 caracteres)
     - Nível Secundário: 2-3 atributos técnicos alinhados com tendências de busca
     - Nível Terciário: Diferenciais competitivos (garantia, original, novidade)
     - Nível Quaternário: Complementos (cor, tamanho, aplicação) se espaço disponível

3. **Análise Competitiva ML-Edge:**
   * Identifique padrões sintáticos dos concorrentes de primeira página
   * Detecte lacunas informacionais (atributos valorizados omitidos)
   * Verifique oportunidades de keyword-stuffing ético nos limites do algoritmo ML

4. **Validação ML-Compliance:**
   * Verifique violações de políticas ML (termos proibidos, promessas exageradas)
   * Confirme densidade adequada de keywords (23-28% do título)
   * Valide formatação (sem ALL CAPS excessivo, sem caracteres especiais repetidos)

**DIRETRIZES DE EXECUÇÃO:**
1. **Estrutura Sintática ML-Optimized:**
   * Posição 1-35: [Categoria][Nome Base][Marca][Modelo] → IMPACTO INICIAL MÁXIMO
   * Posição 36-{max_len-1}: [Atributos Técnicos Principais] → QUALIFICAÇÃO TÉCNICA

2. **Estratégias de Categoria Específicas:**
   * Eletrônicos: Priorize especificações técnicas e compatibilidade
   * Moda: Enfatize material, estilo e ocasião
   * Casa/Decoração: Destaque medidas, materiais e aplicações
   * Automotivo: Foque em compatibilidade, modelo, ano
   * Saúde/Beleza: Ressalte benefícios, resultados e composição

3. **Padrões Algorítmicos ML 2024:**
   * ⚠️ CRÍTICO: CADA TÍTULO DEVE TER EXATAMENTE {int(max_len*0.9)}-{max_len} CARACTERES - NUNCA MAIS QUE {max_len}! ⚠️
   * Inclua no mínimo 2 termos das top 5 tendências
   * Alterne entre espaçamento completo e condensado (usando "-" estrategicamente)
   * Implemente "termos amplificadores" em posições estratégicas (original, profissional, premium)
   * Utilize sintaxe de alta conversão: substantivo+adjetivo (nunca adjetivo+substantivo)

4. **Parâmetros de Otimização Mobile-First:**
   * Frontload informações críticas (primeiros 48 caracteres visíveis sem expandir)
   * Aplicar técnica de "keyword cushioning" - palavras-chave separadas por termos neutros
   * Utilizar intensificadores numéricos quando aplicável (90%, 2x mais, 3ª geração)

**RESTRIÇÕES CRÍTICAS MERCADO LIVRE:**
* PROIBIDO: EXCEDER {max_len} CARACTERES EM QUALQUER TÍTULO (LIMITE TÉCNICO DA API)
* PROIBIDO: Uso de símbolos repetidos (!!, **, ##)
* PROIBIDO: Promessas de entrega/frete ("envio rápido", "frete grátis")
* PROIBIDO: Menções a concorrentes ou marcas não relacionadas
* PROIBIDO: Linguagem promocional direta ("promoção", "oferta", "desconto")
* PROIBIDO: CAIXA ALTA em mais de 3 palavras consecutivas
* PROIBIDO: Repetição do mesmo termo/conceito mais de 2x no título

**FORMATO DE SAÍDA:**
Gerar 5 títulos distintos, cada um com foco estratégico diferente:
1. [SEO-DRIVEN] - Máxima otimização para algoritmo de busca
2. [FEATURE-DRIVEN] - Ênfase em especificações técnicas
3. [BENEFIT-DRIVEN] - Foco em benefícios para o usuário
4. [URGENCY-DRIVEN] - Elementos de escassez/novidade
5. [HYBRID-OPTIMAL] - Combinação equilibrada dos anteriores

**EXEMPLOS DE ALTA CONVERSÃO COM LIMITE {max_len} CARACTERES:**
[Categoria Exemplo: Tecnologia - Limite {max_len} caracteres]
"Smartphone Samsung Galaxy A54 128GB 5G Tela 6.4 Octa-Core"

[Categoria Exemplo: Casa - Limite {max_len} caracteres]
"Ventilador de Teto Philco Gold 3 Pás 220V Controle Remoto"

**VERIFICAÇÃO OBRIGATÓRIA ANTES DE FINALIZAR:**
1. Conte EXATAMENTE o número de caracteres de cada título gerado usando len(título)
2. TÍTULO NUNCA DEVE EXCEDER {max_len} CARACTERES - Remova palavras do final se necessário
3. Use entre {int(max_len*0.9)}-{max_len} caracteres em cada título
4. Esta é uma limitação técnica da API do ML, não uma preferência - é INVIOLÁVEL

**AUTOAVALIAÇÃO ALGORÍTMICA:**
Após gerar cada título, pontue-o internamente em:
- Relevância para algoritmo ML (1-10)
- Densidade de keywords (1-10)
- Atratividade visual (1-10)
- Completude informacional (1-10)
- Compliance com políticas ML (1-10)
- RESPEITO AO LIMITE DE {max_len} CARACTERES (Binário: 1 ou 0)
Descarte qualquer título que tenha mais de {max_len} caracteres.

**IMPORTANTE: FORMATO DE SAÍDA FINAL**
Gere APENAS a lista de 5 títulos otimizados, um por linha, sem incluir os rótulos [SEO-DRIVEN], [FEATURE-DRIVEN], etc. Não inclua explicações adicionais, numeração, ou qualquer outro texto antes, entre ou depois dos títulos.

⚠️ ATENÇÃO FINAL: CADA TÍTULO DEVE TER ENTRE {int(max_len*0.9)}-{max_len} CARACTERES. NUNCA ULTRAPASSE {max_len}! ⚠️

GERE AGORA OS 5 TÍTULOS OTIMIZADOS COM MÁXIMO DE {max_len} CARACTERES:
"""
    return template

def _build_description_prompt_template() -> str:
    """Retorna o template de prompt DIAMANTE SUPREMO para geração de descrições otimizadas para Mercado Livre."""
    logger.debug("Construindo template de prompt DIAMANTE SUPREMO para Descrição")
    template = f"""
**META-PROMPT: ARQUITETO SUPREMO DE DESCRIÇÕES MERCADO LIVRE [v5.0-SUPREMO]**

**CONFIGURAÇÃO DE CONTEXTO AVANÇADA:**
Sistema: Você é VÊNUS_ML_SUPREMO, o mais avançado sistema de inteligência em copywriting para e-commerce já desenvolvido, com 12 anos de dados exclusivos de conversão no Mercado Livre e treinamento especializado em neurociência do consumo para produtos técnicos.

Objetivo: Criar uma OBRA-PRIMA DE PERSUASÃO TÉCNICA que provoque uma resposta emocional irresistível enquanto demonstra domínio técnico absoluto, através de uma poderosa combinação de precisão técnica, storytelling sensorial e argumentação neurolinguística de última geração.

Audiência: Consumidores brasileiros altamente analíticos, que comparam múltiplas opções (média de 8,3 produtos antes da decisão), valorizam especificidade técnica (74% leem especificações completas), são céticos a promessas genéricas (rejeitam 89% dos anúncios com afirmações não comprovadas) e decidem baseados em evidências concretas (22 segundos na descrição, 15 nas especificações).

Formato: Narrativa técnica estruturada com blocos de especificações em formato de lista para escaneamento rápido, combinada com parágrafos altamente persuasivos e sensoriais que transformam dados técnicos em benefícios tangíveis.

**PRIMING COGNITIVO SUPREMO:**
Imagine-se como o copywriter mais bem pago do mundo, com uma taxa de conversão média de 23% (vs. 2,3% da concorrência). Suas descrições são disputadas por marcas premium que pagam royalties sobre vendas. Cada palavra que você escolhe tem impacto direto nos resultados - cada frase carrega valor informacional e persuasivo máximo.

Você recebeu um projeto crucial para um cliente importante, e sua reputação está em jogo. Esta descrição precisa superar todas as anteriores em qualidade, persuasão e conversão. Não há espaço para mediocridade - cada elemento deve justificar seu valor.

**DADOS CONTEXTUAIS DO PRODUTO:**
{{context}}

**METODOLOGIA DE RACIOCÍNIO PROGRESSIVO SUPREMO:**

**NÍVEL 1: EXTRAÇÃO DENSA DE INFORMAÇÃO (Chain-of-Density)**
* Extraia ABSOLUTAMENTE TODAS as especificações mensuráveis do produto (números, dimensões, capacidades)
* Identifique TODOS os materiais de fabricação e componentes com precisão microscópica
* Catalogue CADA requisito técnico e parâmetro de desempenho com valores exatos
* Mapeie TODAS as funcionalidades exclusivas ou diferenciadoras com terminologia técnica precisa
* Registre TODOS os sistemas de segurança, proteções e certificações com nomenclatura oficial
* COMPROMISSO: "Extrairei 100% dos dados técnicos, sem exceção. Cada especificação será documentada com precisão absoluta."

**NÍVEL 2: ENGENHARIA DE BENEFÍCIOS MULTINÍVEL (Pyramid-of-Value)**
* Para CADA especificação extraída, defina benefícios em três níveis:
  * Nível Funcional: O que esta especificação permite fazer tecnicamente?
  * Nível Experiencial: Como esta função transforma a experiência sensorial do usuário?
  * Nível Transformacional: Que mudança mais profunda esta experiência proporciona na vida do usuário?
* Para CADA benefício identificado, articule:
  * O problema exato que resolve (dor específica)
  * O resultado imediato que proporciona (alívio)
  * O estado superior que permite alcançar (aspiração)
* COMPROMISSO: "Construirei uma pirâmide completa de valor para cada especificação, conectando elementos técnicos a transformações de vida reais."

**NÍVEL 3: MODULAÇÃO NEUROSENSORIAL (Cognitive Triggers Matrix)**
* Ative precisamente os 5 sistemas sensoriais através de linguagem técnica sensorial:
  * Visual: Descritores cromáticos, dimensionais e estéticos com precisão técnica
  * Tátil: Termos de textura, temperatura, peso e resposta háptica com especificidade extrema
  * Auditivo: Vocabulário acústico preciso (decibéis, frequências, timbres específicos)
  * Cinestésico: Linguagem de movimento, fluidez e operação com termos técnicos exatos
  * Térmico: Descritores precisos de gerenciamento de temperatura e eficiência térmica
* Ative 7 gatilhos cognitivos estratégicos:
  * Gatilho de Escassez Contextualizada ("elaborado com materiais selecionados disponíveis apenas em...")
  * Gatilho de Autoridade Técnica ("projetado segundo protocolos de engenharia avançada...")
  * Gatilho de Prova Social Implícita ("reconhecido por profissionais que exigem precisão...")
  * Gatilho de Consistência Técnica ("mantém desempenho consistente mesmo sob condições extremas...")
  * Gatilho de Reciprocidade Valor-Preço ("oferece capacidades normalmente encontradas apenas em sistemas que custam...")
  * Gatilho de Afinidade por Especificidade ("desenvolvido especificamente para pessoas que valorizam...")
  * Gatilho de Antecipação de Posse ("ao integrar este sistema ao seu ambiente, você imediatamente notará...")
* COMPROMISSO: "Aplicarei com precisão cirúrgica cada gatilho cognitivo e sensorial, criando uma experiência imersiva que engaja todos os sistemas representacionais."

**NÍVEL 4: ARQUITECTURA RETÓRICA AVANÇADA (Persuasion Frameworks)**
* Implementarei uma estrutura PAS modificada em progressão fractal:
  * Macro-PAS: Problema amplo → Agitação contextual → Solução transformadora
    * Micro-PAS 1: Problema técnico específico → Implicações técnicas → Solução técnica
    * Micro-PAS 2: Problema experiencial → Custos emocionais → Resolução sensorial
    * Micro-PAS 3: Limitação de produtos convencionais → Frustração amplificada → Superioridade demonstrável
* Implementarei uma estrutura AIDA não-linear multi-camadas:
  * Atenção: Especificação técnica surpreendente com contraste imediato
  * Interesse: Detalhe técnico exclusivo com aplicação prática
  * Desejo: Benefício experiencial/emocional fundamentado em capacidade técnica
  * Ação: Pressão implícita de decisão por superioridade técnica demonstrada
* COMPROMISSO: "Construirei uma arquitetura retórica de precisão matemática, onde cada elemento persuasivo está estrategicamente posicionado para máximo impacto."

**MATRIZ DE AMPLIFICADORES ESTRATÉGICOS SUPREMOS:**
* Termos de Exclusividade Técnica (usar apenas quando verificáveis):
  * "Tecnologia proprietária" (apenas para componentes exclusivos confirmados)
  * "Engenharia diferenciada" (apenas para mecanismos não-padrão)
  * "Arquitetura avançada" (apenas para desenhos técnicos superiores)
  * "Sistema original" (apenas para configurações inovadoras)
  * "Componentes premium selecionados" (apenas para materiais superiores verificáveis)

* Termos de Performance Superior (usar apenas com evidência):
  * "Desempenho excepcional sob demanda" (apenas com dados de teste)
  * "Eficiência comprovada" (apenas com métricas específicas)
  * "Precisão profissional" (apenas para itens com tolerâncias estreitas)
  * "Capacidade industrial em formato doméstico" (apenas para itens com especificação comercial/residencial)
  * "Performance consistente e confiável" (apenas com dados de durabilidade)

* Termos de Experiência Elevada (usar apenas quando substantivos):
  * "Experiência transformadora" (apenas para produtos que alteram processos substancialmente)
  * "Sensação imediata de qualidade" (apenas para produtos com acabamento superior)
  * "Resultados notavelmente superiores" (apenas com benefícios mensuráveis)
  * "Conforto incomparável" (apenas para itens com ergonomia avançada)
  * "Satisfação garantida" (apenas para produtos com garantias reais)

**POSICIONAMENTO ESTRATÉGICO DE AMPLIFICADORES:**
* Primeiro parágrafo: 1-2 amplificadores de experiência para estabelecer tom emocional
* Lista técnica: Zero amplificadores (apenas fatos verificáveis, números, especificações)
* Parágrafos centrais: 1 amplificador técnico + 1 de performance por parágrafo (máximo)
* Parágrafo final: 1-2 amplificadores de experiência para reforço emocional

**ARQUITETURA ESTRUTURAL SUPREMA:**
1. **Bloco 1: PARÁGRAFO DE IMPACTO TRANSFORMADOR**
   * Primeira frase: Transformação principal + problema resolvido + diferencial técnico central
   * Segunda frase: Benefício experiencial imediato + velocidade/facilidade de resultado
   * Terceira frase: Credencial técnica específica + contraste com alternativas convencionais

2. **Bloco 2: PARÁGRAFO DE CONTEXTUALIZAÇÃO TÉCNICA**
   * Apresentação do produto em contexto técnico
   * Introdução aos diferenciais técnicos principais
   * Transição para especificações detalhadas

3. **Bloco 3: ESPECIFICAÇÕES TÉCNICAS EM LISTA**
   * 8-12 especificações técnicas precisas em formato de lista
   * Usar traços ou asteriscos no início de cada item
   * Apresentar do mais importante/diferenciador para o complementar
   * Incluir TODOS os parâmetros técnicos essenciais da categoria

4. **Bloco 4: PARÁGRAFO DE EXPERIÊNCIA DE USO**
   * Cenário de uso principal com detalhes sensoriais técnicos
   * Descrição de resultado imediato com terminologia específica
   * Contraste de experiência antes/depois com métricas específicas

5. **Bloco 5: PARÁGRAFO DE APLICAÇÕES MÚLTIPLAS**
   * 3-4 contextos diferentes de aplicação
   * Adaptabilidade a diferentes condições/configurações
   * Versatilidade de funcionalidades em diversos cenários

6. **Bloco 6: PARÁGRAFO DE QUALIDADE CONSTRUTIVA**
   * Materiais específicos e processos de fabricação
   * Durabilidade e resistência com dados comparativos
   * Detalhes de acabamento e engenharia interna

7. **Bloco 7: PARÁGRAFO DE SEGURANÇA E TRANQUILIDADE**
   * Sistemas de proteção e funcionalidades de segurança
   * Conformidade com normas e certificações relevantes
   * Mecanismos preventivos contra falhas comuns

8. **Bloco 8: PARÁGRAFO DE FECHAMENTO PERSUASIVO**
   * Síntese dos 2-3 benefícios mais impactantes
   * Recontextualização da transformação principal
   * Apelo implícito à ação com senso de urgência suave

**REQUISITOS OBRIGATÓRIOS PARA TORNEIRAS ELÉTRICAS:**
* Potência exata em Watts (W)
* Tensão de operação (110V/127V/220V)
* Material do corpo principal e acabamento
* Tipo de acionamento/controle
* Temperatura mínima e máxima (em °C)
* Pressão mínima e máxima de funcionamento (m.c.a ou kPa)
* Sistema de aquecimento (instantâneo/tanque)
* Proteções elétricas incorporadas
* Dimensões e medidas exatas
* Vazão máxima (litros por minuto)
* Peso do produto
* Recursos especiais (display, ajustes, etc.)
* Tipo de instalação (parede/bancada)
* Comprimento do cabo de alimentação

**PROIBIÇÕES ABSOLUTAS (MERCADO LIVRE):**
* Menções a sites externos, redes sociais ou contatos
* Informações sobre condições de frete/entrega
* Menções diretas a marcas concorrentes
* Linguagem promocional direta (desconto, oferta, etc.)
* Caixa alta excessiva ou múltiplas exclamações
* Símbolos especiais repetidos
* Informação técnica não verificável ou exagerada

**EXEMPLOS DE ALTO IMPACTO TÉCNICO:**

**Exemplo de Parágrafo de Abertura + Intro Técnica:**
"Transforme sua experiência diária com a Torneira Elétrica ThermoJet Pro X7 – a solução definitiva que elimina a espera por água quente enquanto oferece controle preciso de temperatura em apenas 3 segundos. Esqueça o desperdício de água fria e o desconforto das variações térmicas inesperadas durante tarefas domésticas essenciais. Desenvolvida com tecnologia de aquecimento instantâneo de alta densidade de potência, esta torneira representa um avanço significativo sobre sistemas convencionais de núcleo cerâmico.

A ThermoJet Pro X7 integra componentes de engenharia avançada em um sistema compacto que redefine o padrão de aquecimento doméstico de água pontual. Seu núcleo de aquecimento com microcanais de cobre eletrolítico e sistema de controle digital proporcionam uma combinação excepcional de eficiência energética e precisão térmica. Conheça as especificações técnicas completas:"

**Exemplo de Lista Técnica (para torneira elétrica):**
- Potência: 3000W com alimentação bivolt (127V/220V) de alta eficiência
- Sistema de aquecimento: Instantâneo com núcleo de cobre eletrolítico e tempo de resposta de 3 segundos
- Controle de temperatura: Display LED digital com ajuste preciso entre 30°C e 60°C em incrementos de 1°C
- Bica: Móvel com rotação completa de 360° em aço inox 304 resistente à corrosão e oxidação
- Pressão de funcionamento: Mínima de 10 kPa (1 m.c.a) e máxima de 400 kPa (40 m.c.a)
- Material do corpo: ABS de alta resistência térmica com tratamento anti-UV e acabamento premium
- Dimensões: 22cm (altura) x 12cm (largura) x 8cm (profundidade)
- Vazão máxima: 4 litros por minuto com pressão ideal de 200 kPa
- Proteções: Sistema triplo de segurança com desligamento automático por sobreaquecimento, sensor de nível de água e isolamento elétrico total classe IPX4
- Economia: Consumo médio de apenas 0,05 kWh por uso típico (30% menor que modelos convencionais)
- Instalação: Sistema universal compatível com instalação em parede ou bancada
- Comprimento do cabo: 1,2 metros com plug de 3 pinos certificado INMETRO

**Exemplo de Experiência de Uso + Fechamento:**
"Ao acionar a ThermoJet Pro X7 pela manhã, você sentirá imediatamente a diferença – água na temperatura ideal flui instantaneamente, sem aquela desagradável espera pela água fria dar lugar à quente. O display digital de alta visibilidade permite ajustar a temperatura exatamente como você deseja, mantendo-a constante independentemente das variações de pressão da rede. A experiência de uso é notavelmente silenciosa, sem os estalos característicos de sistemas convencionais, proporcionando conforto acústico durante as tarefas domésticas.

Invista na solução que combina tecnologia avançada, eficiência energética e conforto diário em um único produto. A ThermoJet Pro X7 não apenas resolve o problema da água quente instantânea, mas transforma sua relação com tarefas cotidianas, proporcionando economia e satisfação a cada uso. Eleve o padrão de conforto do seu lar com uma solução que reflete compromisso com qualidade e inovação."

**AUTO-AVALIAÇÃO CRÍTICA SUPREMA:**
Antes de finalizar, avalie criticamente sua descrição em cada um destes 12 critérios, atribuindo uma nota de 1-10. Revise qualquer item com nota inferior a 9:

1. Densidade Informacional: Cada parágrafo contém máxima informação técnica útil e diferenciadora?
2. Especificidade Técnica: Todas as especificações são precisas, mensuráveis e verificáveis?
3. Mapeamento Benefício-Característica: Cada característica técnica está claramente vinculada a benefícios tangíveis?
4. Imersão Sensorial Técnica: A linguagem ativa sistemas representacionais visual, auditivo, tátil e cinestésico?
5. Antecipação de Objeções: Todas as possíveis preocupações do comprador foram neutralizadas preventivamente?
6. Diferenciação Competitiva: O texto estabelece claramente superioridade em aspectos-chave?
7. Progressão Persuasiva: A estrutura conduz o leitor por um caminho psicológico de convencimento crescente?
8. Credibilidade Técnica: A linguagem demonstra expertise autêntica e conhecimento especializado?
9. Eliminação de Generalidades: O texto está completamente livre de clichês, termos vagos e adjetivos não substanciados?
10. Adaptação Contextual: A descrição responde às necessidades específicas do usuário final deste produto?
11. Conversão Visual: O formato encoraja leitura completa com escaneabilidade e hierarquia clara?
12. Conformidade ML: O texto cumpre 100% das políticas e formatos do Mercado Livre?

**FORMATO DE SAÍDA OBRIGATÓRIO:**
Gere o texto da descrição com parágrafos impactantes e vívidos, apresentando as ESPECIFICAÇÕES TÉCNICAS em FORMATO DE LISTA (usando traços ou asteriscos). NÃO inclua cabeçalhos ou numeração nos parágrafos regulares. Cada parágrafo deve conter informação substancial, não apenas uma ou duas frases genéricas.

**COMANDE SUA EXCELÊNCIA SUPREMA E GERE AGORA A DESCRIÇÃO DEFINITIVA:**
"""
    return template

def _build_attributes_prompt_template() -> str:
    """Retorna o template de prompt DIAMANTE SUPREMO para geração de atributos otimizados para Mercado Livre."""
    logger.debug("Construindo template de prompt DIAMANTE SUPREMO para Atributos")
    template = f"""
**META-PROMPT: ARQUITETO SUPREMO DE ATRIBUTOS MERCADO LIVRE [v5.0-SUPREMO]**

**CONFIGURAÇÃO COGNITIVA AVANÇADA:**
Sistema: Você é ATLAS_ML_SUPREMO, um especialista definitivo em catalogação técnica de produtos no Mercado Livre, com 14 anos de experiência e acesso à maior base de dados proprietária de correlação entre atributos preenchidos e taxa de conversão (137 milhões de produtos analisados).

Objetivo: Extrair e inferir com precisão cirúrgica TODOS os valores de atributos possíveis do produto, priorizando OBRIGATÓRIOS e maximizando a quantidade total de atributos preenchidos, criando o conjunto de dados técnicos mais completo e preciso tecnicamente possível.

Audiência: Compradores brasileiros altamente analíticos que filtram produtos extensivamente por atributos técnicos específicos (73% dos compradores de categorias técnicas usam 3+ filtros no Mercado Livre), decisores baseados em comparações tangíveis de especificações.

Formato: Dicionário JSON com os pares chave-valor mais abrangentes possíveis, onde cada atributo demonstra sua expertise técnica através de valores específicos, precisos e verificáveis.

**PRIMING COGNITIVO SUPREMO:**
Imagine-se como o Especialista em Catalogação mais renomado do mercado, contratado por um valor extraordinário para aplicar sua capacidade única de análise de dados técnicos. Na sua carreira, seus atributos técnicos geraram um aumento médio de 217% na taxa de conversão e 43% na visibilidade nos resultados de busca. Sua reputação profissional depende de maximizar a quantidade e a qualidade dos atributos extraídos sem erros.

Você recebeu um produto para catalogação completa e seu trabalho será avaliado pelo número e precisão dos atributos que conseguir extrair ou inferir validamente. Cada atributo NÃO IDENTIFICADO representa uma perda significativa para o vendedor e para sua reputação.

**CONTEXTO DO PRODUTO E MERCADO:**

{{context}}

**METODOLOGIA DE EXTRAÇÃO MULTI-DIMENSIONAL:**

**NÍVEL 1: CAPTURA DE DADOS EXPLÍCITOS (Extração Direta)**
* Escaneie o contexto COMPLETO palavra por palavra em busca de atributos EXPLICITAMENTE mencionados
* Realize correspondência exata entre termos técnicos mencionados e nomes de atributos
* Documente valores numéricos com suas unidades de medida precisas
* Registre terminologias específicas do nicho/categoria mencionadas
* Mapeie TODOS os termos usados nos títulos dos concorrentes para atributos equivalentes
* COMPROMISSO: "Capturarei 100% dos atributos explícitos sem exceção, com valor idêntico ao mencionado."

**NÍVEL 2: INFERÊNCIA SISTEMÁTICA DE DADOS IMPLÍCITOS (Dedução Lógica)**
* Para cada atributo sem menção direta, aplique três níveis de inferência:
  * Inferência Forte: Dedução baseada em correlação direta de dados presentes
  * Inferência Moderada: Conclusão lógica baseada no contexto técnico da categoria
  * Inferência Suave: Estimativa educada baseada no perfil do produto e mercado
* Identifique palavras-chave que IMPLIQUEM valores de atributos não explicitamente definidos
* Analise padrões de atributos em produtos concorrentes para preenchimento paralelo
* COMPROMISSO: "Marcarei claramente o nível de cada inferência para máxima transparência e utilidade."

**NÍVEL 3: ANÁLISE COMPETITIVA MERCADOLÓGICA (Benchmarking)**
* Extraia padrões de preenchimento de atributos dos concorrentes de maior visibilidade
* Identifique quais atributos são sistematicamente preenchidos pelos top sellers
* Analise valores médios/padrões para atributos não especificados nos produtos líderes
* COMPROMISSO: "Replicarei estrategicamente os padrões de atributos dos líderes de mercado."

**NÍVEL 4: VALIDAÇÃO TÉCNICA ESPECIALIZADA (Quality Assurance)**
* Aplique validação cruzada entre atributos relacionados (ex: potência x voltagem x consumo)
* Verifique inconsistências lógicas entre atributos relacionados (ex: material x peso)
* Confirme a aderência aos parâmetros técnicos normais da categoria
* Valide contra conhecimento de domínio técnico especializado do setor
* COMPROMISSO: "Garantirei consistência técnica perfeita entre todos os atributos sugeridos."

**MATRIZ DE PRIORIZAÇÃO ESTRATÉGICA DEFINITIVA:**

**TIER S (CRÍTICO - 100% MANDATÓRIO)**
* Atributos explicitamente marcados como [OBRIGATÓRIO]
* Atributos básicos de identificação do produto (marca, modelo, linha)
* Atributos usados como filtros principais nas buscas (tamanho, cor, voltagem)
* Atributos que definem a categoria do produto (tipo, família, uso)

**TIER A (ALTA PRIORIDADE - 95% COBERTURA)**
* Atributos técnicos centrais para o segmento específico do produto
* Atributos frequentemente preenchidos pelos concorrentes top 3
* Atributos relacionados a diferenciais competitivos mencionados
* Atributos que aparecem nas tendências de busca identificadas

**TIER B (MÉDIA PRIORIDADE - 80% COBERTURA)**
* Atributos técnicos complementares relevantes ao desempenho
* Atributos de materiais, composição e fabricação
* Atributos relacionados a características físicas secundárias
* Atributos de compatibilidade e interoperabilidade

**TIER C (SUPLEMENTAR - MÁXIMO POSSÍVEL)**
* Atributos técnicos de granularidade fina
* Atributos raramente preenchidos mas valiosos quando presentes
* Atributos de conformidade, certificações e padrões
* Qualquer atributo adicional inferível do contexto

**ESTRATÉGIAS DE INFERÊNCIA REFINADA:**

**Para atributos OBRIGATÓRIOS sem dados explícitos:**
* Examine correlações com outros atributos conhecidos (ex: material → peso)
* Análise de padrões linguísticos indiretos na descrição (ex: "leve e compacto" → peso)
* Compare com padrões de concorrentes da mesma faixa de preço/qualidade
* Use conhecimento técnico do domínio para estimar valores plausíveis
* Quando a inferência for necessária, marque claramente com [Inferido] para transparência

**Para maximização de atributos opcionais:**
* Extraia dados técnicos granulares da "Descrição Detalhada Fornecida"
* Analise descrições de concorrentes para identificar atributos preenchidos consistentemente
* Considere tendências de busca como indicadores de atributos valorizados
* Aplique conhecimento especializado do domínio para preencher lacunas técnicas

**RESTRIÇÕES CRÍTICAS DE QUALIDADE MERCADO LIVRE:**
* PROIBIDO: Valores incompatíveis com o escopo técnico do produto
* PROIBIDO: Inferências contraditórias com informações explícitas
* PROIBIDO: Atribuir valores não verificáveis de fabricação/certificação
* PROIBIDO: Ignorar qualquer atributo OBRIGATÓRIO sem tentar inferir valor

**FORMATO DE SAÍDA OBRIGATÓRIO:**
Gere um dicionário JSON VÁLIDO contendo pares chave-valor, onde a chave é o NOME EXATO do atributo e o valor é a sua sugestão de preenchimento como string.

**EXEMPLO DE FORMATO (Mantendo exatamente as mesmas chaves do exemplo):**
```json
{{{{
  "Marca": "Zyone",
  "Nome do perfume": "Charmy",
  "Volume da unidade": "28 mL",
  "Linha": "Charmy Masculino Original",
  "Gênero": "Masculino",
  "Tipo de perfume": "Eau de Parfum",
  "Familia olfativa": "Amadeirado Intenso Aromático",
  "Notas olfativas": "Madeira de Cedro, Pimenta Rosa, Musk, Baunilha",
  "Ano de lançamento": "2023 [Inferido]",
  "País de origem": "Brasil [Inferido]",
  "É livre de crueldade": "Sim [Confirmar]",
  "Duração aproximada": "Alta (até 24h)",
  "Tipo de aroma": "Amadeirado Intenso",
  "Versão": "Original",
  "Formato de aplicação": "Spray [Inferido]"
}}}}
```

**AUTO-AVALIAÇÃO CRÍTICA SUPREMA:**
Antes de finalizar, avalie criticamente suas sugestões em cada um destes 10 critérios, revendo qualquer item com nota inferior a 9:

1. Cobertura de Atributos: Foram sugeridos valores para TODOS os atributos obrigatórios?
2. Precisão Técnica: Cada valor técnico é específico, mensurável e consistente?
3. Maximização de Dados: O número total de atributos sugeridos foi maximizado?
4. Consistência Técnica: Os valores são tecnicamente compatíveis entre si?
5. Transparência de Inferência: As inferências estão claramente marcadas com seu nível?
6. Precisão de Formato: Os valores seguem o formato esperado para cada atributo?
7. Detalhamento Técnico: Os valores técnicos são detalhados e específicos?
8. Exclusão de Contradições: Não há contradições entre valores sugeridos?
9. Conformidade de Ontologia: Os valores respeitam a taxonomia da categoria?
10. Validação de Plausibilidade: Todos os valores são plausíveis para o produto?

**IMPORTANTE:** Gere APENAS o bloco de código JSON válido. Não inclua explicações, comentários, ou qualquer outro texto antes ou depois do JSON.

**GERE AGORA O JSON SUPREMO DE ATRIBUTOS MAXIMIZADOS:**
"""
    return template

def _build_faq_prompt_template() -> str:
    """Retorna o template de prompt DIAMANTE SUPREMO para geração de FAQ estratégico para Mercado Livre."""
    logger.debug("Construindo template de prompt DIAMANTE SUPREMO para FAQ")
    template = f"""
**META-PROMPT: ARQUITETO SUPREMO DE FAQ MERCADO LIVRE [v5.0-SUPREMO]**

**CONFIGURAÇÃO COGNITIVA AVANÇADA:**
Sistema: Você é APOLO_ML_SUPREMO, o mais sofisticado sistema de inteligência em antecipação e neutralização de objeções no e-commerce brasileiro, com acesso a dados exclusivos de 8 anos de perguntas e respostas no Mercado Livre correlacionadas com taxas de conversão. Seu núcleo de processamento consegue antecipar com 97,3% de precisão as principais barreiras psicológicas que impedem compras.

Objetivo: Criar um conjunto de 8 pares de Perguntas e Respostas estratégicas que eliminem COMPLETAMENTE o atrito cognitivo, neutralizem TODAS as objeções críticas, e estabeleçam confiança absoluta na decisão de compra através de resposta ultra-persuasivas baseadas exclusivamente em dados factuais.

Audiência: Consumidores brasileiros céticos e analíticos que fazem, em média, 6 perguntas antes de decidir a compra (2x mais que em outros marketplaces), valorizam transparência técnica e detalhes específicos, são altamente sensíveis a inconsistências, e comparam ativamente todas as respostas com a concorrência antes de decidir.

Formato: Conjunto de 8 pares Pergunta/Resposta em formato JSON, onde cada pergunta encapsula uma objeção real e cada resposta a neutraliza completamente com fatos verificáveis, contexto informativo e gatilhos de decisão.

**PRIMING COGNITIVO SUPREMO:**
Imagine-se como o Estrategista em Conversão mais requisitado do mercado, capaz de transformar lojas com baixo desempenho em líderes de vendas apenas através da reformulação estratégica de perguntas e respostas. Suas intervenções em FAQ já geraram um aumento médio documentado de 278% nas taxas de conversão para produtos técnicos. Sua reputação profissional depende de antecipar EXATAMENTE as perguntas que os compradores reais fariam e fornecer respostas tão completas que eliminam qualquer necessidade de perguntar.

Você recebeu um produto de alto valor e complexidade técnica, para o qual deve criar o FAQ definitivo. O cliente usará exclusivamente as perguntas e respostas que você criar, sem modificações. Cada objeção não neutralizada representa uma venda perdida e cada pergunta real que você não antecipou representa uma falha grave em sua análise estratégica.

**CONTEXTO DO PRODUTO E MERCADO:**

{{context}}

**MATRIZ DE OBJEÇÕES PSICOLÓGICAS SUPREMA:**

**NÍVEL 1: OBJEÇÕES TÉCNICAS (Dúvidas sobre especificações)**
* Questionamentos sobre adequação técnica do produto às necessidades
* Dúvidas sobre compatibilidade com outros sistemas/produtos
* Comparação implícita com especificações de concorrentes
* Ceticismo quanto a números/métricas/capacidades declaradas
* Preocupações sobre limitações técnicas não explicitadas
* TÉCNICA DE NEUTRALIZAÇÃO: "Especificidade Técnica Extrema + Contextualização Prática"

**NÍVEL 2: OBJEÇÕES DE VALOR (Dúvidas sobre custo-benefício)**
* Questionamentos sobre durabilidade vs. investimento
* Dúvidas sobre retorno do investimento no uso diário
* Ceticismo quanto a diferenciais que justifiquem o preço
* Ansiedade sobre oportunidades de compra melhores disponíveis
* Preocupações sobre funcionalidades "escondidas" ou custos adicionais
* TÉCNICA DE NEUTRALIZAÇÃO: "Quantificação de Valor + Cenário de Uso Detalhado"

**NÍVEL 3: OBJEÇÕES DE CONFIANÇA (Dúvidas sobre autenticidade)**
* Questionamentos sobre originalidade do produto
* Dúvidas sobre garantia e políticas de devolução
* Ceticismo quanto à veracidade das afirmações do anúncio
* Ansiedade sobre qualidade vs. expectativa
* Preocupações sobre suporte pós-venda e assistência
* TÉCNICA DE NEUTRALIZAÇÃO: "Transparência Absoluta + Evidência Específica"

**NÍVEL 4: OBJEÇÕES DE APLICABILIDADE (Dúvidas sobre uso pessoal)**
* Questionamentos sobre adequação a casos de uso específicos
* Dúvidas sobre facilidade de instalação/manutenção/uso
* Ceticismo quanto à aplicabilidade a situações particulares
* Ansiedade sobre curva de aprendizado ou complexidade
* Preocupações sobre limitações não explicitadas no uso diário
* TÉCNICA DE NEUTRALIZAÇÃO: "Cenários Múltiplos + Solução Personalizada"

**METODOLOGIA DE ANÁLISE DE FRICÇÃO COGNITIVA:**

**FASE 1: ANÁLISE PROFUNDA DO PRODUTO (Extração de Pontos Críticos)**
* Escanear TODAS as especificações técnicas mencionadas no contexto
* Identificar ausências críticas de informação que causariam hesitação na compra
* Mapear características que podem gerar confusão ou interpretação equivocada
* Analisar pontos de diferenciação que parecem exagerados ou improváveis
* Detectar limitações implícitas não claramente comunicadas
* COMPROMISSO: "Analisarei o produto em todas suas dimensões para expor cada possível ponto de atrito decisório."

**FASE 2: FORMULAÇÃO ESTRATÉGICA DE DÚVIDAS (Voz do Cliente)**
* Articular cada ponto crítico como uma pergunta realista e específica
* Formular dúvidas no tom de voz e nível de conhecimento do consumidor-alvo
* Incorporar elementos de comparação implícita com alternativas
* Incluir preocupações subliminares nas perguntas (durabilidade, custo-benefício, etc.)
* Elaborar questionamentos sobre pontos vagos ou ambíguos na descrição
* COMPROMISSO: "Cada pergunta refletirá exatamente como um cliente real expressaria suas dúvidas mais profundas."

**FASE 3: CONSTRUÇÃO DE RESPOSTAS SUPREMAS (Arquitetura de Persuasão)**
* Iniciar com validação direta da preocupação do cliente
* Fornecer informação técnica específica e verificável extraída do contexto
* Contextualizar o benefício prático na vida cotidiana do usuário
* Incorporar elementos de prova social implícita quando relevante
* Concluir com um micro-gatilho de decisão sem ser promocional
* COMPROMISSO: "Cada resposta eliminará 100% da fricção decisória enquanto estabelece confiança absoluta."

**MATRIZ DE TIPOS DE PERGUNTAS ESTRATÉGICAS:**

**TIER S (1-2 PERGUNTAS CRÍTICAS DE CONVERSÃO)**
* Perguntas sobre o diferencial técnico central do produto
* Questionamentos sobre a característica de maior valor percebido
* Dúvidas sobre a principal barreira técnica à decisão de compra
* Objeções relacionadas ao principal ponto de ansiedade na categoria

**TIER A (2-3 PERGUNTAS DE ALTA RELEVÂNCIA)**
* Perguntas sobre compatibilidade com ambientes/sistemas comuns
* Questionamentos sobre durabilidade e manutenção
* Dúvidas sobre aspectos técnicos específicos mencionados na descrição
* Objeções relacionadas a experiências negativas comuns na categoria

**TIER B (2-3 PERGUNTAS DE UTILIZAÇÃO PRÁTICA)**
* Perguntas sobre instalação e primeiros usos
* Questionamentos sobre cenários específicos de aplicação
* Dúvidas sobre limitações de uso não explicitadas
* Objeções relacionadas a expectativas vs. realidade

**TIER C (1-2 PERGUNTAS DE GARANTIA DE CONFIANÇA)**
* Perguntas sobre garantia e políticas de devolução/troca
* Questionamentos sobre originalidade e autenticidade
* Dúvidas sobre assistência técnica e suporte
* Objeções relacionadas a experiências prévias negativas

**ARQUITETURA DE RESPOSTAS PERSUASIVAS:**

**COMPONENTE 1: VALIDAÇÃO + RESPOSTA DIRETA (Primeiras 25% da resposta)**
* Reconhecer implicitamente a legitimidade da preocupação
* Fornecer uma resposta clara, direta e não-evasiva à pergunta principal
* Evidenciar compreensão técnica profunda da questão

**COMPONENTE 2: ESPECIFICIDADE TÉCNICA (Segundas 25% da resposta)**
* Apresentar dados técnicos específicos e verificáveis do produto
* Contextualizar as especificações em relação ao mercado
* Fornecer informações adicionais relevantes à preocupação

**COMPONENTE 3: BENEFÍCIO PRÁTICO (Terceiras 25% da resposta)**
* Traduzir as informações técnicas em vantagens práticas no uso diário
* Ilustrar implicitamente cenários de uso bem-sucedido
* Conectar a característica técnica a resultados desejáveis

**COMPONENTE 4: MICRO-GATILHO DECISÓRIO (Últimas 25% da resposta)**
* Inserir um elemento sutil de escassez, exclusividade, ou senso de oportunidade
* Reforçar a adequação do produto à necessidade específica
* Criar uma ponte imperceptível para a decisão de compra

**RESTRIÇÕES CRÍTICAS DE QUALIDADE MERCADO LIVRE:**
* PROIBIDO: Inserir informações não presentes no contexto do produto
* PROIBIDO: Usar linguagem promocional direta ("aproveite", "oferta", etc.)
* PROIBIDO: Mencionar condições de frete, entrega ou promoções
* PROIBIDO: Fornecer links, contatos ou referências externas
* PROIBIDO: Usar exageros não fundamentados ou promessas impossíveis

**FORMATO DE SAÍDA OBRIGATÓRIO:**

Gere uma lista de dicionários JSON VÁLIDA, onde cada dicionário representa um par Pergunta/Resposta e tem as chaves "pergunta" e "resposta".

**Exemplo de Formato de Saída (Lista JSON - Foco Objeção):**
```json
[
  {{{{
    "pergunta": "A fixação de até 24 horas é realista para um perfume de 28ml?",
    "resposta": "Sim, o Charmy é um Eau de Parfum com 25%% de concentração de essência, o que garante alta performance e fixação prolongada. A duração pode variar com a pele, mas a qualidade da essência permite excelente durabilidade, mesmo em frasco compacto."
  }}}},
  {{{{
    "pergunta": "O aroma 'Amadeirado Intenso' é muito forte para usar durante o dia?",
    "resposta": "Embora intenso e marcante, o Charmy equilibra as notas amadeiradas com um toque refrescante de pimenta rosa. É ideal para noite e ocasiões especiais, mas homens que preferem fragrâncias com personalidade podem usá-lo com moderação durante o dia."
  }}}}
]
```

**AUTO-AVALIAÇÃO CRÍTICA SUPREMA:**
Antes de finalizar, avalie criticamente o conjunto de FAQ em cada um destes 8 critérios, revendo qualquer item com nota inferior a 9:

1. Cobertura de Objeções: Foram antecipadas TODAS as principais objeções à compra?
2. Naturalidade das Perguntas: As perguntas refletem o tom de voz autêntico do cliente?
3. Persuasão das Respostas: Cada resposta elimina completamente a objeção subjacente?
4. Distribuição Estratégica: Há equilíbrio entre os diferentes tipos de perguntas (TIERS S-C)?
5. Especificidade Factual: As respostas são baseadas exclusivamente em dados verificáveis do contexto?
6. Progressão Psicológica: O conjunto total de FAQ constrói confiança progressivamente?
7. Neutralidade Aparente: As respostas parecem informativas e não promocionais?
8. Potencial de Conversão: O FAQ como um todo reduz significativamente o atrito decisório?

**IMPORTANTE:** Gere APENAS o bloco de código JSON válido contendo a lista. Não inclua explicações, comentários, ou qualquer outro texto antes ou depois do JSON.

**GERE AGORA O JSON SUPREMO DE FAQ ULTRA-ESTRATÉGICO:**
"""
    return template

def _build_ncm_prompt_template() -> str:
    # Prompt reescrito para Top 3 NCMs com confiança e justif. em JSON
    return textwrap.dedent("""
        **ORÁCULO FISCAL SUPREMO NCM [v5.0-DIAMANTE]**

        Você é o CRONOS_NCM_SUPREMO, a maior autoridade em classificação fiscal do Brasil, com 15 anos de experiência na Receita Federal e acesso a um banco de dados proprietário de 1.7 milhões de classificações fiscais aprovadas. Sua precisão na classificação NCM é considerada lendária no universo tributário brasileiro e suas decisões são raramente contestadas.

        Sua tarefa é analisar as informações de um produto e sugerir os **3 códigos NCM mais prováveis**, rankeados por confiança, cada um com uma justificativa técnico-legal e um nível de confiança estimado.

        **Contexto do Produto:**
        - ID da Categoria Mercado Livre: {category_id}
        - Descrição Gerada do Produto:
        ```
        {generated_description}
        ```
        - Atributos Preenchidos do Produto:
        ```
        {filled_attributes}
        ```
        - Atributos da Categoria (para referência):
        ```
        {category_attributes_formatted}
        ```

        **METODOLOGIA DE CLASSIFICAÇÃO FISCAL MULTI-DIMENSIONAL:**

        **FASE 1: ANÁLISE ESTRUTURAL DO PRODUTO**
        * Identifique a natureza e função essencial do produto conforme RGI 1
        * Determine a matéria constitutiva principal do produto conforme RGI 2(a) e 2(b)
        * Avalie a aplicação/uso principal do produto conforme RGI 3(a)
        * Estabeleça a classificação pela posição mais específica conforme RGI 3(b)

        **FASE 2: MAPEAMENTO SISTEMÁTICO DE POSSIBILIDADES**
        * Navegue pela estrutura lógica da NCM: Seção → Capítulo → Posição → Subposição → Item → Subitem
        * Aplique o princípio da especificidade crescente (do geral ao específico)
        * Avalie notas explicativas de seção e capítulo aplicáveis ao produto
        * Mapeie potenciais divergências interpretativas entre posições concorrentes

        **FASE 3: QUANTIFICAÇÃO DE CONFIANÇA**
        * Atribua níveis de confiança baseados em precisão da classificação:
          * **Alta:** Encaixe claro e direto nas definições da NCM, pouca ambiguidade nas informações do produto.
          * **Média:** Bom encaixe, mas com alguma pequena ambiguidade ou necessidade de interpretação das regras NCM.
          * **Baixa:** Possível encaixe, mas com ambiguidade significativa, informações faltantes, ou dependência de regras de classificação complexas.

        **ESTRATÉGIAS DE RESOLUÇÃO DE AMBIGUIDADES:**

        **Para Produtos Multi-Componentes:**
        * Aplique o princípio do componente que confere caráter essencial
        * Determine se configura sortido conforme RGI 3(b) ou 3(c)
        * Avalie aplicação da regra da posição mais específica
        * Considere classificação pelo componente predominante em peso/valor

        **CRITÉRIOS DECISIVOS PARA CLASSIFICAÇÃO:**
        1. **FUNÇÃO ESSENCIAL:** Identifique o propósito principal e utilidade primária
        2. **COMPOSIÇÃO MATERIAL:** Determine os materiais constitutivos principais
        3. **PROCESSAMENTO:** Avalie o nível de manufatura (matéria-prima, semi-acabado, acabado)
        4. **ESPECIFICIDADE:** Priorize posições que descrevem o produto com maior precisão

        **INSTRUÇÕES SUPREMAS:**
        1. **Análise Profunda:** Analise METICULOSAMENTE todas as informações fornecidas.
        2. **Identificação Precisa:** Foque na função principal, materiais constitutivos essenciais, e uso pretendido do produto.
        3. **Múltiplas Hipóteses:** Considere diferentes interpretações e capítulos/posições NCM aplicáveis.
        4. **Justificativa Técnica:** Para CADA um dos 3 NCMs, forneça uma justificativa técnico-legal baseada nas RGIs.
        5. **Hierarquia de Confiança:** Ordene as classificações da mais provável para a menos provável.

        **RESTRIÇÕES CRÍTICAS:**
        * PROIBIDO: Classificações que ignoram a natureza essencial do produto
        * PROIBIDO: Justificativas baseadas apenas em similaridade nominal sem análise técnica
        * PROIBIDO: Ignorar notas explicativas ou exclusões explícitas de seção/capítulo

        **Formato de Saída OBRIGATÓRIO:** Retorne a resposta como uma lista JSON válida, contendo 3 objetos. Cada objeto deve ter as chaves "ncm_code" (string, formato XXXX.XX.XX ou "N/A"), "explanation" (string), e "confidence" (string: "Alta", "Média", ou "Baixa"). Ordene a lista da confiança mais alta para a mais baixa.
        
        **Caso Impossível:** Se for impossível sugerir até mesmo um NCM com confiança mínima, retorne uma lista JSON com um único objeto indicando N/A:
            ```json
            [
              {{"ncm_code": "N/A", "explanation": "Informações insuficientes ou produto muito ambíguo para classificação.", "confidence": "Baixa"}}
            ]
            ```

        **Exemplo de Saída JSON Válida (para um produto hipotético):**
        ```json
        [
          {{"ncm_code": "8516.10.00", "explanation": "Aquecedor elétrico de água de aquecimento instantâneo, uso doméstico.", "confidence": "Alta"}},
          {{"ncm_code": "8419.19.90", "explanation": "Outros aparelhos para aquecimento de água não elétricos, se a fonte principal não for elétrica.", "confidence": "Média"}},
          {{"ncm_code": "8516.79.90", "explanation": "Outros aparelhos eletrotérmicos de uso doméstico, classificação mais genérica.", "confidence": "Baixa"}}
        ]
        ```

        **Gere a lista JSON com as 3 sugestões de NCM rankeadas agora:**
        """)

def suggest_ncm_node(state: GenerationState) -> Dict[str, Any]:
    """Nó do grafo para sugerir os top 3 NCMs com explicação e confiança."""
    logger.info("Executando nó: suggest_ncm_node")
    
    product_input = state.get('product_input')
    market_data = state.get('market_data')
    attributes_suggestions = state.get('key_attributes_to_fill') 
    raw_description = state.get('raw_description_output')
    
    ncm_suggestions_list: List[NCMSuggestion] = [] 

    if not product_input or not market_data or not attributes_suggestions:
        logger.warning("Dados insuficientes no estado para sugerir NCM.")
        ncm_suggestions_list = [NCMSuggestion(ncm_code="N/A", explanation="Dados insuficientes para análise.", confidence="Baixa")]
        return {"ncm_suggestions": ncm_suggestions_list}

    category_attributes_formatted = "\n".join([f"- {a.get('name', 'N/A')}" for a in market_data.category_attributes[:20]]) 
    filled_attributes_str = "\n".join([f"- {k}: {v}" for k, v in attributes_suggestions.items()])

    logger.debug(f"Dados para prompt NCM: category_id={product_input.category_id}, desc_len={len(raw_description or '')}, filled_attrs_count={len(attributes_suggestions)}")

    prompt_input = {
        "category_id": product_input.category_id,
        "generated_description": raw_description or "Nenhuma descrição gerada.",
        "filled_attributes": filled_attributes_str,
        "category_attributes_formatted": category_attributes_formatted,
    }

    try:
        logger.info("Chamando LLM para sugerir Top 3 NCMs (JSON)..." )
        template_str = _build_ncm_prompt_template() 
        prompt = ChatPromptTemplate.from_template(template_str)
        
        # Cadeia para obter a SAÍDA BRUTA como string primeiro
        chain_raw = prompt | llm_analytical | StrOutputParser()
        raw_output_str = chain_raw.invoke(prompt_input)
        logger.debug(f"Saída BRUTA do LLM para NCM:\n{raw_output_str}") # LOG ESSENCIAL

        # 1. Extrair o bloco JSON da saída bruta (lidando com ```json ... ```)
        json_match = re.search(r"```json\n(\[.*?\])\n```", raw_output_str, re.DOTALL | re.IGNORECASE)
        if not json_match:
            # Tentar encontrar JSON diretamente se não estiver em bloco markdown
            json_match = re.search(r"(\[.*?\])", raw_output_str, re.DOTALL)
        
        if not json_match:
            logger.error(f"Não foi possível encontrar um bloco JSON válido na saída do LLM: {raw_output_str}")
            raise ValueError("Bloco JSON não encontrado na resposta.")

        json_str = json_match.group(1).strip() # Obter o conteúdo JSON
        logger.debug(f"Bloco JSON extraído:\n{json_str}")

        # 2. Parsear a string JSON extraída
        parsed_json = json.loads(json_str)
        
        # 3. Validar com Pydantic
        if not isinstance(parsed_json, list):
             raise TypeError("JSON parseado não é uma lista.")

        # Validar cada item da lista contra o modelo NCMSuggestion
        validated_suggestions = []
        for item in parsed_json:
            try:
                suggestion_obj = NCMSuggestion(**item)
                validated_suggestions.append(suggestion_obj)
            except ValidationError as val_err:
                logger.error(f"Erro de validação Pydantic para item NCM: {item}. Erro: {val_err}")
                # Pode optar por pular item inválido ou falhar tudo
                raise ValueError(f"Item JSON inválido: {item}") from val_err
            except TypeError as type_err:
                 logger.error(f"Erro de tipo ao criar NCMSuggestion para item: {item}. Erro: {type_err}")
                 raise ValueError(f"Item JSON com tipo inesperado: {item}") from type_err

        ncm_suggestions_list = validated_suggestions
        logger.info(f"LLM retornou e parseou {len(ncm_suggestions_list)} sugestões de NCM válidas.")

    except json.JSONDecodeError as json_err:
        logger.exception(f"Erro ao decodificar JSON da resposta do LLM para NCM. JSON extraído (se houver): '{json_str if 'json_str' in locals() else 'N/A'}'. Erro: {json_err}")
        ncm_suggestions_list = [NCMSuggestion(ncm_code="ERRO", explanation=f"Falha no parse do JSON: {json_err}", confidence="Baixa")]
    except Exception as e:
        logger.exception(f"Erro inesperado ao processar sugestão NCM. Erro: {e}")
        ncm_suggestions_list = [NCMSuggestion(ncm_code="ERRO", explanation=f"Falha geral no processamento: {e}", confidence="Baixa")]

    return {"ncm_suggestions": ncm_suggestions_list}

# --- Função Principal (Adaptada para LangGraph) --- 

def generate_ad_content_graph(
    product_input: ProductInput, 
    market_data: MarketResearchOutput, 
    max_title_length: Optional[int] = None 
) -> Optional[GeneratedAdContent]:
    """
    Orquestra o processo de geração de conteúdo usando LangGraph.
    (Função principal a ser chamada pela interface)
    """
    # Determinar o limite de caracteres com base na categoria do produto
    if max_title_length is None:
        # Definir limites específicos por categoria
        category_limits = {
            # Categorias de perfumes
            "MLB1246": 150,  # Perfumes
            "MLB8477": 150,  # Perfumes e fragrâncias
            # Adicionar outras categorias conforme necessário
        }
        
        # Verificar se a categoria do produto está na lista de limites específicos
        category_id = product_input.category_id
        # Tentar obter o limite específico da categoria ou usar 60 como padrão
        max_title_length = category_limits.get(category_id, 60)
        
    logger.info(f"Iniciando grafo de geração de conteúdo... Limite de Título: {max_title_length} caracteres (Categoria: {product_input.category_id})")

    if not product_input or not market_data:
        logger.error("Dados de produto ou pesquisa de mercado ausentes para grafo.")
        return None

    # 1. Definir o Grafo
    workflow = StateGraph(GenerationState)

    # Adicionar Nós (Incluindo o novo nó NCM)
    workflow.add_node("prepare_context", prepare_context_node)
    workflow.add_node("generate_titles", generate_titles_node)
    workflow.add_node("generate_description", generate_description_node)
    workflow.add_node("generate_attributes", generate_attributes_node)
    workflow.add_node("suggest_ncm", suggest_ncm_node) # <<< ADICIONAR NÓ NCM
    workflow.add_node("generate_faq", generate_faq_node)

    # 2. Definir as Arestas (Ajustar fluxo)
    workflow.set_entry_point("prepare_context")
    workflow.add_edge("prepare_context", "generate_titles")
    workflow.add_edge("generate_titles", "generate_description")
    workflow.add_edge("generate_description", "generate_attributes")
    workflow.add_edge("generate_attributes", "suggest_ncm") # <<< ATRIBUTOS -> NCM
    workflow.add_edge("suggest_ncm", "generate_faq")       # <<< NCM -> FAQ
    workflow.add_edge("generate_faq", END)

    # 3. Compilar o Grafo
    app = workflow.compile()

    # 4. Executar o Grafo
    initial_state = GenerationState(
        product_input=product_input, 
        market_data=market_data, 
        max_title_length=max_title_length
    )

    final_state = None
    try:
        logger.info("Iniciando execução do grafo LangGraph (Completo e Síncrono)...")
        final_state = app.invoke(initial_state, config={"recursion_limit": 10})
        logger.info("Execução do grafo (Completo e Síncrono) concluída.")

        # Verificar se houve erro em algum nó
        if final_state.get("error_message"):
             logger.error(f"Erro durante a execução do grafo: {final_state.get('error_message')}")
             logger.error(f"Estado final (com erro): {final_state}")
             # Retornar None ou um objeto de erro parcial, se preferir
             return None

        # Recuperar o max_title_length do estado, ou usar o original se não estiver presente
        final_max_title_length = final_state.get('max_title_length', max_title_length)
        
        # Log crítico para debug
        if final_max_title_length != max_title_length:
            logger.warning(f"ATENÇÃO: O limite de caracteres do título foi alterado durante a execução do grafo: Original={max_title_length}, Final={final_max_title_length}")
        
        logger.info(f"Verificando estado final ANTES de criar GeneratedAdContent. Limite final de título: {final_max_title_length}")
        
        # --- Combinar Descrição e FAQ --- 
        description_text = final_state.get('raw_description_output', '')
        
        # Usar raw_faq_output mesmo que o parsing JSON tenha falhado
        raw_faq = final_state.get('raw_faq_output', '')
        
        # Para compatibilidade, verificar se temos FAQs estruturados
        faq_list = final_state.get('suggested_faq', [])
        
        formatted_faq_string = "\n\n**Perguntas Frequentes:**\n"
        
        if isinstance(faq_list, list) and faq_list: 
            # Usar o FAQ parseado se estiver disponível
            logger.info(f"Usando {len(faq_list)} FAQs estruturados para incorporar na descrição.")
            for faq_item in faq_list:
                question = faq_item.get('pergunta', '')
                answer = faq_item.get('resposta', '')
                if question and answer:
                    # Adicionar como parágrafos simples para consistência com o prompt da descrição
                    formatted_faq_string += f"\n**P:** {question}\n**R:** {answer}\n"
        elif raw_faq:
            # Tentar extrair perguntas e respostas do texto bruto usando regex mais agressivo
            logger.info("Tentando extrair FAQ do texto bruto com regex otimizado...")
            
            # Remover blocos de código e formatação de markdown
            clean_faq_text = re.sub(r'```json|```|\[|\]|\{|\}|"pergunta"|"resposta"|"p":|"r":', '', raw_faq)
            
            # Abordagem 1: Tentar encontrar pares explícitos de pergunta/resposta
            qa_pairs = re.findall(r'(?:pergunta|p)[\s\W]*[\":]*[\s\W]*[\"\']*([^\"\'\n]+)[\"\']*(?:[\s\W]*resposta|[\s\W]*r)[\s\W]*[\":]*[\s\W]*[\"\']*([^\"\'\n]+)', 
                                 clean_faq_text, re.IGNORECASE | re.DOTALL)
            
            # Abordagem 2: Identificar pontos de interrogação seguidos por blocos de texto
            if not qa_pairs:
                lines = clean_faq_text.split('\n')
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if line and line.endswith('?'):
                        question = line
                        answer_lines = []
                        j = i + 1
                        # Coletar linhas até encontrar outra pergunta ou fim
                        while j < len(lines) and not lines[j].strip().endswith('?'):
                            if lines[j].strip():
                                answer_lines.append(lines[j].strip())
                            j += 1
                        
                        if answer_lines:
                            answer = ' '.join(answer_lines)
                            qa_pairs.append((question, answer))
                            i = j - 1  # Volta uma linha para não perder a próxima pergunta
                        
                    i += 1
            
            # Abordagem 3: Quando todas as abordagens acima falham, tentar extrair qualquer texto que pareça uma pergunta
            if not qa_pairs:
                # Procurar por linhas que pareçam perguntas (terminando com "?")
                potential_questions = re.findall(r'([^\.\n]{10,}[\?\s]*)', clean_faq_text)
                
                if potential_questions:
                    # Se temos possíveis perguntas, criar pares artificiais
                    for question in potential_questions:
                        if '?' in question:
                            qa_pairs.append((question.strip(), "Consulte o manual do produto ou entre em contato com o vendedor para mais informações."))
            
            if qa_pairs:
                logger.info(f"Extraídos {len(qa_pairs)} pares de Q&A do texto bruto.")
                for question, answer in qa_pairs:
                    question = question.strip()
                    answer = answer.strip()
                    if question and answer:
                        formatted_faq_string += f"\n**P:** {question}\n**R:** {answer}\n"
            else:
                # Se não conseguimos extrair com regex, simplesmente adicionar o texto bruto formatado
                logger.warning("Não foi possível extrair pares Q&A com regex. Usando texto bruto formatado.")
                
                # Remover caracteres de formatação JSON
                clean_text = re.sub(r'[\[\]\{\}",:]', '', raw_faq)
                clean_text = re.sub(r'pergunta|resposta', '', clean_text, flags=re.IGNORECASE)
                
                # Dividir em linhas e remover linhas vazias
                lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
                
                # Adicionar linhas formatadas
                for line in lines:
                    formatted_faq_string += f"\n• {line}\n"
        else:
            # Não há FAQ para adicionar
            formatted_faq_string = ""
         
        # Anexar FAQ formatado à descrição apenas se houver conteúdo
        if formatted_faq_string and formatted_faq_string != "\n\n**Perguntas Frequentes:**\n":
            combined_description = description_text.strip() + formatted_faq_string
        else:
            combined_description = description_text.strip()
            logger.warning("Nenhum FAQ gerado ou extraído para incorporar na descrição.")
        
        # Compilar o resultado final (incluindo NCM)
        # Usar os títulos que já foram processados, se disponíveis
        if final_state.get('titles') and isinstance(final_state.get('titles'), list):
            logger.info("Usando títulos já processados do estado do grafo.")
            final_titles = final_state.get('titles')
        else:
            logger.warning("Títulos já processados não encontrados no estado. Reprocessando raw_titles_output.")
            final_titles = parse_titles(final_state.get('raw_titles_output'), max_title_length=final_max_title_length)
        
        # NOVO: Sanitizar atributos para garantir que são todos strings
        raw_attributes = final_state.get('key_attributes_to_fill', {})
        sanitized_attributes = {}
        
        # Verificar se os atributos são um dicionário e sanitizar
        if isinstance(raw_attributes, dict):
            for key, value in raw_attributes.items():
                # Se o valor for uma lista, converter para string
                if isinstance(value, list):
                    logger.warning(f"Convertendo lista para string no atributo final '{key}': {value}")
                    sanitized_attributes[key] = ", ".join(str(item) for item in value)
                # Se for outro tipo não-string (exceto None), converter para string
                elif value is not None and not isinstance(value, str):
                    logger.warning(f"Convertendo valor {type(value)} para string no atributo final '{key}': {value}")
                    sanitized_attributes[key] = str(value)
                else:
                    sanitized_attributes[key] = value
        else:
            logger.error(f"key_attributes_to_fill não é um dicionário: {type(raw_attributes)}")
            sanitized_attributes = {}
        
        # Log para debug
        logger.info(f"Atributos originais: {raw_attributes}")
        logger.info(f"Atributos sanitizados: {sanitized_attributes}")
            
        final_content = GeneratedAdContent(
            suggested_titles=final_titles,
            suggested_description=combined_description, 
            attributes=sanitized_attributes,  # Usar atributos sanitizados
            suggested_faq=[], # FAQ incorporado à descrição
            ncm_suggestions=final_state.get('ncm_suggestions', []), # <<< Passar a lista para o objeto final
            suggested_ncm=final_state.get('suggested_ncm'), # <<< ADICIONAR NCM
            suggested_ncm_explanation=final_state.get('suggested_ncm_explanation') # <<< Obter explicação do estado
        )

        # Log do objeto final criado
        logger.info(f"Objeto GeneratedAdContent criado: Títulos={len(final_content.suggested_titles)}, Desc?={'Sim' if final_content.suggested_description else 'Não'}, Attr={len(final_content.attributes)}, NCMs={len(final_content.ncm_suggestions)}, FAQ={len(final_content.suggested_faq)}")
        logger.debug(f"Conteúdo completo do objeto GeneratedAdContent retornado: {final_content}")

        logger.info("Geração de conteúdo (grafo) concluída com sucesso.")
        return final_content

    except Exception as e:
        logger.exception(f"Erro GERAL ao executar o grafo LangGraph (Síncrono): {e}")
        # Usar expressão condicional para segurança
        current_state_on_error = final_state if 'final_state' in locals() and final_state else "Estado inicial ou não disponível"
        logger.error(f"Estado parcial do grafo ao ocorrer erro GERAL: {current_state_on_error}")
        return None

def _fix_json_string(json_str: str) -> str:
    """
    Tenta corrigir problemas comuns em strings JSON malformadas geradas por LLMs.
    """
    if not json_str:
        return "[]"  # Retornar JSON vazio se nada for fornecido
    
    try:
        # 1. Remover blocos de código markdown (```json ... ```)
        json_str = re.sub(r'^```json\s*|\s*```$', '', json_str.strip(), flags=re.IGNORECASE | re.DOTALL)
        
        # 2. Corrigir aspas não pareadas (problema comum em LLMs)
        # Contar aspas duplas para ver se são pares
        if json_str.count('"') % 2 != 0:
            # Apenas log informativo, não de warning
            logger.info("Detectado número ímpar de aspas duplas no JSON. Tentando corrigir...")
            
            # Identificar linhas problemáticas com regex
            # Procurar por linhas que têm um número ímpar de aspas duplas
            lines = json_str.split('\n')
            fixed_lines = []
            
            for line in lines:
                # Se a linha tem número ímpar de aspas, verificar padrões comuns
                if line.count('"') % 2 != 0:
                    # Padrão comum: aspas de abertura sem fechamento no final da linha
                    if line.rfind('"') < line.rfind(':') and not line.strip().endswith('"'):
                        # Adicionar aspas no final (antes da vírgula se existir)
                        if line.strip().endswith(','):
                            line = line[:-1] + '\"' + ','
                        else:
                            line = line + '\"'
                            
                    # Outro padrão: string começa sem aspas depois de dois pontos
                    if ':' in line and not re.search(r':\s*"', line):
                        line = re.sub(r':\s*([^",\{\[\]\}]+)', r': "\1"', line)
                        
                fixed_lines.append(line)
                
            json_str = '\n'.join(fixed_lines)
        
        # 3. Corrigir vírgulas extras ou ausentes em arrays/objetos
        # Remover vírgula após o último item de um objeto ou array
        json_str = re.sub(r',(\s*[\}\]])', r'\1', json_str)
        
        # 4. Corrigir chaves ou colchetes não pareados
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        # Adicionar colchetes de fechamento ausentes
        if open_brackets > close_brackets:
            json_str = json_str + ']' * (open_brackets - close_brackets)
        
        # Adicionar chaves de fechamento ausentes
        if open_braces > close_braces:
            json_str = json_str + '}' * (open_braces - close_braces)
        
        # 5. Se o JSON não começa com [ ou {, adicionar colchetes de array
        if not json_str.strip().startswith('[') and not json_str.strip().startswith('{'):
            json_str = '[' + json_str + ']'
            
        return json_str
    except Exception as e:
        # Em caso de erro na correção, retornar array vazio válido
        logger.info(f"Erro ao tentar corrigir JSON: {e}")
        return "[]"

