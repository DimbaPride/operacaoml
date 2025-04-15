# -*- coding: utf-8 -*-
# FINALIDADE: Define as estruturas (dataclasses) para organizar os dados do agente.
"""Define estruturas de dados (dataclasses ou Pydantic models) 
para representar o input do produto, os resultados da pesquisa e o output do anúncio.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

@dataclass
class ProductInput:
    """Dados iniciais fornecidos pelo usuário sobre o produto."""
    category_id: str
    product_name_base: str
    brand: Optional[str] = None
    model: Optional[str] = None
    ean: Optional[str] = None
    # Adicionar mais campos conforme necessário (specs, manual_text, etc.)
    manual_competitor_urls: List[str] = field(default_factory=list) # Novo campo para URLs de concorrentes manuais
    detailed_description: Optional[str] = None
    # specifications: Dict[str, str] = field(default_factory=dict)

# --- Nova Dataclass para Info do Concorrente ---
@dataclass
class CompetitorInfo:
    """Armazena informações extraídas de um anúncio concorrente."""
    mlb_id: str
    title: str
    price: Optional[float] = None
    listing_type_id: Optional[str] = None
    shipping_mode: Optional[str] = None
    free_shipping: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict) # { "attribute_name": "value_name", ... }
    description: str = ""

@dataclass
class MarketResearchOutput:
    """Resultados da pesquisa de mercado."""
    trends: List[Dict[str, Any]] = field(default_factory=list)
    category_attributes: List[Dict[str, Any]] = field(default_factory=list)
    # competitor_titles: List[str] = field(default_factory=list) # Campo antigo removido
    competitor_analysis: List[CompetitorInfo] = field(default_factory=list) # Novo campo
    # Adicionar mais campos (detalhes de concorrentes, etc.)

@dataclass
class AdOutput:
    """Sugestões geradas para o anúncio."""
    suggested_titles: List[str] = field(default_factory=list)
    suggested_attributes: Dict[str, Any] = field(default_factory=dict) # Atributos com valores sugeridos
    missing_required_attributes: List[str] = field(default_factory=list) # Atributos obrigatórios sem valor
    suggested_description: str = ""
    suggested_faq: List[Dict[str, str]] = field(default_factory=list) # Lista de {q: pergunta, a: resposta}

# Modelo para uma única sugestão de NCM
class NCMSuggestion(BaseModel):
    ncm_code: str = Field(..., description="Código NCM sugerido (XXXX.XX.XX ou N/A).")
    explanation: str = Field(..., description="Justificativa para a sugestão do NCM.")
    confidence: str = Field(..., description="Nível de confiança estimado (ex: Alta, Média, Baixa).")

class GeneratedAdContent(BaseModel):
    """Representa o conteúdo final gerado para o anúncio."""
    suggested_titles: List[str] = Field(default_factory=list, description="Lista de sugestões de títulos otimizados para o anúncio.")
    suggested_description: str = Field(default="", description="Sugestão de texto completo para a descrição do anúncio.")
    attributes: Dict[str, Optional[str]] = Field(default_factory=dict, description="Dicionário com sugestões para os atributos chave da ficha técnica.")
    suggested_faq: List[Dict[str, str]] = Field(default_factory=list, description="Lista de dicionários, cada um com 'pergunta' e 'resposta' para o FAQ.")
    ncm_suggestions: List[NCMSuggestion] = Field(default_factory=list, description="Lista das top 3 sugestões de NCM rankeadas por confiança.") 