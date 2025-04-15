# -*- coding: utf-8 -*-
# FINALIDADE: Coleta e valida os dados iniciais do produto fornecidos pelo usuário.
# Módulo para coletar e validar o input inicial do produto do usuário 

import asyncio
from .data_models import ProductInput # Importa a estrutura de dados


async def collect_product_input() -> ProductInput:
    """Coleta interativamente os dados básicos do produto do usuário."""
    print("\n--- Coleta de Informações do Produto ---")
    print("Por favor, forneça os detalhes do produto para gerar o anúncio.")

    while True:
        category_id = await asyncio.to_thread(input, "- ID da Categoria (Obrigatório, ex: MLB6284): ")
        if category_id.strip():
            break
        print("  Erro: O ID da Categoria é obrigatório.")

    while True:
        product_name_base = await asyncio.to_thread(input, "- Nome Base do Produto (Obrigatório): ")
        if product_name_base.strip():
            break
        print("  Erro: O Nome Base do Produto é obrigatório.")

    # Campos opcionais
    brand = await asyncio.to_thread(input, "- Marca (Opcional, recomendado): ")
    model = await asyncio.to_thread(input, "- Modelo (Opcional): ")
    ean = await asyncio.to_thread(input, "- EAN / GTIN (Opcional, recomendado): ")

    # Limpar espaços extras
    category_id = category_id.strip()
    product_name_base = product_name_base.strip()
    brand = brand.strip() if brand else None
    model = model.strip() if model else None
    ean = ean.strip() if ean else None

    # Criar e retornar o objeto ProductInput
    product_data = ProductInput(
        category_id=category_id,
        product_name_base=product_name_base,
        brand=brand,
        model=model,
        ean=ean
    )

    print("\n--- Dados Coletados ---")
    print(f"  Categoria: {product_data.category_id}")
    print(f"  Nome Base: {product_data.product_name_base}")
    print(f"  Marca: {product_data.brand or 'N/A'}")
    print(f"  Modelo: {product_data.model or 'N/A'}")
    print(f"  EAN: {product_data.ean or 'N/A'}")
    print("-------------------------")

    return product_data

# Exemplo de como usar (pode ser movido para main_flow.py depois)
# async def main_test():
#     product_info = await collect_product_input()
#     # Proximos passos: passar product_info para market_research
# 
# if __name__ == "__main__":
#     asyncio.run(main_test()) 