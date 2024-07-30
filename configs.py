import streamlit as st

MODEL_NAME = "gpt-4"
RETRIEVAL_SEARCH_TYPE = "mmr"  # MMR ou Semantic Search
RETRIEVAL_KWARGS = {
    "k": 5,
    "fetch_k": 30,
}  # K = quantidade de documentos / fetch_k = quantas buscas semânticas
PROMPT = """
Você é um advogado brasileiro. Suas funções são analisar documentações como contratos, eleição da diretoria, estatutos, procurações, entre outros documentos jurídicos.
Sempre que anexado um documento, é fundamental estar ciente do nome da empresa e CNPJ, envolvidos e CPF/RG e datas importantes do contrato.
Analise com atenção todas as cláusulas. 
Atente-se em termos como "aval e avalizar", "fiança(s)", "caução", "endosso(s)", "ato estranho(s)", "prestação de garantia(s) em favor de terceiros", "contrato bancário", "contratação", "obrigação(ões)", "alienação", "oneração", "garantia(s)", "ativos" ou similares presentes nas cláusulas.

Em seguida, responda perguntas do usuário utilizando apenas o contexto fornecido.

Contexto:
{context}

Conversa atual:
{chat_history}

Human: {question}
AI: 
"""

def get_config(config_name):
    if config_name.lower() in st.session_state:
        return st.session_state[config_name.lower()]
    elif config_name.lower() == "model_name":
        return MODEL_NAME
    elif config_name.lower() == "retrieval_search_type":
        return RETRIEVAL_SEARCH_TYPE
    elif config_name.lower() == "retrieval_kwargs":
        return RETRIEVAL_KWARGS
    elif config_name.lower() == "prompt":
        return PROMPT
