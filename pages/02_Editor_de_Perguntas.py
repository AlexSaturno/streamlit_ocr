################################################################################################################################
# Bibliotecas
################################################################################################################################
import os
import json
import streamlit as st
import string
import random
from utils import PASTA_RAIZ

st.set_page_config(
    page_title="FPO - Processamento de documentos",
    page_icon=":black_medium_square:",
    layout="wide",
    initial_sidebar_state="collapsed",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)

def gerar_chave_aleatoria(tamanho=5):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(tamanho))

def formatar_perguntas(texto):
    perguntas = texto.split("\n")
    perguntas_formatadas = {}

    for pergunta in perguntas:
        chave = gerar_chave_aleatoria()
        perguntas_formatadas[chave] = pergunta.strip()
    return perguntas_formatadas

username = "palomar" #os.getenv("HADOOP_USER_NAME")

if username in ["max.saito", "palomar", "mdtorre"]:

    # Cabeçalho
    st.subheader("Editor de Perguntas")

    tipo_documento = st.radio(
        "Selecione o tipo de documento:",
        ("Contrato Social", "Procuração", "Estatuto Social", "Eleição de Diretoria"),
        horizontal=True,
    )
    
    file_path = PASTA_RAIZ / "perguntas_sidebar.json"
    with open(file_path, "r", encoding="utf8") as f:
        json_data = json.load(f)

    formatted_output_per = ""
    for pergunta, resposta in json_data[tipo_documento].items():
        formatted_output_per += f"{resposta}\n\n"

    result_box = st.text_area(
        "", value=formatted_output_per, disabled=False, height=350
    )

    perguntas_formatadas = formatar_perguntas(result_box)

    st.write("")

    if st.button("Enviar perguntas"):
        st.session_state["perguntas_editadas"] = perguntas_formatadas
        st.success("Perguntas editadas com sucesso!")


else:
    st.error("Edição de perguntas não permitida para este usuário")
