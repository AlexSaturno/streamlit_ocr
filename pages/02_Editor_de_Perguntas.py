################################################################################################################################
# Bibliotecas
################################################################################################################################
import os
import json
import streamlit as st
import string
import random
from utils import PASTA_RAIZ
import shutil
from datetime import datetime


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

# Leitura do arquivo css de estilização
with open("./styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def gerar_chave_aleatoria(tamanho=5):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(tamanho))


def formatar_perguntas(texto):
    sentences = [
        sentence.strip() for sentence in texto.split("\n\n") if sentence.strip()
    ]
    perguntas_formatadas = {i + 1: sentences[i] for i in range(len(sentences))}
    return perguntas_formatadas


username = "palomar"

if username in ["max.saito", "palomar", "mdtorre"]:

    # Cabeçalho
    st.subheader("Editor de Perguntas")

    tipo_documento = st.radio(
        "Tipo de documento:",
        (
            "Contrato Social",
            "Procuração PJ",
            "Estatuto Social",
            "Eleição de Diretoria",
            "Procuração PF",
        ),
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
    st.write("")

    if st.button("Sobrescrever Prompt"):
        perguntas_formatadas = formatar_perguntas(result_box)
        json_data[tipo_documento] = perguntas_formatadas
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file_path = str(file_path)[:-5] + f"_backup_{timestamp}.json"
        # Create a backup of the old file
        shutil.copy(file_path, backup_file_path)
        # Write JSON data to a file
        with open(file_path, "w") as file:
            json.dump(json_data, file, indent=4)
        st.success("Novo prompt salvo com sucesso!")

else:
    st.error("Edição de perguntas não permitida para este usuário")
