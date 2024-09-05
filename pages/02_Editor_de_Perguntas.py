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


def formatar_apelidos(texto):
    apelidos_formatados = [
        sentence.strip() for sentence in texto.split("\n\n") if sentence.strip()
    ]
    return apelidos_formatados  # Retorna uma lista de apelidos


username = st.session_state["logged_user"]
username = username.lower()

if username in ["alesatu", "riccord", "palomar", "mdtorre", "deninas", "igorsan"]:

    tab1, tab2 = st.tabs(["Editor de perguntas", "Editor de apelidos"])
    # ---------------------------------------------------------------------------------------- PERGUNTAS
    with tab1:
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
            key="rb_perguntas",
        )

        file_path = PASTA_RAIZ / "perguntas_sidebar.json"
        with open(file_path, "r", encoding="utf8") as f:
            json_data = json.load(f)

        st.write("")
        st.write("**Editar Perguntas Padrão:**")
        formatted_output_per_padrao = ""
        for pergunta, resposta in json_data[tipo_documento]["perguntas_padrao"].items():
            formatted_output_per_padrao += f"{resposta}\n\n"

        result_box_padrao = st.text_area(
            "",
            value=formatted_output_per_padrao,
            disabled=False,
            height=350,
            key="padrao",
        )
        st.write("")
        st.write("")

        st.write("**Editar Perguntas Condições Especiais:**")
        formatted_output_per_condiespec = ""
        for pergunta, resposta in json_data[tipo_documento][
            "condicoes_especiais"
        ].items():
            formatted_output_per_condiespec += f"{resposta}\n\n"

        result_box_condiespec = st.text_area(
            "",
            value=formatted_output_per_condiespec,
            disabled=False,
            height=350,
            key="condiespec",
        )
        st.write("")

        if st.button("Sobrescrever Prompts"):
            # Alterar perguntas padrão
            perguntas_formatadas_padrao = formatar_perguntas(result_box_padrao)
            json_data[tipo_documento]["perguntas_padrao"] = perguntas_formatadas_padrao
            # Alterar perguntas condições especiais
            perguntas_formatadas_condiespec = formatar_perguntas(result_box_condiespec)
            json_data[tipo_documento][
                "condicoes_especiais"
            ] = perguntas_formatadas_condiespec
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file_path = str(file_path)[:-5] + f"_backup_{timestamp}.json"
            # Create a backup of the old file
            shutil.copy(file_path, backup_file_path)
            # Write JSON data to a file
            with open(file_path, "w") as file:
                json.dump(json_data, file, indent=4)
            st.success("Novo prompt salvo com sucesso!")
    # ---------------------------------------------------------------------------------------- APELIDOS
    with tab2:
        st.subheader("Editor de Apelidos")

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
            key="rb_apelidos",
        )

        file_path = PASTA_RAIZ / "json_keys_dict.json"
        with open(file_path, "r", encoding="utf8") as f:
            json_data = json.load(f)

        formatted_output_apelido = ""
        for apelido in json_data[tipo_documento]:
            formatted_output_apelido += f"{apelido}\n\n"

        result_box = st.text_area(
            "",
            value=formatted_output_apelido,
            disabled=False,
            height=350,
            key="ta_apelidos",
        )
        st.write("")

        if st.button("Sobrescrever Apelidos"):
            apelidos_formatados = formatar_apelidos(result_box)
            json_data[tipo_documento] = apelidos_formatados
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file_path = str(file_path)[:-5] + f"_backup_{timestamp}.json"
            # Create a backup of the old file
            shutil.copy(file_path, backup_file_path)
            # Write JSON data to a file
            with open(file_path, "w") as file:
                json.dump(json_data, file, indent=4)
            st.success("Novos apelidos salvos com sucesso!")


else:
    st.error("Edição de perguntas não permitida para este usuário")
