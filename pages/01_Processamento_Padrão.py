# PASTAS NECESSÁRIAS:
# avaliacao
# uploaded_files
# vectordb
################################################################################################################################
# Bibliotecas
################################################################################################################################
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from langchain_openai import AzureOpenAIEmbeddings

from datetime import datetime, timedelta
import os
import time
import json
import tiktoken
import streamlit as st
from PyPDF2 import PdfReader
import numpy as np
import pandas as pd
from unidecode import unidecode
import unicodedata
from time import sleep

import io
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph

from utils import *

################################################################################################################################
# Ambiente
################################################################################################################################

# Parametros das APIS
# arquivo de secrets

## Melhor modelo de embedding da OPENAI, veja documentação comparando métrica
# contra o ADA
OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-3-large"
OPENAI_ADA_EMBEDDING_MODEL_NAME = "text-embedding-3-large"
OPENAI_ADA_DEPLOYMENT_ENDPOINT = (st.secrets["AZURE_OPENAI_ENDPOINT"],)
OPENAI_ADA_API_KEY = (st.secrets["AZURE_OPENAI_API_KEY"],)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
    model=OPENAI_ADA_EMBEDDING_MODEL_NAME,
    openai_api_type="azure",
    chunk_size=1,
    api_version=st.secrets["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    api_key=st.secrets["AZURE_OPENAI_API_KEY"],
)

embeddings_ocr = AzureOpenAIEmbeddings(
    api_version=st.secrets["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    api_key=st.secrets["AZURE_OPENAI_API_KEY"],
)

llm = AzureChatOpenAI(
    azure_deployment=st.secrets["AZURE_OPENAI_DEPLOYMENT"],
    model=st.secrets["AZURE_OPENAI_MODEL"],
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    api_version=st.secrets["AZURE_OPENAI_API_VERSION"],
    api_key=st.secrets["AZURE_OPENAI_API_KEY"],
    openai_api_type="azure",
)

###############################################################################################################
####################### Parametros de modelagem ###############################################################
k_similarity = 10  # lang_chain similarity search

# Tente utilizar tamanhos de chunk_sizes = [128, 256, 512, 1024, 2048]
# https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5
pdf_chunk = 1024

# https://learn.microsoft.com/en-us/answers/questions/1551865/how-do-you-set-document-chunk-length-and-overlap-w
# Recomendado 10%
pdf_overlap = 110
##############################################################################################################


# Função OCR
def ocr(pdf_file, only_text=False):
    for file_name_relative in PASTA_ARQUIVOS.glob(
        "*.pdf"
    ):  ## first get full file name with directores using for loop
        file_name_absolute = os.path.basename(
            file_name_relative
        )  ## Now get the file name with os.path.basename
    extracted_text = ""
    contador_de_figuras = 0
    imgs_path = convert_pdf_to_images(f"{PASTA_ARQUIVOS}/{file_name_absolute}")
    file_pages = os.listdir(imgs_path)
    # print(len(file_pages))
    for file_page in file_pages:
        image_path = os.path.join(imgs_path, file_page)
        imagem = cv2.imread(image_path)
        if imagem is not None:
            if only_text is True:
                page_text = detect_text(imagem)
            else:
                page_text, contador_de_figuras = detect_figures(
                    imagem, contador_de_figuras
                )
            extracted_text += page_text + "\n"
        else:
            print(f"Erro ao carregar a imagem {image_path}")

    return extracted_text


# Funcoes auxiliares
def normalize_filename(filename):
    # Mapeamento de caracteres acentuados para não acentuados
    substitutions = {
        "á": "a",
        "à": "a",
        "ã": "a",
        "â": "a",
        "ä": "a",
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "í": "i",
        "ì": "i",
        "î": "i",
        "ï": "i",
        "ó": "o",
        "ò": "o",
        "õ": "o",
        "ô": "o",
        "ö": "o",
        "ú": "u",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "ç": "c",
        "Á": "A",
        "À": "A",
        "Ã": "A",
        "Â": "A",
        "Ä": "A",
        "É": "E",
        "È": "E",
        "Ê": "E",
        "Ë": "E",
        "Í": "I",
        "Ì": "I",
        "Î": "I",
        "Ï": "I",
        "Ó": "O",
        "Ò": "O",
        "Õ": "O",
        "Ô": "O",
        "Ö": "O",
        "Ú": "U",
        "Ù": "U",
        "Û": "U",
        "Ü": "U",
        "Ç": "C",
    }

    # Substitui caracteres especiais conforme o dicionário
    normalized_filename = "".join(substitutions.get(c, c) for c in filename)

    # Remove caracteres não-ASCII
    ascii_filename = normalized_filename.encode("ASCII", "ignore").decode("ASCII")

    # Substitui espaços por underscores
    safe_filename = ascii_filename.replace(" ", "_")

    return safe_filename


def clear_respostas():
    st.session_state["clear_respostas"] = True
    st.session_state["Q&A_done"] = False
    st.session_state["Q&A"] = {}
    st.session_state["Q&A_downloadable"] = {}
    st.session_state["data_processamento"] = None
    st.session_state["hora_processamento"] = None
    st.session_state["tempo_ia"] = None
    st.session_state["answer_downloads"] = False


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def zera_vetorizacao():
    st.session_state["vectordb_object"] = None
    st.session_state["status_vetorizacao"] = False
    st.session_state["clear_respostas"] = True
    st.session_state["Q&A_done"] = False
    st.session_state["Q&A"] = {}
    st.session_state["Q&A_downloadable"] = {}
    st.session_state["data_processamento"] = None
    st.session_state["hora_processamento"] = None
    st.session_state["tempo_ia"] = None
    st.session_state["answer_downloads"] = False


def write_stream(stream):
    result = ""
    container = st.empty()
    for chunk in stream:
        result += chunk
        container.markdown(
            f'<p class="font-stream">{result}</p>', unsafe_allow_html=True
        )


def get_stream(texto):
    for word in texto.split(" "):
        yield word + " "
        time.sleep(0.01)


# Function to initialize session state
def initialize_session_state():
    if "my_dict" not in st.session_state:
        st.session_state.my_dict = []  # Initialize as an empty list


################################################################################################################################
# UX
################################################################################################################################

# # Inicio da aplicação
initialize_session_state()

st.set_page_config(
    page_title="FPO - Processamento de documentos",
    page_icon=":black_medium_square:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Leitura do arquivo css de estilização
with open("./styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


################################################################################################################################
# UI
################################################################################################################################


# # Inicio da aplicação
def main():

    if "selecionadas" not in st.session_state:
        st.session_state["selecionadas"] = None

    if "respostas_download_txt" not in st.session_state:
        st.session_state["respostas_download_txt"] = None

    if "respostas_download_pdf" not in st.session_state:
        st.session_state["respostas_download_pdf"] = None

    if "file_name" not in st.session_state:
        st.session_state["file_name"] = None

    if "vectordb_object" not in st.session_state:
        st.session_state["vectordb_object"] = None

    if "status_vetorizacao" not in st.session_state:
        st.session_state["status_vetorizacao"] = False

    if "tipo_documento" not in st.session_state:
        st.session_state["tipo_documento"] = None

    if "Q&A" not in st.session_state:
        st.session_state["Q&A"] = {}

    if "Q&A_done" not in st.session_state:
        st.session_state["Q&A_done"] = False

    if "clear_respostas" not in st.session_state:
        st.session_state["clear_respostas"] = False

    if "data_processamento" not in st.session_state:
        st.session_state["data_processamento"] = None

    if "hora_processamento" not in st.session_state:
        st.session_state["hora_processamento"] = None

    if "pdf_IMG" not in st.session_state:
        st.session_state["pdf_IMG"] = None

    if "versao_prompt" not in st.session_state:
        st.session_state["versao_prompt"] = "v1"

    if "tempo_ia" not in st.session_state:
        st.session_state["tempo_ia"] = 0

    if "tempo_vetorizacao" not in st.session_state:
        st.session_state["tempo_vetorizacao"] = 0

    if "tempo_Q&A" not in st.session_state:
        st.session_state["tempo_Q&A"] = 0

    if "tempo_manual" not in st.session_state:
        st.session_state["tempo_manual"] = 0

    if "tokens_doc_embedding" not in st.session_state:
        st.session_state["tokens_doc_embedding"] = 0

    if "disable_downloads" not in st.session_state:
        st.session_state["disable_downloads"] = True

    if "pdf_store" not in st.session_state:
        st.session_state["pdf_store"] = True

    if "id_unico" not in st.session_state:
        st.session_state["id_unico"] = True

    def clear_respostas():
        st.session_state["clear_respostas"] = True
        st.session_state["Q&A_done"] = False
        st.session_state["Q&A"] = {}
        st.session_state["data_processamento"] = None
        st.session_state["hora_processamento"] = None
        st.session_state["tempo_ia"] = None

    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def zera_vetorizacao():
        st.session_state["vectordb_object"] = None
        st.session_state["status_vetorizacao"] = False
        st.session_state["clear_respostas"] = True
        st.session_state["Q&A_done"] = False
        st.session_state["Q&A"] = {}
        st.session_state["data_processamento"] = None
        st.session_state["hora_processamento"] = None
        st.session_state["tempo_ia"] = None

    def write_stream(stream):
        result = ""
        container = st.empty()
        for chunk in stream:
            result += chunk
            container.markdown(
                f'<p class="font-stream">{result}</p>', unsafe_allow_html=True
            )

    def get_stream(texto):
        for word in texto.split(" "):
            yield word + " "
            time.sleep(0.01)

    username = "max.saito"
    id = np.random.rand()
    if "." not in username:
        username = "User_NA" + id
    session_name = username

    st.subheader("Firmas e Poderes - Análise Automática de Documentos")
    st.write("")

    tab1, tab2 = st.tabs(["Perguntas padrão", "Perguntas adicionais"])
    # ----------------------------------------------------------------------------------------------
    with tab1:
        with st.container(border=True):
            st.session_state["tipo_documento"] = st.radio(
                "",
                [
                    "Contrato Social",
                    "Procuração PJ",
                    "Estatuto Social",
                    "Eleição de Diretoria",
                    "Procuração PF",
                ],
                horizontal=True,
                index=None,
                on_change=clear_respostas,
                label_visibility="collapsed",
            )
            st.write("")
            if st.session_state["tipo_documento"] is not None:
                pdf_file = st.file_uploader(
                    "Carregamento de arquivo",
                    type=["pdf"],
                    key="pdf_file",
                    on_change=zera_vetorizacao,
                )

                if pdf_file is not None and not st.session_state["status_vetorizacao"]:
                    # Se tiver PDFs na pasta quando inicializar a aplicação, apagá-los
                    for arquivo in PASTA_ARQUIVOS.glob("*.pdf"):
                        arquivo.unlink()
                    savefile_name = normalize_filename(pdf_file.name)
                    with open(PASTA_ARQUIVOS / f"{savefile_name}", "wb") as f:
                        f.write(pdf_file.read())

                    st.session_state["pdf_store"] = pdf_file.getbuffer()
                    st.session_state["file_name"] = pdf_file.name[:-4]

                    data_processamento = datetime.now().strftime("%Y-%m-%d")
                    hora_processamento = (datetime.now() - timedelta(hours=3)).strftime(
                        "%H:%M"
                    )
                    st.session_state["data_processamento"] = data_processamento
                    st.session_state["hora_processamento"] = hora_processamento
                    tipo = unidecode(
                        str(st.session_state["tipo_documento"]).replace(" ", "_")
                    )
                    file_name = st.session_state["file_name"]

                    id_unico = (
                        str(st.session_state["data_processamento"])
                        + "_"
                        + str(st.session_state["hora_processamento"]).replace(":", "-")
                        + "_"
                        + unidecode(str(st.session_state["file_name"]).lower())
                    )
                    st.session_state["id_unico"] = id_unico

                    pdf_store_full_path = f"{str(PASTA_ARQUIVOS)}/{id_unico}" + ".pdf"
                    pdf_store_full_path = str(PASTA_ARQUIVOS) + "/" + id_unico + ".pdf"

                    with open(pdf_store_full_path, "wb") as file:
                        file.write(st.session_state["pdf_store"])

                    vectordb_store_folder = str(PASTA_VECTORDB) + id_unico
                    if not os.path.exists(vectordb_store_folder):
                        os.makedirs(vectordb_store_folder)

                    if not st.session_state["status_vetorizacao"]:
                        # vetoriza = st.button('Vetorizar')

                        # if vetoriza:
                        st.session_state["tempo_ia"] = 0
                        start_time = time.time()
                        with st.spinner("Vetorizando documento..."):
                            # Extração de imagens do PDF
                            ocr_images_text = ""
                            ocr_images_text = ocr(pdf_file)

                            pdf_reader = PdfReader(pdf_file)

                            text = ""
                            for page in pdf_reader.pages:
                                text += page.extract_text()

                            # Extração de texto com OCR
                            ocr_text = ""
                            if (text == "") or (len(text) < 1000):
                                text = ocr(pdf_file, True)
                                ocr_text = text

                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=pdf_chunk,
                                chunk_overlap=pdf_overlap,
                                length_function=len,
                            )

                            full_text = ocr_images_text + text + ocr_text
                            chunks = text_splitter.split_text(text=full_text)

                            try:
                                VectorStore = FAISS.from_texts(
                                    chunks, embedding=embeddings
                                )
                                VectorStore.save_local(vectordb_store_folder)
                                tokens_doc_embedding = num_tokens_from_string(
                                    " ".join(chunks), "cl100k_base"
                                )
                                st.session_state["tokens_doc_embedding"] = (
                                    tokens_doc_embedding
                                )
                                st.session_state["vectordb"] = VectorStore
                                st.session_state["status_vetorizacao"] = True
                                end_time = time.time()
                                tempo_vetorizacao = end_time - start_time
                                st.session_state["tempo_vetorizacao"] = (
                                    tempo_vetorizacao
                                )
                                st.session_state["tempo_ia"] = 0

                            except Exception as e:
                                st.warning(
                                    "Arquivo contém imagem !! Deve ser processado com OCR! Favor subir outro arquivo!"
                                )

        st.write("")
        if st.session_state["status_vetorizacao"]:
            if st.session_state["tipo_documento"] is not None:
                with open(
                    str(PASTA_RAIZ) + "/perguntas_sidebar.json", "r", encoding="utf8"
                ) as f:
                    perguntas = json.load(f)
                perguntas_selecionadas = list(
                    perguntas[st.session_state["tipo_documento"]].values()
                )

                st.session_state["selecionadas"] = perguntas_selecionadas

                with open(str(PASTA_RAIZ) + "/json_keys_dict.json", "r") as f:
                    json_keys_dict = json.load(f)
                json_keys = json_keys_dict[st.session_state["tipo_documento"]]

                llm_call = st.button(
                    f"Processar perguntas padrão para {st.session_state.tipo_documento}"
                )
                st.write("")
                ph = st.empty()
                with ph.container():
                    if llm_call:
                        if not st.session_state["clear_respostas"]:
                            start_time = time.time()
                            perguntas_json = st.session_state["selecionadas"]
                            total = len(perguntas_json)

                            st.session_state["Q&A"] = {}
                            contador = 1

                            with st.container(border=True):
                                grid = st.columns([0.6, 2.8, 4.5, 1.2, 4.5])

                                with st.container(border=True):
                                    grid[0].markdown("**#**")
                                    grid[1].markdown("**Item**")
                                    grid[2].markdown("**Resposta IA**")
                                    grid[3].markdown("**Avaliação**")
                                    grid[4].markdown("**Saída FPO**")

                                for i, pergunta in enumerate(perguntas_json):
                                    tokens_query_embedding = num_tokens_from_string(
                                        pergunta, "cl100k_base"
                                    )

                                    additional_instructions_general = f"""Be correct and concise. All replies must be in Portuguese from Brazil. ONLY return a valid JSON object (no other text is necessary). The json keys are: {json_keys[i]}. The values must be a single string. If mulitple items are included in the answer, separate them by comma. If you don't find the answer in the given context, just reply the string 'Informação não encontrada' as the value for each key. The JSON MUST conform to the XML format, including any types and format requests e.g. requests for lists, objects and specific types. Do not wrap the json codes in JSON md markers. Do not wrap the json codes in JSON md markers. Do not wrap the json codes in JSON md markers."""
                                    query = pergunta + additional_instructions_general
                                    VectorStore = st.session_state["vectordb"]
                                    docs = VectorStore.similarity_search(
                                        query=query, k=k_similarity
                                    )

                                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                                    with get_openai_callback() as cb:
                                        response = chain.run(
                                            input_documents=docs, question=query
                                        )
                                    full_response = response
                                    response = response.replace(
                                        "```json\n", ""
                                    ).replace("\n```", "")
                                    try:
                                        response = json.loads(response)
                                    except:
                                        pass

                                    st.session_state["Q&A"].update(
                                        {
                                            str(contador): {
                                                "pergunta": pergunta,
                                                "resposta_ia": response,
                                                "resposta_ia_completa": full_response,
                                                "tokens_completion": cb.completion_tokens,
                                                "tokens_prompt": cb.prompt_tokens,
                                                "tokens_query_embedding": tokens_query_embedding,
                                                "retrieved_docs": str(docs),
                                            }
                                        }
                                    )

                                    atributos_pergunta = st.session_state["Q&A"][
                                        str(contador)
                                    ]

                                    pergunta_prompt = atributos_pergunta["pergunta"]
                                    resposta_llm = atributos_pergunta["resposta_ia"]

                                    itens_respostas = [
                                        (item, resposta)
                                        for item, resposta in resposta_llm.items()
                                    ]
                                    j = 1
                                    for item, resposta in itens_respostas:
                                        grid = st.columns([0.6, 2.8, 4.5, 1.2, 4.5])
                                        indice = str(contador) + "." + str(j)
                                        grid[0].markdown(indice)
                                        grid[1].write_stream(stream=get_stream(item))
                                        grid[2].write_stream(
                                            stream=get_stream(resposta)
                                        )
                                        j += 1
                                        avaliacao = grid[3].checkbox(
                                            "ok",
                                            key=f"check_avalia1_{indice}",
                                            disabled=True,
                                        )
                                        saida_fpo = grid[4].text_input(
                                            "",
                                            value="",
                                            key=f"text_input1_{indice}",
                                            label_visibility="collapsed",
                                            disabled=True,
                                        )

                                    if contador == total:
                                        st.session_state["Q&A_done"] = True
                                        end_time = time.time()
                                        tempo_qa = end_time - start_time
                                        st.session_state["tempo_Q&A"] = tempo_qa
                                    else:
                                        contador += 1

                ### PRINT VISUALIZAÇÃO DAS PERGUNTAS PALOMA
                itens_respostas = (
                    []
                )  # Inicializa uma lista vazia para armazenar os pares pergunta-resposta
                for i, atributos_pergunta in st.session_state["Q&A"].items():
                    pergunta_prompt = atributos_pergunta.get(
                        "pergunta", "Pergunta não encontrada"
                    )
                    resposta_llm = atributos_pergunta.get(
                        "resposta_ia_completa", "Resposta completa não encontrada"
                    )
                    itens_respostas.append((pergunta_prompt, resposta_llm))

                j = 1
                for item, resposta in itens_respostas:
                    st.write(f"Pergunta {j}: {item}")
                    st.write(f"R: {resposta}")
                    st.write("")
                    j += 1
                ########################################################

                if st.session_state["clear_respostas"]:
                    ph.empty()
                    sleep(0.01)
                    st.session_state.clear_respostas = False

                else:
                    if st.session_state["Q&A_done"]:
                        id_unico = st.session_state["id_unico"]

                        df_avaliacao = pd.DataFrame(
                            columns=[
                                "id_unico",
                                "data_processamento",
                                "hora_processamento",
                                "tipo_documento",
                                "versao_prompt",
                                "pergunta_prompt",
                                "#",
                                "item",
                                "resposta_ia",
                                "retrieved_docs",
                                "tokens_prompt",
                                "tokens_completion",
                                "tokens_doc_embedding",
                                "tokens_query_embedding",
                                "custo_prompt",
                                "custo_completion",
                                "custo_doc_embedding",
                                "custo_query_embedding",
                                "avaliacao",
                                "saida_FPO",
                            ]
                        )

                        token_cost = {
                            "tokens_prompt": 15 / 1e6,
                            "tokens_completion": 30 / 1e6,
                            "tokens_doc_embedding": 0.001 / 1e3,
                            "tokens_query_embedding": 0.001 / 1e3,
                        }

                        with ph.container():
                            with st.container(border=True):
                                grid = st.columns([0.6, 2.8, 4.5, 1.2, 4.5])
                                with st.container(border=True):
                                    grid[0].markdown("**#**")
                                    grid[1].markdown("**Item**")
                                    grid[2].markdown("**Resposta IA**")
                                    grid[3].markdown("**Avaliação**")
                                    grid[4].markdown("**Saída FPO**")

                                for i, atributos_pergunta in st.session_state[
                                    "Q&A"
                                ].items():

                                    pergunta_prompt = atributos_pergunta["pergunta"]
                                    resposta_llm = atributos_pergunta["resposta_ia"]
                                    retrieved_docs = atributos_pergunta[
                                        "retrieved_docs"
                                    ]
                                    tokens_prompt = atributos_pergunta["tokens_prompt"]
                                    tokens_completion = atributos_pergunta[
                                        "tokens_completion"
                                    ]
                                    tokens_doc_embedding = st.session_state[
                                        "tokens_doc_embedding"
                                    ]
                                    tokens_query_embedding = atributos_pergunta[
                                        "tokens_query_embedding"
                                    ]

                                    custo_prompt = (
                                        token_cost["tokens_prompt"] * tokens_prompt
                                    )
                                    custo_completion = (
                                        token_cost["tokens_completion"]
                                        * tokens_completion
                                    )
                                    custo_doc_embedding = round(
                                        token_cost["tokens_doc_embedding"]
                                        * tokens_doc_embedding,
                                        6,
                                    )
                                    custo_query_embedding = round(
                                        token_cost["tokens_query_embedding"]
                                        * tokens_query_embedding,
                                        6,
                                    )

                                    itens_respostas = [
                                        (item, resposta)
                                        for item, resposta in resposta_llm.items()
                                    ]

                                    st.session_state["tempo_ia"] = (
                                        st.session_state["tempo_vetorizacao"]
                                        + st.session_state["tempo_Q&A"]
                                    )

                                    j = 1
                                    for item, resposta in itens_respostas:
                                        grid = st.columns([0.6, 2.8, 4.5, 1.2, 4.5])
                                        indice = str(i) + "." + str(j)
                                        grid[0].markdown(indice)
                                        grid[1].markdown(item)
                                        grid[2].markdown(resposta)
                                        j += 1
                                        grid[3].checkbox(
                                            "ok", key=f"check_avalia_{indice}"
                                        )
                                        if st.session_state[f"check_avalia_{indice}"]:
                                            if len(resposta) <= 44:
                                                grid[4].text_input(
                                                    "",
                                                    value=resposta,
                                                    key=f"text_input_{indice}",
                                                    label_visibility="collapsed",
                                                    disabled=True,
                                                )
                                            else:
                                                grid[4].text_area(
                                                    "",
                                                    value=resposta,
                                                    key=f"text_input_{indice}",
                                                    label_visibility="collapsed",
                                                    height=int(len(resposta) / 2) + 10,
                                                    disabled=True,
                                                )
                                        else:
                                            grid[4].text_input(
                                                "",
                                                value="",
                                                key=f"text_input_{indice}",
                                                label_visibility="collapsed",
                                            )

                                        df_avaliacao.loc[len(df_avaliacao)] = [
                                            id_unico,
                                            st.session_state["data_processamento"],
                                            st.session_state["hora_processamento"],
                                            st.session_state["tipo_documento"],
                                            st.session_state["versao_prompt"],
                                            pergunta_prompt,
                                            indice,
                                            item,
                                            resposta,
                                            retrieved_docs,
                                            tokens_prompt,
                                            tokens_completion,
                                            tokens_doc_embedding,
                                            tokens_query_embedding,
                                            custo_prompt,
                                            custo_completion,
                                            custo_doc_embedding,
                                            custo_query_embedding,
                                            st.session_state[f"check_avalia_{indice}"],
                                            st.session_state[f"text_input_{indice}"],
                                        ]

                        df_avaliacao["flag_acerto"] = df_avaliacao["avaliacao"].apply(
                            lambda x: 1 if x else 0
                        )
                        df_avaliacao["flag_saida"] = df_avaliacao.apply(
                            lambda x: 0 if x.avaliacao or len(x.saida_FPO) > 0 else 1,
                            axis=1,
                        )
                        flag = df_avaliacao["flag_saida"].sum()
                        if flag == 0:
                            st.session_state["disable_downloads"] = False
                        else:
                            st.session_state["disable_downloads"] = True
                        df_avaliacao = df_avaliacao.reset_index().rename(
                            {"index": "nro_pergunta"}, axis=1
                        )

                        df_avaliacao["nro_pergunta"] = df_avaliacao[
                            "nro_pergunta"
                        ].apply(lambda x: "P" + str(x + 1))

                        # Formatando as respostas para saída IMG
                        formatted_output_sistema = ""
                        formatted_output_IMG = "<b>SISTEMA IMG</b><br/><br/><br/>"
                        formatted_output_IMG += f"<b>Nome do Arquivo:</b> {st.session_state['file_name']}.pdf<br/>"
                        formatted_output_IMG += f"<b>Tipo de Documento:</b> {st.session_state['tipo_documento']} <br/>"
                        formatted_output_IMG += f"<b>Data de Processamento:</b> {st.session_state['data_processamento']} <br/>"
                        formatted_output_IMG += f"<b>Hora de Processamento:</b> {st.session_state['hora_processamento']}<br/><br/><br/>"
                        formatted_output_IMG += (
                            f"<b>PERGUNTAS E RESPOSTAS</b><br/><br/>"
                        )

                        formatted_output_FPO = df_avaliacao[["item", "saida_FPO"]].T
                        formatted_output_FPO = formatted_output_FPO.to_csv(
                            sep=";", index=False, header=False, lineterminator="\n"
                        )

                        for k in range(len(df_avaliacao)):
                            nro = df_avaliacao.iloc[k]["#"]
                            item = df_avaliacao.iloc[k]["item"]
                            saida_fpo = df_avaliacao.iloc[k]["saida_FPO"]
                            formatted_output_IMG += (
                                f"<b>{item}</b><br/>{saida_fpo}<br/><br/>"
                            )

                        st.session_state["respostas_download_txt"] = (
                            formatted_output_FPO
                        )
                        st.session_state["respostas_download_pdf"] = (
                            formatted_output_IMG
                        )

                        buf = io.StringIO()
                        buf.write(formatted_output_FPO)
                        buf.seek(0)

                        def export_result():
                            buf.seek(0)

                        full_path = os.path.join(PASTA_RESPOSTAS, id_unico)

                        col1, col2, col3 = st.columns([4, 1, 1])
                        with col2:
                            st.write("")
                            txt_file_download_name = id_unico + ".txt"
                            if st.download_button(
                                "Download FPO",
                                buf.getvalue().encode("utf-8"),
                                txt_file_download_name,
                                "text/plain",
                                on_click=export_result,
                                disabled=st.session_state["disable_downloads"],
                            ):

                                with open(
                                    full_path + ".json", "w", encoding="utf-8"
                                ) as f:
                                    json.dump(st.session_state["Q&A"], f, indent=4)
                                with open(full_path + ".txt", "w") as file:
                                    file.write(formatted_output_FPO)

                        with col3:
                            st.write("")
                            # Output PDF file
                            pdf_file_store_name = full_path + ".pdf"
                            st.session_state["pdf_IMG"] = pdf_file_store_name

                            # Create a PDF document
                            pdf_document = SimpleDocTemplate(pdf_file_store_name)
                            pdf_elements = []

                            # Create a stylesheet for styling
                            styles = getSampleStyleSheet()

                            # Parse the HTML-like text into a Paragraph
                            paragraph = Paragraph(
                                formatted_output_IMG, styles["Normal"]
                            )

                            # Add the Paragraph to the PDF elements
                            pdf_elements.append(paragraph)

                            # Build the PDF document
                            pdf_document.build(pdf_elements)

                            pdf_file_download_name = id_unico + ".pdf"
                            with open(pdf_file_store_name, "rb") as f:
                                st.download_button(
                                    "Download IMG",
                                    f,
                                    pdf_file_download_name,
                                    disabled=st.session_state["disable_downloads"],
                                )

                        st.write("")
                        st.write("")
                        st.write("")
                        if not st.session_state["disable_downloads"]:
                            with st.container(border=True):

                                colA, colB, colC = st.columns(
                                    [2, 3, 1.5],
                                    gap="large",
                                )
                                colA.radio(
                                    "Complexidade do Documento",
                                    ["Baixa", "Média", "Alta"],
                                    key="complexidade",
                                    horizontal=True,
                                    index=None,
                                )
                                st.write("")
                                colB.selectbox(
                                    "Tempo estimado para processamento manual",
                                    (5, 10, 15, 20, 25, 30),
                                    index=None,
                                    placeholder="Selecione...",
                                    key="tempo_manual",
                                )

                                # ----- df_resumo
                                doc_tokens_prompt_count = df_avaliacao[
                                    "tokens_prompt"
                                ].sum()
                                doc_tokens_completion_count = df_avaliacao[
                                    "tokens_completion"
                                ].sum()
                                doc_tokens_doc_embedding_count = df_avaliacao[
                                    "tokens_doc_embedding"
                                ].values[0]
                                doc_tokens_query_embedding_count = df_avaliacao[
                                    "tokens_query_embedding"
                                ].sum()
                                doc_tokens_prompt_cost = df_avaliacao[
                                    "custo_prompt"
                                ].sum()
                                doc_tokens_completion_cost = df_avaliacao[
                                    "custo_completion"
                                ].sum()
                                doc_tokens_doc_embedding_cost = df_avaliacao[
                                    "custo_doc_embedding"
                                ].values[0]
                                doc_tokens_query_embedding_cost = df_avaliacao[
                                    "custo_query_embedding"
                                ].sum()

                                custo_ia = round(
                                    doc_tokens_prompt_cost
                                    + doc_tokens_completion_cost
                                    + doc_tokens_doc_embedding_cost
                                    + doc_tokens_query_embedding_cost,
                                    3,
                                )

                                total_perguntas = len(df_avaliacao)
                                acertos = df_avaliacao.flag_acerto.sum()
                                precisao = round(acertos / total_perguntas, 3)

                                df_aux = (
                                    df_avaliacao[["nro_pergunta", "flag_acerto"]]
                                    .set_index("nro_pergunta")
                                    .T
                                )
                                df_aux["index"] = 0
                                df_aux = df_aux.set_index("index")

                                if st.session_state["complexidade"] is not None:
                                    complexidade = st.session_state["complexidade"]
                                else:
                                    complexidade = ""

                                if st.session_state["tempo_manual"] is not None:
                                    tempo_manual = st.session_state["tempo_manual"]
                                else:
                                    tempo_manual = ""

                                if st.session_state["tempo_ia"] is not None:
                                    tempo_ia = st.session_state["tempo_ia"]
                                    tempo_ia = round(tempo_ia / 60, 3)
                                else:
                                    tempo_ia = ""

                                try:
                                    eficiencia = round(
                                        (tempo_ia - tempo_manual) / tempo_manual, 3
                                    )
                                except:
                                    eficiencia = f"Manual:{tempo_manual}, IA:{tempo_ia}"

                                df_resumo = pd.DataFrame(
                                    {
                                        "Usuario": username,
                                        "Id_Unico": id_unico,
                                        "Data_Processamento": [
                                            st.session_state["data_processamento"]
                                        ],
                                        "Hora_Processamento": [
                                            st.session_state["hora_processamento"]
                                        ],
                                        "Tipo_Documento": [
                                            st.session_state["tipo_documento"]
                                        ],
                                        "Versao_Prompt": [
                                            st.session_state["versao_prompt"]
                                        ],
                                        "Nome_Arquivo": [st.session_state["file_name"]],
                                        "Complexidade": [complexidade],
                                        "Tempo_Manual (min)": [tempo_manual],
                                        "Tempo_IA (min)": [tempo_ia],
                                        "Eficiencia_Tempo": [eficiencia],
                                        "Custo_IA": [custo_ia],
                                        "Qtd_Total_Perguntas": [total_perguntas],
                                        "Qtd_Total_Acertos": [acertos],
                                        "Precisao": [precisao],
                                    }
                                )

                                df_resumo = pd.concat([df_resumo, df_aux], axis=1)

                                l1 = [
                                    "Usuario",
                                    "Id_Unico",
                                    "Data_Processamento",
                                    "Hora_Processamento",
                                    "Tipo_Documento",
                                    "Versao_Prompt",
                                    "Nome_Arquivo",
                                    "Complexidade",
                                    "Tempo_Manual (min)",
                                    "Tempo_IA (min)",
                                    "Eficiencia_Tempo",
                                    "Custo_IA",
                                    "Qtd_Total_Perguntas",
                                    "Qtd_Total_Acertos",
                                    "Precisao",
                                ]

                                l2 = list(df_resumo)
                                cols_p = set(l2) - set(l1)

                                for col in cols_p:
                                    df_resumo[col] = df_resumo[col].astype(int)
                                # ----- df_resumo - FIM

                                with colC:
                                    st.write("")
                                    botao_avalia = st.button(
                                        "Salvar Avaliação",
                                        key="botao_avalia",
                                        disabled=st.session_state["disable_downloads"],
                                    )

                                if botao_avalia:
                                    df_avaliacao = df_avaliacao.drop(
                                        ["flag_saida"], axis=1
                                    )
                                    try:
                                        df_avaliacao_current = pd.read_csv(
                                            str(PASTA_RAIZ)
                                            + "/avaliacao/df_avaliacao_teste.csv"
                                        )
                                        df_avaliacao = pd.concat(
                                            [df_avaliacao_current, df_avaliacao]
                                        ).reset_index(drop=True)
                                    except:
                                        pass
                                    df_avaliacao.to_csv(
                                        str(PASTA_RAIZ)
                                        + "/avaliacao/df_avaliacao_teste.csv",
                                        index=False,
                                    )

                                    try:
                                        df_resumo_current = pd.read_csv(
                                            str(PASTA_RAIZ)
                                            + "/avaliacao/df_resumo_teste.csv"
                                        )
                                        df_resumo = pd.concat(
                                            [df_resumo_current, df_resumo]
                                        ).reset_index(drop=True)
                                    except:
                                        pass
                                    df_resumo.to_csv(
                                        str(PASTA_RAIZ)
                                        + "/avaliacao/df_resumo_teste.csv",
                                        index=False,
                                    )

                                    st.success("Avaliação enviada com sucesso!")

            # ----------------------------------------------------------------------------------------------
        with tab2:

            def clear_text():
                st.session_state.query_add = st.session_state.widget
                st.session_state.widget = ""

            st.write("")
            if st.session_state["status_vetorizacao"]:
                st.text_input("**Digite aqui a sua pergunta**", key="widget")
                query_add = st.session_state.get("query_add", "")
                with st.form(key="myform1"):
                    submit_button = st.form_submit_button(
                        label="Enviar", on_click=clear_text
                    )

                    if submit_button:
                        VectorStore = st.session_state["vectordb"]
                        docs = VectorStore.similarity_search(
                            query=query_add, k=k_similarity
                        )
                        st.session_state["doc_retrieval"] = docs
                        chain = load_qa_chain(llm=llm, chain_type="stuff")

                        with get_openai_callback() as cb:
                            response = chain.run(
                                input_documents=docs,
                                question=query_add
                                + ".\n\nAnswer in a very objective way. Explanations are not required.\nFor multiple information return 1 item per line, adding line breaks in between preceeded by 2 blank spaces.\nThe reply must be always in Portuguese from Brazil.",
                            )
                        with st.empty():
                            st.markdown(f"**{query_add}**" + "  \n " + response)

            else:
                st.write("Documento não vetorizado!")


if __name__ == "__main__":
    main()
