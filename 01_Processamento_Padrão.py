################################################################################################################################
# Bibliotecas
################################################################################################################################
import os
import openai
from datetime import datetime, date, timedelta
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
import json
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.callbacks import get_openai_callback
from langchain_openai import AzureOpenAIEmbeddings
import tiktoken
import time
import numpy as np
import io
from io import BytesIO
import unicodedata
from unidecode import unidecode
from time import sleep
import sys
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import base64
from utils import *

################################################################################################################################
# UX
################################################################################################################################
st.set_page_config(
    page_title="FPO - Processamento de documentos",
    page_icon=":black_medium_square:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
pdf_chunk = 2048

# https://learn.microsoft.com/en-us/answers/questions/1551865/how-do-you-set-document-chunk-length-and-overlap-w
# Recomendado 10%
pdf_overlap = 205
##############################################################################################################

# Funcoes auxiliares
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


def chain_price_estimate(cb):
    encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    price = 0.0001
    normalizer = 1000
    cambio = 5
    chain_cost = (cb.total_cost) * cambio
    return chain_cost


def rag_price_estimate(doc):
    encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    price = 0.0001
    normalizer = 1000
    cambio = 5
    for i in doc:
        num_tokens = encoding.encode(i)
        total_tokens += len(num_tokens)
    # custo total em reais
    vector_cost = (total_tokens / normalizer) * price * cambio
    return vector_cost


# Function to initialize session state
def initialize_session_state():
    if "my_dict" not in st.session_state:
        st.session_state.my_dict = []  # Initialize as an empty list


# Example of appending data
def append_data(new_data):
    st.session_state.my_dict.append(new_data)


def add_message(role, message):
    st.session_state.chat_history.append({"role": role, "content": message})


def tratar_resposta(resposta, max_length=78):
    # Remove acentos e cedilha
    resposta = "".join(
        c
        for c in unicodedata.normalize("NFD", resposta)
        if not unicodedata.combining(c)
    )
    # Quebra de linhas a cada 78 caracteres
    words = resposta.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= max_length:

            current_line += word + " "
        else:

            lines.append(current_line.rstrip())
            current_line = word + " "
    lines.append(current_line.rstrip())
    return "\n".join(lines)


def quebra_texto(texto, max_length=78):
    linhas = []
    inicio = 0
    while inicio < len(texto):
        fim = min(inicio + max_length, len(texto))
        linhas.append(texto[inicio:fim])
        inicio = fim
    return "\n".join(linhas)


def zera_vetorizacao():
    st.session_state["status_vetorizacao"] = 0
    st.session_state["clear_respostas"] = True
    st.session_state["Q&A_done"] = False


# # Inicio da aplicação
initialize_session_state()


def main():
    if "doc_retrieval" not in st.session_state:
        st.session_state["doc_retrieval"] = ""

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

    if "pdf_object" not in st.session_state:
        st.session_state["pdf_object"] = None

    if "Q&A" not in st.session_state:
        st.session_state["Q&A"] = {}

    if "Q&A_done" not in st.session_state:
        st.session_state["Q&A_done"] = False

    if "clear_respostas" not in st.session_state:
        st.session_state["clear_respostas"] = False

    def clear_respostas():
        st.session_state["clear_respostas"] = True

    if "data_processamento" not in st.session_state:
        st.session_state["data_processamento"] = None

    if "hora_processamento" not in st.session_state:
        st.session_state["hora_processamento"] = None

    username = "max.saito"  # os.getenv("HADOOP_USER_NAME")
    id = np.random.rand()
    if "." not in username:
        username = "User_NA" + id
    session_name = username

    st.subheader("Firmas e Poderes - Análise Automática de Documentos")
    st.write("")

    # ----------------------------------------------------------------------------------------------
    with st.expander("Pré-processamento", expanded=True):
        ## Upload do PDF
        pdf_file = st.file_uploader(
            "Carregamento de arquivo",
            type=["pdf"],
            key="pdf_file",
            on_change=zera_vetorizacao,
        )
        if (
            pdf_file is not None
        ):  # Se tiver PDFs na pasta quando inicializar a aplicação, apagá-los
            for arquivo in PASTA_ARQUIVOS.glob("*.pdf"):
                arquivo.unlink()
            with open(PASTA_ARQUIVOS / pdf_file.name, "wb") as f:
                f.write(pdf_file.read())

        if pdf_file is not None:
            st.session_state["pdf"] = pdf_file

            if not st.session_state["status_vetorizacao"]:
                vetoriza = st.button("Vetorizar")
                extrair_texto_ocr = False
                if vetoriza:

                    with st.spinner("Vetorizando documento..."):
                        text = ""
                        text = ocr(pdf_file)
                        ocr_text = text
                        # print(f"OCR TEXT: {ocr_text}\n\n\n")
                        pdf_reader = PdfReader(pdf_file)

                        for page in pdf_reader.pages:
                            text += page.extract_text()
                        # print(f"RAG TEXT: {text.replace(ocr_text, '', 1)}\n\n\n")

                        # metodo de extrair texto do Pdf
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=pdf_chunk,
                            chunk_overlap=pdf_overlap,
                            length_function=len,
                        )

                        # Masse de texto do pdf
                        chunks = text_splitter.split_text(text=text)

                        # Escrevendo o nome do DB
                        st.session_state["file_name"] = pdf_file.name[:-4]

                        file_name = st.session_state["file_name"]
                        folder_path = PASTA_ARQUIVOS
                        full_path = folder_path / file_name

                        try:
                            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                            VectorStore.save_local(full_path)
                            st.session_state["vectordb"] = VectorStore
                            st.session_state["status_vetorizacao"] = True
                            st.rerun()
                        except Exception as e:
                            extrair_texto_ocr = True
                            st.warning("Arquivo será processado com OCR.")
                if extrair_texto_ocr is True:
                    text = ""
                    text = ocr(pdf_file, True)
                    ocr_text = text
                    # print(f"OCR SOMENTE TEXTO: {ocr_text}\n\n\n")

                    # metodo de extrair texto do Pdf
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=pdf_chunk,
                        chunk_overlap=pdf_overlap,
                        length_function=len,
                    )

                    # Masse de texto do pdf
                    chunks = text_splitter.split_text(text=text)

                    # Escrevendo o nome do DB
                    st.session_state["file_name"] = pdf_file.name[:-4]

                    file_name = st.session_state["file_name"]
                    folder_path = PASTA_ARQUIVOS
                    full_path = folder_path / file_name

                    try:
                        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                        VectorStore.save_local(full_path)
                        st.session_state["vectordb"] = VectorStore
                        st.session_state["status_vetorizacao"] = True
                        st.rerun()
                    except Exception as e:
                        st.warning("Arquivo não pode ser processado.")

    if st.session_state["status_vetorizacao"]:
        tab1, tab2 = st.tabs(["Perguntas padrão", "Perguntas adicionais"])
        with tab1:
            # ----------------------------------------------------------------------------------------------
            with st.expander("Tipo de Documento", expanded=True):

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
                )

                if st.session_state["tipo_documento"] is not None:
                    with open(
                        PASTA_RAIZ / "perguntas_sidebar.json", "r", encoding="utf8"
                    ) as f:
                        perguntas = json.load(f)
                    perguntas_selecionadas = list(
                        perguntas[st.session_state["tipo_documento"]].values()
                    )
                    st.session_state["selecionadas"] = perguntas_selecionadas

                    st.write("")
                    if st.session_state["status_vetorizacao"]:
                        llm_call = st.button(
                            f"Processar perguntas padrão para {st.session_state.tipo_documento}"
                        )
                        st.write("")
                        ph = st.empty()
                        with ph.container():
                            if llm_call:
                                if not st.session_state["clear_respostas"]:
                                    perguntas_json = st.session_state["selecionadas"]
                                    total = len(perguntas_json)

                                    st.session_state["Q&A"] = {}
                                    contador = 1

                                    for pergunta in perguntas_json:
                                        query = pergunta
                                        VectorStore = st.session_state["vectordb"]
                                        docs = VectorStore.similarity_search(
                                            query=query, k=k_similarity
                                        )
                                        st.session_state["doc_retrieval"] = (
                                            docs  # variavel estado para ser mantida na pagina
                                        )
                                        chain = load_qa_chain(
                                            llm=llm, chain_type="stuff"
                                        )
                                        with get_openai_callback() as cb:
                                            response = chain.run(
                                                input_documents=docs,
                                                question=query
                                                + ".\n\nAnswer in a very objective way. Explanations are not required.\nFor multiple information return 1 item per line, adding line breaks in between preceeded by 2 blank spaces.\nThe reply must be always in Portuguese from Brazil.",
                                            )
                                            # response = response.replace('\n', ' \n ')
                                        st.session_state["Q&A"].update(
                                            {contador: {pergunta: response}}
                                        )
                                        for pergunta, resposta in st.session_state[
                                            "Q&A"
                                        ][contador].items():
                                            st.write(
                                                f"**{contador}) {pergunta}**"
                                                + "  \n "
                                                + response
                                            )
                                        contador += 1

                                        if contador == total:
                                            st.session_state["Q&A_done"] = True

                                if st.session_state["Q&A_done"]:
                                    data_processamento = datetime.now().strftime(
                                        "%Y-%m-%d"
                                    )
                                    hora_processamento = (
                                        datetime.now() - timedelta(hours=3)
                                    ).strftime("%H:%M")
                                    st.session_state["data_processamento"] = (
                                        data_processamento
                                    )
                                    st.session_state["hora_processamento"] = (
                                        hora_processamento
                                    )

                                    tipo = unidecode(
                                        str(st.session_state["tipo_documento"]).replace(
                                            " ", "_"
                                        )
                                    )
                                    file_name = st.session_state["file_name"]
                                    # Formatando as respostas para saída no FPO
                                    formatted_output_sistema = ""
                                    formatted_output_IMG = (
                                        "<b>SISTEMA IMG</b><br/><br/><br/>"
                                    )
                                    formatted_output_IMG += (
                                        f"<b>Nome do Arquivo:</b> {file_name}.pdf<br/>"
                                    )
                                    formatted_output_IMG += (
                                        f"<b>Tipo de Documento:</b> {tipo} <br/>"
                                    )
                                    formatted_output_IMG += f"<b>Data de Processamento:</b> {data_processamento} <br/>"
                                    formatted_output_IMG += f"<b>Hora de Processamento:</b> {hora_processamento}<br/><br/><br/>"
                                    formatted_output_IMG += (
                                        f"<b>PERGUNTAS E RESPOSTAS</b><br/><br/>"
                                    )

                                    for key1, value1 in st.session_state["Q&A"].items():
                                        for pergunta, resposta in value1.items():
                                            formatted_output_sistema += (
                                                f"{pergunta}\n{resposta}\n\n"
                                            )
                                            formatted_output_IMG += f"<b>{pergunta}</b><br/>{resposta}<br/><br/>"
                                    st.session_state["respostas_download_txt"] = (
                                        formatted_output_sistema
                                    )
                                    st.session_state["respostas_download_pdf"] = (
                                        formatted_output_IMG
                                    )

                        if st.session_state["clear_respostas"]:
                            ph.empty()
                            sleep(0.01)
                            st.session_state.clear_respostas = False

                        else:
                            ph.empty()
                            sleep(0.01)
                            if st.session_state["Q&A_done"]:
                                with ph.container():
                                    for i, pair in st.session_state["Q&A"].items():
                                        for key, value in pair.items():
                                            st.markdown(
                                                f"**{i}) {key}**" + "  \n " + value
                                            )

                                formatted_output_sistema = st.session_state[
                                    "respostas_download_txt"
                                ]
                                formatted_output_IMG = st.session_state[
                                    "respostas_download_pdf"
                                ]

                                buf = io.StringIO()
                                buf.write(formatted_output_sistema)
                                buf.seek(0)

                                def export_result():
                                    buf.seek(0)

                                with st.container():
                                    # criacão da pasta do usuario
                                    pasta_respostas = os.path.join(
                                        PASTA_RESPOSTAS, username
                                    )
                                    if not os.path.exists(pasta_respostas):
                                        os.makedirs(pasta_respostas)

                                    data_processamento = str(
                                        st.session_state["data_processamento"]
                                    ).replace("-", "_")

                                    hora_processamento = str(
                                        st.session_state["hora_processamento"]
                                    ).replace(":", "_")
                                    tipo = unidecode(
                                        str(st.session_state["tipo_documento"]).replace(
                                            " ", "_"
                                        )
                                    )
                                    file_name = st.session_state["file_name"]
                                    full_path = os.path.join(
                                        pasta_respostas,
                                        data_processamento
                                        + "_"
                                        + hora_processamento
                                        + "_"
                                        + tipo
                                        + "_"
                                        + file_name,
                                    )

                                    col1, col2, col3 = st.columns([1, 1, 1])

                                    with col1:
                                        st.write("")
                                        if st.download_button(
                                            "Download txt",
                                            buf.getvalue().encode("utf-8"),
                                            str(st.session_state["file_name"])
                                            + "_respostas.txt",
                                            "text/plain",
                                            on_click=export_result,
                                        ):

                                            with open(
                                                full_path + ".json",
                                                "w",
                                                encoding="utf-8",
                                            ) as f:
                                                json.dump(
                                                    st.session_state["Q&A"], f, indent=4
                                                )
                                            with open(full_path + ".txt", "w") as file:
                                                file.write(formatted_output_sistema)

                                    with col2:
                                        st.write("")
                                        # Output PDF file
                                        pdf_file_store_name = full_path + ".pdf"

                                        # Create a PDF document
                                        pdf_document = SimpleDocTemplate(
                                            pdf_file_store_name
                                        )
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

                                        pdf_file_download_name = (
                                            data_processamento
                                            + "_"
                                            + hora_processamento
                                            + "_"
                                            + tipo
                                            + "_"
                                            + file_name
                                            + ".pdf"
                                        )
                                        with open(pdf_file_store_name, "rb") as f:
                                            st.download_button(
                                                "Download pdf",
                                                f,
                                                pdf_file_download_name,
                                            )

                    else:
                        st.write("")
                        st.write("Documento não vetorizado")

                else:
                    st.write("")
                    st.write("Selecione o tipo de documento")

            # ----------------------------------------------------------------------------------------------
        with tab2:

            def clear_text():
                st.session_state.query_add = st.session_state.widget
                st.session_state.widget = ""

            with st.expander("Perguntas adicionais", expanded=True):
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
                    st.write("Documento não vetorizado")


if __name__ == "__main__":
    main()
