### ====================================
### Importações e Globais
### ====================================
from pathlib import Path
import os
from pdf2image import convert_from_path
import cv2
import pytesseract
import shutil
import numpy as np
import re
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import streamlit as st
from configs import *

PASTA_ARQUIVOS = Path(__file__).parent / "arquivos"
PASTA_IMAGENS_DEBUG = Path(__file__).parent / "imagens_debug"
PASTA_IMAGENS = Path(__file__).parent / "files_images"
if not os.path.exists(PASTA_ARQUIVOS):
    os.makedirs(PASTA_ARQUIVOS)


### ====================================
### Funções
### ====================================
def convert_pdf_to_images(pdf_path):
    img_path = os.path.join(
        "files_images", os.path.basename(pdf_path).strip(".pdf") + "_images"
    )
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    images = convert_from_path(pdf_path=pdf_path)
    for i in range(len(images)):
        images[i].save(os.path.join(img_path, "page" + str(i) + ".jpg"), "JPEG")
    return img_path


def preprocess_for_ocr(imagem):
    # Converter para escala de cinza
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Binarização com Thresholding
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return binary


# Função de pré-processamento para melhorar o OCR das figuras
def preprocess_for_ocr_figuras(imagem):
    # Converter para escala de cinza
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # # Remoção de ruído com suavização
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # # Ajuste de contraste e brilho
    alpha = 1.3  # Contraste
    beta = 0  # Brilho
    adjusted = cv2.convertScaleAbs(blur, alpha=alpha, beta=beta)

    # Binarização com Thresholding
    _, binary = cv2.threshold(adjusted, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Operações morfológicas para melhorar o texto
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    return eroded


# Extraír regiões no arquivo
def extract_regions(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    return bounding_boxes


# Detecção de contornos para as figuras
def combine_close_contours(bounding_boxes, max_distance=50):
    combined_boxes = []
    while bounding_boxes:
        box = bounding_boxes.pop(0)
        x, y, w, h = box
        for other_box in bounding_boxes[:]:
            ox, oy, ow, oh = other_box
            if abs(ox - x) < max_distance and abs(oy - y) < max_distance:
                x = min(x, ox)
                y = min(y, oy)
                w = max(x + w, ox + ow) - x
                h = max(y + h, oy + oh) - y
                bounding_boxes.remove(other_box)
        combined_boxes.append((x, y, w, h))
    return combined_boxes


def is_figure(roi, box, min_width, min_height):
    x, y, w, h = box

    max_width = 1500
    max_height = 2000
    # Critério 1: Proporção largura/altura (página inteira)
    if w > max_width or h > max_height:  # Ajustar tamanho máximo
        return False, 1

    # Critério 1: Proporção largura/altura
    if w < min_width or h < min_height:  # Ajustar tamanho mínimo
        return False, 1

    # Critério 2: Densidade de contornos
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_area = sum(cv2.contourArea(c) for c in contours)
    roi_area = w * h
    density = contour_area / roi_area
    if density < 0.02:  # Ajuste o valor conforme necessário
        return False, 2

    # Critério 3: Presença de linhas horizontais e verticais
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10
    )
    if lines is None or len(lines) < 5:  # Ajuste o valor conforme necessário
        return False, 3

    return True, 0


def detect_figures(imagem, contador_de_figuras):
    text_chunks = []
    bounding_boxes = []
    min_width = 100
    min_height = 100
    figures = []
    criterios = []

    # Pré-processar a imagem antes de aplicar o OCR
    imagem_processada = preprocess_for_ocr(imagem)

    text = pytesseract.image_to_string(imagem_processada).strip()
    text_chunks.append(text)

    # Identificar contornos
    bounding_boxes = extract_regions(imagem_processada)
    bounding_boxes = [
        box for box in bounding_boxes if box[2] > min_width and box[3] > min_height
    ]
    bounding_boxes = combine_close_contours(bounding_boxes)
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[1])

    for box in bounding_boxes:
        x, y, w, h = box
        roi = imagem[y : y + h, x : x + w]

        # Desenhar contorno azul na imagem original se for figura
        cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0, 255), 4)

        is_fig, criterio = is_figure(roi, box, min_width, min_height)
        if is_fig:
            figures.append(roi)
            contador_de_figuras = contador_de_figuras + 1

            if not os.path.exists(img_path):
                os.makedirs(img_path)
            cv2.imwrite(f"{PASTA_IMAGENS_DEBUG}/figure_{contador_de_figuras}.jpg", roi)
            imagem_recortada = cv2.imread(
                f"{PASTA_IMAGENS_DEBUG}/figure_{contador_de_figuras}.jpg"
            )
            imagem_processada = preprocess_for_ocr_figuras(imagem_recortada)
            text = pytesseract.image_to_string(imagem_processada).strip()
            # print(f"Texto da figura {contador_de_figuras}: {text}")
            text_chunks.append(text)

            # Desenhar contorno verde na imagem original se for figura
            cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            criterios.append(criterio)

    extracted_text = "\n".join(text_chunks)

    return extracted_text, contador_de_figuras


def ocr():
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
    for file_page in file_pages:
        image_path = os.path.join(imgs_path, file_page)
        imagem = cv2.imread(image_path)
        if imagem is not None:
            page_text, contador_de_figuras = detect_figures(imagem, contador_de_figuras)
            extracted_text += page_text + "\n"
        else:
            print(f"Erro ao carregar a imagem {image_path}")

    return extracted_text


def document_loader():
    documentos = []
    for arquivo in PASTA_ARQUIVOS.glob("*.pdf"):
        loader = PyPDFLoader(str(arquivo))
        documentos_arquivo = loader.load()
        documentos.extend(documentos_arquivo)
    return documentos


def text_splitter(documentos):
    recur_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500, chunk_overlap=250, separators=["\n\n", "\n", " ", ""]
    )

    documentos = recur_splitter.split_documents(documentos)

    for i, doc in enumerate(documentos):
        doc.metadata["source"] = doc.metadata["source"].split("/")[
            -1
        ]  # Pegar só o nome do arquivo
        doc.metadata["doc_id"] = i

    return documentos


def vector_store(documentos):
    embedding_model = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents=documentos, embedding=embedding_model)
    return vector_store


def cria_chain_conversa():
    # documentos = document_loader()
    # documentos = text_splitter(documentos)
    # vectorstore = vector_store(documentos)
    texto = ocr()
    # Text Split
    recur_split = RecursiveCharacterTextSplitter(
        chunk_size=2500, chunk_overlap=250, separators=["\n\n", "\n", ".", " ", ""]
    )
    textos = recur_split.split_text(texto)

    # Embeddings
    embeddings_model = OpenAIEmbeddings()

    # Vector Store
    vectorstore = FAISS.from_texts(texts=textos, embedding=embeddings_model)

    chat = ChatOpenAI(
        model=get_config("model_name"), api_key=st.secrets["OPENAI_API_KEY"]
    )
    memory = ConversationBufferMemory(
        return_messages=True, memory_key="chat_history", output_key="answer"
    )
    retriever = vectorstore.as_retriever(
        search_type=get_config("retrieval_search_type"),
        search_kwargs=get_config("retrieval_kwargs"),
    )
    prompt = PromptTemplate.from_template(template=get_config("prompt"))
    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

    st.session_state["chain"] = chat_chain
