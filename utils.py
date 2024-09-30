### ====================================
### Importações e Globais
### ====================================
from pathlib import Path
import os
from pdf2image import convert_from_path
from configs import *
import shutil


PASTA_RAIZ = (
    Path(__file__).parent
    # r"C:\Projetos\Asimov Academy\Projetos\POCs banco\Projetos\FPO\FPO\02_staging\UI"
)
PASTA_IMAGENS_DEBUG = Path(__file__).parent / "imagens_debug"
PASTA_IMAGENS = Path(__file__).parent / "files_images"
PASTA_VECTORDB = Path(__file__).parent / "vectordb"
PASTA_ARQUIVOS = Path(__file__).parent / "uploaded_files"
PASTA_RESPOSTAS = Path(__file__).parent / "respostas"
PASTA_AVALIACAO = Path(__file__).parent / "avaliacao"
if not os.path.exists(PASTA_IMAGENS):
    os.makedirs(PASTA_IMAGENS)
if not os.path.exists(PASTA_ARQUIVOS):
    os.makedirs(PASTA_ARQUIVOS)
if not os.path.exists(PASTA_RESPOSTAS):
    os.makedirs(PASTA_RESPOSTAS)
if not os.path.exists(PASTA_AVALIACAO):
    os.makedirs(PASTA_AVALIACAO)


### ====================================
### Funções
### ====================================
def convert_pdf_to_images(pdf_path):
    img_path = os.path.join(
        "files_images", os.path.basename(pdf_path).strip(".pdf") + "_images"
    )
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    images = convert_from_path(
        pdf_path=pdf_path,
        poppler_path=r"C:\Release-24.02.0-0\poppler-24.02.0\Library\bin",
    )
    for i in range(len(images)):
        images[i].save(os.path.join(img_path, "page" + str(i) + ".jpg"), "JPEG")
    return img_path
