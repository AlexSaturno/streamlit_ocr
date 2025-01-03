### ====================================
### Importações e Globais
### ====================================
from pathlib import Path
import os
import re
from pdf2image import convert_from_path
from configs import *


PASTA_RAIZ = Path(__file__).parent
PASTA_IMAGENS_DEBUG = Path(__file__).parent / "imagens_debug"
PASTA_IMAGENS = Path(__file__).parent / "files_images"
PASTA_VECTORDB = Path(__file__).parent / "vectordb"
PASTA_ARQUIVOS = Path(__file__).parent / "uploaded_files"
PASTA_RESPOSTAS = Path(__file__).parent / "respostas"
PASTA_AVALIACAO = Path(__file__).parent / "avaliacao"
PASTA_DOWNLOAD = r"C:\Users\Public"
CAMINHO_SUGERIDO = Path(
    # r"C:\Users\alexa\Documents\Testes do FPO"
    r"\\vega\BackOffice BS\2 - BO CADASTRO E FIRMAS\FIRMAS\BO Firmas\IA - FPO"
)

if not os.path.exists(PASTA_IMAGENS):
    os.makedirs(PASTA_IMAGENS)
if not os.path.exists(PASTA_ARQUIVOS):
    os.makedirs(PASTA_ARQUIVOS)
if not os.path.exists(PASTA_RESPOSTAS):
    os.makedirs(PASTA_RESPOSTAS)
if not os.path.exists(PASTA_AVALIACAO):
    os.makedirs(PASTA_AVALIACAO)


### ====================================
### Funções FPO front
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


### ====================================
### Funções módulo de tratamento alta
### ====================================
def validar_cpf(cpf):
    cpf = re.sub(r"\D", "", cpf).zfill(11)
    if len(cpf) != 11 or cpf == cpf[0] * len(cpf):
        return False

    def calcular_digito(digitos):
        soma = sum(int(d) * p for d, p in zip(digitos, range(len(digitos) + 1, 1, -1)))
        resto = soma % 11
        return 0 if resto < 2 else 11 - resto

    d1 = calcular_digito(cpf[:9])
    d2 = calcular_digito(cpf[:9] + str(d1))
    return cpf[-2:] == f"{d1}{d2}"


def validar_cnpj(cnpj):
    cnpj = re.sub(r"\D", "", cnpj).zfill(14)
    if len(cnpj) != 14 or cnpj == cnpj[0] * len(cnpj):
        return False

    def calcular_digito(digitos):
        pesos = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        soma = sum(int(d) * p for d, p in zip(digitos, pesos[-len(digitos) :]))
        resto = soma % 11
        return 0 if resto < 2 else 11 - resto

    d1 = calcular_digito(cnpj[:12])
    d2 = calcular_digito(cnpj[:12] + str(d1))
    return cnpj[-2:] == f"{d1}{d2}"


def normalizar_string(texto):
    substituicoes = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "à": "a",
        "è": "e",
        "ì": "i",
        "ò": "o",
        "ù": "u",
        "â": "a",
        "ê": "e",
        "î": "i",
        "ô": "o",
        "û": "u",
        "ã": "a",
        "õ": "o",
        "ç": "c",
    }
    texto = texto.lower()
    for original, substituto in substituicoes.items():
        texto = texto.replace(original, substituto)
    texto = re.sub(r"[^a-z0-9\s]", "", texto)
    return texto.upper().strip()


def validar_condicoes_especiais(dados):
    condicoes = [
        "garantias",
        "alienacao",
        "alien_moveis2",
        "obrigacoes",
        "movimentacao",
        "procuracao",
    ]
    for cond in condicoes:
        if dados.get(cond, "").upper() not in [
            "AUTORIZADO",
            "INFORMAÇÃO NÃO ENCONTRADA",
        ]:
            return False
    return True
