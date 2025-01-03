# Premissas:
# •	Efetua De_Para dos 74 objetos a serem cadastrados de acordo com a resposta das 6 perguntas destinadas a tipo de representação - condições especiais;
# o	garantias;
# o	alienacao;
# o	alien_moveis2;
# o	obrigacoes;
# o	movimentacao;
# o	procuracao.
# •	Aplica regra de condição especial conforme resultado das 6 perguntas mencionadas acima;
# •	O apelido combinacao_adm pode vir como "Informação não encontrada", se positivo deve-se preencher "administrador" no campo W109-ATRIBUICAO (layout FPO). Em caso de vir preenchido o campo é composto por grupos separados por vírgula onde dentro de cada grupo o primeiro elemento é o nome do grupo e os demais elementos são CPFs dos administradores desse grupo, estando sempre separados os elementos de um mesmo grupo por "|". Desta forma os CPFs de um mesmo grupo deverão ter o nome do grupo carregado no campo W109-ATRIBUICAO;
# •	O apelido combinacao_adm é composto por combinações separadas por vírgula onde dentro de cada combinação o primeiro elemento é o sequencial da combinação e os demais elementos são CPFs dos administradores que podem representar em conjunto, estando sempre separados os elementos de uma mesma combinação por "|". Os 74 objetos deverão ser cadastrados a cada aparição de um CPF em alguma combinação, conforme o De_Para de grupos de objetos e as 6 perguntas de condição especial;
# •	Os poderes devem ser cadastrados para todas as contas do cliente;
# •	Toda validação inconsistente deve ser descrita em relatório de recusa para tratativa manual do Back Office;

# Validações:
# Para o resultado da IA ser considerado para cadastro em FPO o código efetua as seguintes validações:
# 1.	Se há campo vazio ou como "Informação não encontrada" em algum dos apelidos obrigatórios: CNPJ_companhia, junta_comercial, data_ata, numero_registro_ata, data_contrato, consolidação, cpf_administradores, combinacao_adm e qtd_min_representacao;
# 2.	Se o apelido consolidacao está preenchido como "Não" deverá haver recusa para ser tratado via análise manual;
# 3.	Se o apelido numero_registro_ata = "NÃO ESTÁ REGISTRADO" deverá haver recusa para ser tratado via análise manual;
# 4.	Se o apelido qtd_min_representacao é um número >=1, caso contrário deverá haver recusa para ser tratado via análise manual;
# 5.	Se a quantidade de pipes no apelido combinacao_adm é igual ao número do apelido qtd_min_representacao, caso contrário deverá haver recusa para ser tratado via análise manual;
# 6.	Cada combinação do apelido combinacao_adm deve conter o primeiro elemento sendo um número único, caso contrário deverá haver recusa para ser tratado via análise manual;
# 7.	Um CPF pode aparecer somente uma vez a cada combinação do apelido combinacao_adm;
# 8.	Valida se o arquivo possui registros em seu conteúdo, , caso contrário deverá haver recusa para ser tratado via análise manual;
# 9.	Valida se o 'header do arquivo' possui 322 posições, caso contrário deverá haver recusa para ser tratado via análise manual;
# 10.	Se o CNPJ_companhia é um CNPJ válido, caso contrário deverá haver recusa para ser tratado via análise manual;
# 11.	Valida se o campo de data_ata é uma data válida, caso contrário deverá haver recusa para ser tratado via análise manual;
# 12.	Todos os CPFs do apelido cpf_administradores devem estar em ao menos uma combinação do campo combinacao_adm, caso contrário deverá haver recusa para ser tratado via análise manual;
# 13.	Valida se o CNPJ possui cadastro na tabela CCLTBAS;
# 14.	Valida se o CNPJ possui conta corrente na tabela CCLTRCT;
# 15.	Valida se o CNPJ possui filiais na tabela CCLTEND;
# 16.	Valida se combinacao_adm = 'INF NAO ENCONTRADA' e 'Qtd Min Assinatura' = '1', em caso positivo, aplica 'Combinacao' sequencial;
# 17.	Valida se a linha montada possui 2500 posições no total;


# Saída esperada:
# 029005874457116011840001610200000999999IA FPO         BOCONSO01.09.202301.01.0001020000000000000000000040706982487JOSE VALDYR SILVA DA FONSECA L01.09.202301.01.000100000000000000001AUT AB DESASADMINISTRADOR  ADMINISTRADOR                        JUCEPE 20238648877  11.10.2023N001
# 02900
# 5874457
# 11601184000161
# 02
# 00000
# 999999
# IA FPO         ;
# BO
# CONSO
# 01.09.2023
# 01.01.0001
# 02
# 00000000000000000
# 000
# 40706982487
# JOSE VALDYR SILVA DA FONSECA L
# 01.09.2023
# 01.01.0001
# 000000000000000
# 01
# AUT AB DES
# AS
# ADMINISTRADOR  ;
# ADMINISTRADOR  ;
#                ;
#     ;
#    ;
# JUCEPE 20238648877  ;
# 11.10.2023;
# N
# 001
#                                                                                                                                                                                                                                                                                ;
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               ;

import array
import pyodbc
import json
import requests
import re
from datetime import datetime
from utils import (
    validar_cpf,
    validar_cnpj,
    normalizar_string,
    validar_condicoes_especiais,
)


def processar_json(estrutura_json):
    dados = json.loads(estrutura_json)

    obrigatorios = [
        "cnpj_companhia",
        "junta_comercial",
        "data_ata",
        "numero_registro_ata",
        "data_contrato",
        "consolidacao",
        "cpf_administradores",
        "combinacao_adm",
        "qtd_min_representacao",
    ]

    for campo in obrigatorios:
        if not dados.get(campo) or dados[campo].upper() == "INFORMAÇÃO NÃO ENCONTRADA":
            raise ValueError(f"Campo obrigatório ausente ou inválido: {campo}")

    if dados["consolidacao"].upper() == "NÃO":
        raise ValueError("Consolidação marcada como 'NÃO'")

    if dados["numero_registro_ata"].upper() == "NÃO ESTÁ REGISTRADO":
        raise ValueError("Número de registro da ata inválido")

    if (
        not dados["qtd_min_representacao"].isdigit()
        or int(dados["qtd_min_representacao"]) < 1
    ):
        raise ValueError("Quantidade mínima de representação inválida")

    if validar_cnpj(dados["cnpj_companhia"]) is False:
        raise ValueError("CNPJ inválido")

    if not validar_condicoes_especiais(dados):
        raise ValueError("Condições especiais inválidas")

    cpfs_adm = dados["cpf_administradores"].split(",")
    combinacoes = dados["combinacao_adm"].split(",")

    for combinacao in combinacoes:
        elementos = combinacao.split("|")
        if not elementos[0].isdigit():
            raise ValueError("Sequencial de combinação inválido")
        if len(set(elementos[1:])) != len(elementos[1:]):
            raise ValueError("CPF duplicado em combinação")

    administradores_nao_encontrados = [
        cpf for cpf in cpfs_adm if not any(cpf in comb for comb in combinacoes)
    ]
    if administradores_nao_encontrados:
        raise ValueError(
            f"CPFs de administradores não encontrados em combinações: {administradores_nao_encontrados}"
        )

    return dados


def coletar_dados_CCL(cnpj_companhia, tipo_doc):
    # Tratar o CNPJ e separar em base e filial
    cnpj = re.sub(r"\D", "", cnpj_companhia)
    base = cnpj[:8].zfill(9)
    filial = cnpj[8:12].zfill(4)
    flag = ""

    urls = [
        "https://mf-hom.safra.com.br:9443/CCLQ16SEL01/T",
        "https://mf-hom.safra.com.br:9443/CCLQ16SEL02/T",
        "https://mf-hom.safra.com.br:9443/CCLQ16SEL03/T",
    ]

    queries = [
        {"AX_CD_B_CPF_CGC": base, "AX_FILIAL": filial},
        {"AX_CD_B_CPF_CGC": base, "AX_FILIAL": filial},
        {"AX_DOCUMENTO": tipo_doc},
    ]

    headers = {"Content-Type": "application/json"}

    try:
        # Primeira conexão no CCL (CCLQ16SEL01)
        response1 = requests.post(
            urls[0], headers=headers, json=queries[0], verify=False
        )
        if response1.status_code != 200 or response1.json()["StatusCode"] != 200:
            flag = f"Erro ao fazer o POST na primeira query: {response1.text}"

        data1 = response1.json()["ResultSet Output"]
        if not data1:  # Se a resposta for vazia
            print("Resposta da primeira query vazia. Tentando a segunda query...")

            # Segunda conexão no CCL (CCLQ16SEL02)
            response2 = requests.post(
                urls[1], headers=headers, json=queries[1], verify=False
            )
            if response2.status_code != 200 or response2.json()["StatusCode"] != 200:
                flag = f"Erro ao fazer o POST na segunda query: {response2.text}"

            data2 = response2.json()["ResultSet Output"]
            if not data2:  # Resposta vazia da segunda query
                flag = "Erro: Cliente não cadastrado."

        else:  # Resposta positiva da primeira query
            print("Resposta positiva da primeira query. Tentando a terceira query...")

        # Terceira conexão no CCL (CCLQ16SEL03)
        response3 = requests.post(
            urls[2], headers=headers, json=queries[2], verify=False
        )
        if response3.status_code != 200 or response3.json()["StatusCode"]:
            flag = f"Erro ao fazer o POST na terceira query: {response3.text}"

        if not data1:
            return data2, response3.json()["ResultSet Output"], flag
        else:
            return data1, response3.json()["ResultSet Output"], flag
    except requests.exceptions.RequestException as e:
        return f"Erro ao conectar: {str(e)}"


def formatar_saida(dados, administrador, objeto, posicao):
    cpf_administrador = dados["cpf_administradores"].split(",")[posicao]
    grupo_matriz = ""
    objeto_matriz = ""
    ato_matriz = ""
    atribuicao_ia = ""
    combinacao_ia = ""

    objeto_condicao_especial = objeto.split("-")[3]

    combinacoes = dados["combinacao_adm"].split(", ")
    grupos = dados["grupo_administradores"].split(", ")
    for combinacao in combinacoes:
        # Separando o número da combinação e os CPFs/CNPJs
        numero, *cpfs = combinacao.split("|")
        # Verificando se o CPF/CNPJ está na combinação
        if cpf_administrador in cpfs:
            combinacao_ia = numero
            break

    for grupo in grupos:
        # Separando o nome do grupo e os CPFs/CNPJs
        nome_grupo, *cpfs = grupo.split("|")
        # Verificando se o CPF/CNPJ está no grupo
        if cpf_administrador in cpfs:
            atribuicao_ia = nome_grupo
            break

    agencia = dados["agencia"].ljust(5)
    conta = dados["conta"].ljust(7)
    cnpj = dados["cnpj_companhia"].ljust(14)
    tipo_conta = "02".ljust(2)
    cod_gerente = "00000".ljust(5)
    matricula = "999999".ljust(6)
    operador = "IA FPO".ljust(15)
    tipo_usuario = "BO".ljust(2)
    tipo_doc = "CONSO".ljust(5)
    data_inicio_doc = dados["data_contrato"].ljust(10)
    data_fim_doc = "01.01.0001".ljust(10)
    qtd_assinaturas = dados["qtd_min_representacao"].ljust(2)
    valor_alcada = "00000000000000000".ljust(17)
    prazo_alcada = "000".ljust(3)
    cpf_rep = cpf_administrador.ljust(11)
    nome_rep = administrador.ljust(30)
    data_inicio_mandato = dados["data_contrato"].ljust(10)
    data_fim_mandato = "01.01.0001".ljust(10)
    perc_rep = "000000000000000".ljust(15)
    grupo = dados["matriz"].ljust(2)
    objeto = dados["matriz"].ljust(10)
    ato = dados["matriz"].ljust(2)
    cargo = "ADMINISTRADOR".ljust(15)
    atribuicao = atribuicao_ia.ljust(15)
    num_proc = "".ljust(15)
    livro = "".ljust(4)
    folha = "".ljust(3)
    junta_comercial = (dados["junta_comercial"] + dados["numero_registro_ata"]).ljust(
        20
    )
    data_junta_comercial = dados["data_ata"].ljust(10)
    flag_cond_espec = dados["poderes"][posicao]["COND_ESP"].ljust(1)
    combinacao = combinacao_ia.ljust(3)
    filler = "".ljust(271)
    cond_espec = dados[objeto_condicao_especial].ljust(1950)

    # Monta o texto final
    linha = (
        agencia
        + conta
        + cnpj
        + tipo_conta
        + cod_gerente
        + matricula
        + operador
        + tipo_usuario
        + tipo_doc
        + data_inicio_doc
        + data_fim_doc
        + qtd_assinaturas
        + valor_alcada
        + prazo_alcada
        + cpf_rep
        + nome_rep
        + data_inicio_mandato
        + data_fim_mandato
        + perc_rep
        + grupo
        + objeto
        + ato
        + cargo
        + atribuicao
        + num_proc
        + livro
        + folha
        + junta_comercial
        + data_junta_comercial
        + flag_cond_espec
        + combinacao
        + filler
        + cond_espec
    )

    # Validar tamanho da linha
    if len(linha) != 2500:
        raise ValueError(
            f"A linha gerada não possui 2500 caracteres. Tamanho atual: {len(linha)}"
        )

    return linha


def gerar_arquivo_txt(dados, caminho_arquivo):
    array_obj_contrato = [
        "4-AL F B MOV-AS-alien_moveis2",
        "4-ANEXO CAUC-AS-alien_moveis2",
        "4-ANEXO CHEQ-AS-alien_moveis2",
        "4-ANEXO DUPL-AS-alien_moveis2",
        "4-CESSF MOV-AS-alien_moveis2",
        "4-CPR-AS-alien_moveis2",
        "4-PENHOR-AS-alien_moveis2",
        "4-PENHOR AGR-AS-alien_moveis2",
        "4-PENHOR IND-AS-alien_moveis2",
        "4-PENHOR MER-AS-alien_moveis2",
        "3-AL F B IMO-AS-alienacao",
        "3-CED/N CRED-AS-alienacao",
        "3-HIPOTECA-AS-alienacao",
        "2-GAR P/TERC-AS-garantias",
        "1-AUT AB DES-AS-movimentacao",
        "1-AUT DEBITO-AS-movimentacao",
        "1-BORD COB-AS-movimentacao",
        "1-BORD DESC-AS-movimentacao",
        "1-BORDCHEQUE-AS-movimentacao",
        "1-BORDDUPL-AS-movimentacao",
        "1-BX TIT COB-AS-movimentacao",
        "1-C CAMBIO-AS-movimentacao",
        "1-CARTA CIRC-AS-movimentacao",
        "1-CHEQUES-*-movimentacao",
        "1-CONHEC DEP-EN-movimentacao",
        "1-CONV CESS-AS-movimentacao",
        "1-CORRESP-AS-movimentacao",
        "1-DEV SOLID-AS-movimentacao",
        "1-DOC-AS-movimentacao",
        "1-DUPLICATAS-*-movimentacao",
        "1-F PRO POUP-AS-movimentacao",
        "1-F PROPOSTA-AS-movimentacao",
        "1-FIEL DEP-AS-movimentacao",
        "1-INSTR COB-AS-movimentacao",
        "1-L CAMBIO-*-movimentacao",
        "1-ORDEM PGTO-AS-movimentacao",
        "1-PROR VENCT-AS-movimentacao",
        "1-PROTESTO-AS-movimentacao",
        "1-QUITACOES-AS-movimentacao",
        "1-RECIBOS-AS-movimentacao",
        "1-REQ TALOES-AS-movimentacao",
        "1-SOLIC EXTR-AS-movimentacao",
        "1-SOLIC SALD-AS-movimentacao",
        "1-SUST PROT-AS-movimentacao",
        "1-TED-AS-movimentacao",
        "1-TRANSF SDO-AS-movimentacao",
        "1-WARRANTS-EN-movimentacao",
        "5-C AB CRED-AS-obrigacoes",
        "5-C ABT CRED-AS-obrigacoes",
        "5-C ARREND M-AS-obrigacoes",
        "5-C CDC-AS-obrigacoes",
        "5-C CES CRED-AS-obrigacoes",
        "5-C CESS CRE-AS-obrigacoes",
        "5-C CHAN MEC-AS-obrigacoes",
        "5-C COMPROR-AS-obrigacoes",
        "5-C EMPREST-AS-obrigacoes",
        "5-C FINAME-AS-obrigacoes",
        "5-C FINANC-AS-obrigacoes",
        "5-C HEGDE-AS-obrigacoes",
        "5-C LEASING-AS-obrigacoes",
        "5-C MUTUO-AS-obrigacoes",
        "5-C PRES SER-AS-obrigacoes",
        "5-C SWAP-AS-obrigacoes",
        "5-CAMBIO DIV-AS-obrigacoes",
        "5-CARTA FIAN-AS-obrigacoes",
        "5-CCB-AS-obrigacoes",
        "5-CONTR ARV-AS-obrigacoes",
        "5-CONTR VEND-AS-obrigacoes",
        "5-CONTRATOS-AS-obrigacoes",
        "5-CTR SEGURO-AS-obrigacoes",
        "5-N PROMISSO-*-obrigacoes",
        "5-NDF-AS-obrigacoes",
        "5-NOTIF/TRAV-AS-obrigacoes",
        "6-PROCURACAO-AS-procuracao",
    ]

    ag_cc, poderes, flag = coletar_dados_CCL(dados["cnpj_companhia"], "CONSO")
    if flag:
        return flag

    dados["agencia"] = ag_cc["AGE"]
    dados["conta"] = ag_cc["CONTA"]

    dados["poderes"] = poderes

    # Cada administrador da lista tem que ter 74 linhas, e cada linha é associada a um objeto do array
    lista_linhas = []
    for administrador in dados["nome_administradores"].split(","):
        for posicao, objeto in enumerate(array_obj_contrato):
            linha = formatar_saida(dados, administrador, objeto, posicao)
            lista_linhas.append(linha)
    
    string_linhas = "\n".join(lista_linhas)
    with open(caminho_arquivo, "w") as arquivo:
        arquivo.write(string_linhas)


def main():
    # 1. Pulling no banco de dados para ver se tem registros a serem processados
    conn = pyodbc.connect(
        "DRIVER={SQL Server};SERVER=server_name;DATABASE=database_name;UID=user;PWD=password"
    )
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, estrutura_json, status FROM tabela WHERE status = 'PENDENTE'"
    )
    registros = cursor.fetchall()

    # Processamento de cada registro com status PENDENTE
    for registro in registros:
        id_registro, estrutura_json, status = registro

        try:
            # Módulo de processamento e enriquecimento dos dados
            dados_processados = processar_json(estrutura_json)

            # Geração do arquivo com base no array de objetos esperado pelo FPO Alta
            caminho_arquivo = f"registro_{id_registro}.txt"
            flag = gerar_arquivo_txt(dados_processados, caminho_arquivo)

            # Atualização de status do registro no banco de dados
            if not flag:
                cursor.execute(
                    "UPDATE tabela SET status = ?, data_processamento = ? WHERE id = ?",
                    ("PROCESSADO", datetime.now(), id_registro),
                )
                conn.commit()
            else:
                cursor.execute(
                    "UPDATE tabela SET status = ?, data_processamento = ? WHERE id = ?",
                    (flag, datetime.now(), id_registro),
                )
                conn.commit()

        except Exception as e:
            cursor.execute(
                "UPDATE tabela SET status = ?, mensagem_erro = ? WHERE id = ?",
                ("RECUSADO", str(e), id_registro),
            )
            conn.commit()

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()
