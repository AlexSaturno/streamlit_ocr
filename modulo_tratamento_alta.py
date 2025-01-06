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
    de_para_numeros,
)


def processar_json(estrutura_json):
    # dados = json.loads(estrutura_json)
    dados = estrutura_json

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

    if dados["combinacao_adm"] == "Informação não encontrada":
        dados["combinacao_adm"] = "1|" + "|".join(
            dados["cpf_administradores"].split(",")
        )
        dados["qtd_min_representacao"] = "1"
        print("Combinação editada: ", dados["combinacao_adm"])

    for campo in obrigatorios:
        if not dados.get(campo) or dados[campo].upper() == "INFORMAÇÃO NÃO ENCONTRADA":
            raise ValueError(f"Campo obrigatório ausente ou inválido: {campo}")

    if dados["consolidacao"].upper() == "NÃO":
        raise ValueError("Consolidação marcada como 'NÃO'")

    if dados["numero_registro_ata"].upper() == "NÃO ESTÁ REGISTRADO":
        raise ValueError("Número de registro da ata inválido")

    if not dados["qtd_min_representacao"].isdigit():
        dados["qtd_min_representacao"] = de_para_numeros(dados["qtd_min_representacao"])
    elif int(dados["qtd_min_representacao"]) < 1:
        raise ValueError("Quantidade mínima de representação inválida")

    if validar_cnpj(dados["cnpj_companhia"]) is False:
        raise ValueError("CNPJ inválido")
    dados["cnpj_companhia"] = re.sub(r"\D", "", dados["cnpj_companhia"])
    dados["numero_registro_ata"] = re.sub(r"\D", "", dados["numero_registro_ata"])
    dados["data_ata"] = dados["data_ata"].replace("/", ".")
    dados["data_contrato"] = dados["data_contrato"].replace("/", ".")

    # if not validar_condicoes_especiais(dados):
    #     raise ValueError("Condições especiais inválidas")

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
    base = cnpj_companhia[:8].zfill(9)
    filial = cnpj_companhia[8:12].zfill(4)
    flag = ""

    urls = [
        "https://mf-hom.safra.com.br:9443/CCLQ16SEL01/H",
        "https://mf-hom.safra.com.br:9443/CCLQ16SEL02/H",
        "https://mf-hom.safra.com.br:9443/CCLQ16SEL03/H",
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
        if response3.status_code != 200 or response3.json()["StatusCode"] != 200:
            flag = f"Erro ao fazer o POST na terceira query: {response3.text}"

        if not data1:
            return data2, response3.json()["ResultSet Output"], flag
        else:
            return data1, response3.json()["ResultSet Output"], flag
    except requests.exceptions.RequestException as e:
        return f"Erro ao conectar: {str(e)}"


def formatar_saida(
    dados, numero_combinacao, administrador, posicao_adm, objeto, posicao_obj
):
    def ajustar_tamanho_str(info, tamanho):
        if len(info) > tamanho:
            return info[:tamanho]
        return info.ljust(tamanho)

    def ajustar_tamanho_numero(info, tamanho):
        return str(info).zfill(tamanho)

    cpf_administrador = re.sub(
        r"\D", "", (dados["cpf_administradores"].split(",")[posicao_adm])
    )
    atribuicao_ia = ""

    objeto_condicao_especial = objeto.split("-")[3]

    grupos = dados["grupo_administradores"].split(", ")
    for grupo in grupos:
        # Separando o nome do grupo e os CPFs/CNPJs
        nome_grupo, *cpfs = grupo.split("|")
        # Verificando se o CPF/CNPJ está no grupo
        if cpf_administrador in cpfs:
            atribuicao_ia = nome_grupo
            break

    agencia = ajustar_tamanho_numero(dados["agencia"], 5)
    conta = ajustar_tamanho_numero(dados["conta"], 7)
    cnpj = ajustar_tamanho_numero(dados["cnpj_companhia"], 14)
    tipo_conta = ajustar_tamanho_str("02", 2)
    cod_gerente = ajustar_tamanho_str("00000", 5)
    matricula = ajustar_tamanho_str("999999", 6)
    operador = ajustar_tamanho_str("IA FPO", 15)
    tipo_usuario = ajustar_tamanho_str("BO", 2)
    tipo_doc = ajustar_tamanho_str("CONSO", 5)
    data_inicio_doc = ajustar_tamanho_str(dados["data_contrato"], 10)
    data_fim_doc = ajustar_tamanho_str("01.01.0001", 10)
    qtd_assinaturas = ajustar_tamanho_numero(dados["qtd_min_representacao"], 2)
    valor_alcada = ajustar_tamanho_str("00000000000000000", 17)
    prazo_alcada = ajustar_tamanho_str("000", 3)
    cpf_rep = ajustar_tamanho_numero(cpf_administrador, 11)
    nome_rep = ajustar_tamanho_str(administrador, 30)
    data_inicio_mandato = ajustar_tamanho_str(dados["data_contrato"], 10)
    data_fim_mandato = ajustar_tamanho_str("01.01.0001", 10)
    perc_rep = ajustar_tamanho_str("000000000000000", 15)
    grupo = ajustar_tamanho_str(dados["poderes"][posicao_obj]["GRP_OBJ"], 2)
    objeto = ajustar_tamanho_str(dados["poderes"][posicao_obj]["OBJETO"], 10)
    ato = ajustar_tamanho_str(dados["poderes"][posicao_obj]["ATO"], 2)
    cargo = ajustar_tamanho_str("ADMINISTRADOR", 15)
    atribuicao = ajustar_tamanho_str(atribuicao_ia, 15)
    num_proc = ajustar_tamanho_str("", 15)
    livro = ajustar_tamanho_str("", 4)
    folha = ajustar_tamanho_str("", 3)
    junta_comercial = ajustar_tamanho_str(
        dados["junta_comercial"] + dados["numero_registro_ata"], 20
    )
    data_junta_comercial = ajustar_tamanho_str(dados["data_ata"], 10)
    flag_cond_espec = ajustar_tamanho_str(dados["poderes"][posicao_obj]["COND_ESP"], 1)
    combinacao = ajustar_tamanho_numero(numero_combinacao, 3)
    filler = ajustar_tamanho_str("", 271)
    if flag_cond_espec == "N":
        cond_espec = ajustar_tamanho_str("", 1950)
    else:
        cond_espec = ajustar_tamanho_str(dados[objeto_condicao_especial], 1950)

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
        print(linha)
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

    dados["poderes"] = poderes

    lista_nomes = dados["nome_administradores"].split(",")

    # Separando o número da combinação e os CPFs/CNPJs
    combinacoes = dados["combinacao_adm"].split(", ")

    # Cada administrador da lista tem que ter pelo menos 74 linhas, e cada linha é associada a um objeto do array
    # Isso deve ser processado para cada combinação em que o administrador fizer parte
    # E esse processo todo deve ser replicado para cada conta que esse CNPJ possuir (query1/query2)
    lista_linhas = []
    nomes_na_combinacao = []
    for combinacao in combinacoes:
        numero_combinacao, *cpfs_da_combinacao = combinacao.split("|")
    for cpf in cpfs_da_combinacao:
        index = cpfs_da_combinacao.index(cpf)
        nomes_na_combinacao.append(lista_nomes[index])

    for posicao_conta, agencia_conta in enumerate(ag_cc):
        dados["agencia"] = ag_cc[posicao_conta][
            "AGE"
        ]  # COD_CLI; BASE_CNPJ; FILIAL; DV; AGE; CONTA
        dados["conta"] = ag_cc[posicao_conta]["CONTA"]
        for administrador in nomes_na_combinacao:
            for posicao_obj, objeto in enumerate(array_obj_contrato):
                linha = formatar_saida(
                    dados, numero_combinacao, administrador, index, objeto, posicao_obj
                )
                lista_linhas.append(linha)

        # for posicao_adm, administrador in enumerate(dados["nome_administradores"].split(",")):
        #     for posicao_combinacao, combinacao in enumerate(dados["combinacao_adm"].split(", ")):
        #         for posicao_obj, objeto in enumerate(array_obj_contrato):
        #             linha = formatar_saida(dados, administrador, objeto, posicao_adm, posicao_obj)
        #             lista_linhas.append(linha)

    string_linhas = "\n".join(lista_linhas)
    print("Quantidade de linhas: ", len(lista_linhas))
    with open(caminho_arquivo, "w") as arquivo:
        arquivo.write(string_linhas)


def main():
    # 1. Pulling no banco de dados para ver se tem registros a serem processados
    driver_name = ""
    driver_names = [x for x in pyodbc.drivers() if x.endswith(" for SQL Server")]
    if driver_names:
        driver_name = driver_names[0]

    if driver_name:
        server_name = "SDDB039VT"
        database_name = "DB_FPO_LOGS"
        conn = pyodbc.connect(
            f"DRIVER={driver_name};SERVER={server_name};DATABASE={database_name}; Trusted_Connection=Yes;"
        )
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, estrutura_json, status FROM tabela WHERE status = 'PENDENTE'"
        )
        registros = cursor.fetchall()
    else:
        print("(No suitable driver found. Cannot connect.)")

    estrutura_json = {
        "nome_companhia": "DIMA COSMÉTICOS LTDA EPP",
        "cnpj_companhia": "17.439.839/0001-23",
        "junta_comercial": "JUCEPE",
        "data_ata": "22/08/2023",
        "numero_registro_ata": "20238859916",
        "data_contrato": "10/08/2023",
        "consolidacao": "sim",
        "nome_administradores": "Diogo Andrade Rodrigues, Manuela Barbosa de Albuquerque e Mello",
        "cpf_administradores": "028.266.124-76, 036.133.154-16",
        "mandato_administradores": "Informação não encontrada",
        "grupo_administradores": "Informação não encontrada",
        "combinacao_adm": "Informação não encontrada",
        "qtd_min_representacao": "Informação não encontrada",
        "CPF_CNPJ_socios": "028.266.124-76, 036.133.154-16",
        "Tipo_socios": "CPF, CPF",
        "garantias": "Cláusula Décima, parágrafo único: (...) vedado, no entanto, fazê-lo em atividades estranhas ao interesse social ou assumir obrigações seja em favor de qualquer dos quotistas ou de terceiros, bem como onerar ou alienar bens imóveis da sociedade, sem autorização do(s) outro(s) sócio(s).",
        "alienacao": "Cláusula Décima, parágrafo único: (...) vedado, no entanto, fazê-lo em atividades estranhas ao interesse social ou assumir obrigações seja em favor de qualquer dos quotistas ou de terceiros, bem como onerar ou alienar bens imóveis da sociedade, sem autorização do(s) outro(s) sócio(s).",
        "alien_moveis2": "Informação não encontrada",
        "obrigacoes": "Informação não encontrada",
        "movimentacao": "Informação não encontrada",
        "procuracao": "Informação não encontrada",
    }

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
