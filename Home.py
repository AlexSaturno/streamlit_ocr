### ====================================
### Importações e Globais
### ====================================
import streamlit as st
from utils import cria_chain_conversa, PASTA_ARQUIVOS


### ====================================
### Funções
### ====================================
def sidebar():
    uploaded_pdfs = st.file_uploader(
        "Adicione seus arquivos PDF", type=[".pdf"], accept_multiple_files=True
    )
    if (
        not uploaded_pdfs is None
    ):  # Se tiver PDFs na pasta quando inicializar a aplicação, apagá-los
        for arquivo in PASTA_ARQUIVOS.glob("*.pdf"):
            arquivo.unlink()
        for pdf in uploaded_pdfs:  # Salvar o PDF carregado na pasta "arquivos"
            with open(PASTA_ARQUIVOS / pdf.name, "wb") as f:
                f.write(pdf.read())

    label_botao = "Inicializar Chatbot"
    if (
        "chain" in st.session_state
    ):  # Se tiver conversa na session_state o botão altera para Atualizar
        label_botao = "Atualizar Chatbot"
    if st.button(label_botao, use_container_width=True):
        if (
            len(list(PASTA_ARQUIVOS.glob("*.pdf"))) == 0
        ):  # Se não tiver arquivos ao clicar o botão, pedir pra incluir
            st.error("Adicione arquivos PDF para inicializar o chatbot")
        else:  # Se tiver arquivos, inicializa a conversa
            st.success("Processando OCR...")
            cria_chain_conversa()
            st.rerun()  # Rerun para tirar a mensagem de inicialização


def chat_window():
    st.header("Chat", divider=True)
    if not "chain" in st.session_state:
        st.error("Faça o upload de PDFs para começar")
        st.stop()  # Faz com que o código não rode pra baixo

    chain = st.session_state["chain"]
    memory = chain.memory

    mensagens = memory.load_memory_variables({})["chat_history"]

    # Container para exibição no estilo Chat message
    container = st.container()
    for mensagem in mensagens:
        chat = container.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    # Espaço para o humano incluir a mensagem e estruturação da conversa
    nova_mensagem = st.chat_input("Digite uma mensagem")
    if nova_mensagem:
        chat = container.chat_message("human")
        chat.markdown(nova_mensagem)
        chat = container.chat_message("ai")
        chat.markdown("Gerando resposta")

        resposta = chain.invoke({"question": nova_mensagem})
        st.session_state["ultima_resposta"] = resposta
        st.rerun()


def main():
    with st.sidebar:
        sidebar()
    chat_window()


if __name__ == "__main__":
    main()
