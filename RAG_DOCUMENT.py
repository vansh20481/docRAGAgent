import os
import sys
import asyncio
import uuid
import streamlit as st
from dotenv import load_dotenv

if sys.platform == "win32" and sys.version_info < (3, 10):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from streamlit import config
config.set_option("server.fileWatcherType", "none")

from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.chat_message_histories import ChatMessageHistory

from PyPDF2 import PdfReader
from docx import Document


load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    st.error("Please add HUGGINGFACEHUB_API_TOKEN to your .env file")
    st.stop()

st.set_page_config(page_title="LCEL RAG Chat", layout="wide")
st.title(" Chat with Your Documents (LCEL RAG)")

if "chats" not in st.session_state:
    st.session_state["chats"] = {}

if "active_chat_id" not in st.session_state:
    chat_id = f"chat_{uuid.uuid4().hex[:8]}"
    st.session_state["chats"][chat_id] = ChatMessageHistory()
    st.session_state["active_chat_id"] = chat_id


st.sidebar.markdown("## Chats")

if st.sidebar.button(" New Chat"):
    new_chat_id = f"chat_{uuid.uuid4().hex[:8]}"
    st.session_state["chats"][new_chat_id] = ChatMessageHistory()
    st.session_state["active_chat_id"] = new_chat_id
    st.rerun()

chat_ids = list(st.session_state["chats"].keys())
selected_chat = st.sidebar.selectbox(
    "Resume chat",
    chat_ids,
    index=chat_ids.index(st.session_state["active_chat_id"])
)
st.session_state["active_chat_id"] = selected_chat

if st.sidebar.button(" Clear Current Chat"):
    st.session_state["chats"][st.session_state["active_chat_id"]] = ChatMessageHistory()
    st.rerun()

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, DOCX, or TXT files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

def read_pdf(file):
    reader = PdfReader(file)
    return "".join(page.extract_text() or "" for page in reader.pages)

def read_docx(file):
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def read_txt(file):
    return file.read().decode("utf-8")

def extract_text(files):
    text = ""
    for file in files:
        if file.type == "application/pdf":
            text += read_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text += read_docx(file)
        elif file.type == "text/plain":
            text += read_txt(file)
    return text


if uploaded_files:
    with st.spinner("Processing documents and building index..."):

        raw_text = extract_text(uploaded_files)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(raw_text)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_texts(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        llm_endpoint = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            max_new_tokens=512,
            do_sample=False,
        )

        llm = ChatHuggingFace(llm=llm_endpoint)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant. Answer ONLY using the provided context. "
             "If the answer is not present in the context, say 'I don't know'."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
            ("system", "Context:\n{context}")
        ])

        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        #  RAG chain with sources
        rag_chain = (
            {
                "docs": RunnableLambda(lambda x: retriever.invoke(x["question"])),
                "question": RunnableLambda(lambda x: x["question"]),
                "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
            }
            | RunnableLambda(lambda x: {
                "answer": llm.invoke(
                    prompt.invoke({
                        "context": format_docs(x["docs"]),
                        "question": x["question"],
                        "chat_history": x["chat_history"],
                    })
                ),
                "sources": x["docs"],
            })
        )

        def get_session_history(session_id: str):
            return st.session_state["chats"][st.session_state["active_chat_id"]]

        rag_chain_with_memory = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

        st.session_state["rag_chain"] = rag_chain_with_memory
        st.success("Documents indexed. You can start chatting!")

else:
    st.info("⬅ Upload documents from the sidebar to begin.")


if "rag_chain" in st.session_state:

    current_history = st.session_state["chats"][st.session_state["active_chat_id"]]

    # Render past messages
    for msg in current_history.messages:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    user_input = st.chat_input("Ask a question about your documents")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Generating answer..."):
            result = st.session_state["rag_chain"].invoke(
                {"question": user_input},
                config={"configurable": {"session_id": st.session_state["active_chat_id"]}}
            )

        answer = result["answer"].content
        sources = result["sources"]

        with st.chat_message("assistant"):
            st.markdown(answer)

            with st.expander("Sources used"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Source {i+1}:**")
                    st.markdown(doc.page_content[:400] + "…")
