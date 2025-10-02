
import time as tm
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.agent_toolkits import create_retriever_tool
import langchain
from hashlib import sha256
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
import streamlit as st 
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain import hub
from langchain.chains import create_retrieval_chain
from pydantic import SecretStr
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb import PersistentClient
import chromadb
from langchain.chains import ConversationalRetrievalChain


# api_key = os.environ.get("OLLAMA_API_KEY")

huggingface_api_key = SecretStr("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")
secret_str_api_key = huggingface_api_key.get_secret_value()

db_path = "./vectordb"

hg_api_key = os.getenv("HP_Token")

hg_api_key_for_st = st.secrets["HP_TOKEN"]

embeddings = HuggingFaceEndpointEmbeddings(model="mixedbread-ai/mxbai-embed-large-v1",huggingfacehub_api_token=hg_api_key_for_st)


# a


vector_db = Chroma(
    embedding_function=embeddings,
    persist_directory=db_path,   # disk yerine RAM kullan
    client_settings=Settings(
        is_persistent=False,
        allow_reset=True,
        persist_directory = db_path
    )
)


# vectordb was deleted
api_key = os.environ.get("oJ6wgJeUMlciaLyoojF2OUancT1FoOAe")



# vector_db = Chroma(embedding_function=embeddings,client=client,persist_directory=db_path,collection_name=collection)


if "messages" not in st.session_state:
        st.session_state.messages = []
        mesaj = "You are an assistant for question answering tasks"
        st.session_state.messages.append(SystemMessage(content=mesaj))


prompt = ChatPromptTemplate.from_messages([
        ("system"),("Bu belgelerden sana soru sorulacak {context}"),
        ("human"),("{input}")

])


# Geçmiş mesajları göster
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Kullanıcı girişi
asked_question = st.chat_input("Your question:")
if asked_question:
    with st.chat_message("user"):
        st.markdown(asked_question)
        st.session_state.messages.append(HumanMessage(asked_question))

# command line 

    llm = ChatMistralAI(model_name="magistral-small-2509",api_key="oJ6wgJeUMlciaLyoojF2OUancT1FoOAe")
    document_chain = create_stuff_documents_chain(llm=llm,prompt=prompt)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    retriever_chain = create_retrieval_chain(retriever,document_chain)
    result = retriever_chain.invoke({
        "input": asked_question
    })
    try:
        unique_set = set()
        source_text = "Kaynaklar:\n"

        responseofAI = result["answer"]  # cevabı al
        for key in list(result.keys()):
            print("Value of Key : ",key)
        
        

        # source_docs = result["source_documents"]  # kaynakları al

        # for doc in source_docs:
        #     title = doc.metadata.get("source", "bilinmeyen")  # metadata içinden source alanı
        #     if title not in unique_set:
        #         unique_set.add(title)
        #         source_text += f"- {title}\n"

        with st.chat_message("assistant"):
            st.markdown(responseofAI)
            st.session_state.messages.append(AIMessage(content=responseofAI))
            # st.session_state.messages.append(AIMessage(content=source_text))

    except Exception as Hata:
        print("Hata Var : ",Hata)
    # py -m streamlit run streamlit_app.py 
    # gitattiributes deleted
    # düzenleme
    # sil

    