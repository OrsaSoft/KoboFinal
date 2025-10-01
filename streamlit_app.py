
import time as tm
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.agent_toolkits import create_retriever_tool
import langchain
# from langchain_community.embeddings import OllamaEmbeddings
from hashlib import sha256
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
import streamlit as st 
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from pydantic import SecretStr
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from chromadb import PersistentClient
print(f"LangChain version: {langchain.__version__}") # 0.3.27




# api_key = os.environ.get("OLLAMA_API_KEY")

huggingface_api_key = SecretStr("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")
secret_str_api_key = huggingface_api_key.get_secret_value()


embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
# embeddings = HuggingFaceInferenceAPIEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1",api_key="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",api_url="https://huggingface.co/mixedbread-ai/deepset-mxbai-embed-de-large-v1?library=sentence-transformers")
embeddings = HuggingFaceEndpointEmbeddings(model="mixedbread-ai/mxbai-embed-large-v1",huggingfacehub_api_token="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")

client = PersistentClient(path="./vectordb")
db = client.get_or_create_collection(name="my_collection",metadata={"tenant" : "default_tenant","database" : "default_database"})

# vectordb was deleted
api_key = os.environ.get("oJ6wgJeUMlciaLyoojF2OUancT1FoOAe")
db_path = "vectordb"
vector_db = Chroma(persist_directory=db_path,embedding_function=embeddings)


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
    retriever = vector_db.as_retriever()
    retriever_chain = create_retrieval_chain(retriever,document_chain)
    result = retriever_chain.invoke({
        "input": asked_question
    })
    responseofAI = result["answer"]

    with st.chat_message("assistant"):
        st.markdown(responseofAI)
        st.session_state.messages.append(AIMessage(content=responseofAI))

    # py -m streamlit run streamlit_app.py 
    # gitattiributes deleted