from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.tools.retriever import create_retriever_tool
import os

def naive_rag(vectorstore_name):
    # Add to vectorDB
    print(os.path.join(os.path.dirname(os.path.dirname(__file__)), vectorstore_name, "vectorstore"))
    vectorstore = Chroma(
        collection_name=vectorstore_name,
        embedding_function=OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5:latest"),
        persist_directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb", vectorstore_name, "vectorstore"),
    )

    retriever = vectorstore.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        f"{vectorstore_name}_knowledge_base_tool",
        f"search and return information about {vectorstore_name}",
    )
    return retriever_tool