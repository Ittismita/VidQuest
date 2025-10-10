from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY



def get_vectorstore(choice:str, documents, embedding_model, embedding_choice):
    if choice == "faiss":
            vectorstore = build_faiss(documents, embedding_model)
    else:
            
            vectorstore = build_pinecone(documents, embedding_model, PINECONE_API_KEY, embedding_choice)

    return vectorstore

        


def build_faiss(docs, embedding_model):
    return FAISS.from_documents(docs, embedding_model)

def build_pinecone(docs, embedding_model, api_key, embedding_choice):
    pc = Pinecone(api_key=api_key)

    if embedding_choice == "gemini":
        index_name="gemini-vs"
        dimension = 768   # for models/text-embedding-004
    elif embedding_choice == "nvidia":
        index_name="nvidia-vs"
        dimension = 1024  # for nvidia/nv-embedqa-e5-v5
    else:
        raise ValueError("model_choice must be 'gemini' or 'nvidia'")

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            metric="cosine",
            dimension=dimension,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return PineconeVectorStore.from_documents(docs, embedding_model, index_name=index_name)


