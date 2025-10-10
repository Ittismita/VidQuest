import google.generativeai as genai
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.embeddings import Embeddings


def get_embedding_model(choice: str, google_key: str, nvidia_key: str):
    if choice == "gemini":
        return GeminiEmbeddings(api_key=google_key)
    elif choice == "nvidia":
        emb = NVIDIAEmbeddings(nvidia_api_key=nvidia_key)
        print("Model being used:", emb.model)
        return emb
    else:
        raise ValueError("Invalid embedding model choice")



class GeminiEmbeddings(Embeddings):
    def __init__(self, model_name="models/text-embedding-004", api_key=None):
        self.model_name = model_name
        genai.configure(api_key=api_key)

    def embed_documents(self, docs):
        return [genai.embed_content(model=self.model_name, content=d, task_type="RETRIEVAL_DOCUMENT")["embedding"] for d in docs]

    def embed_query(self, query):
        return genai.embed_content(model=self.model_name, content=query, task_type="RETRIEVAL_DOCUMENT")["embedding"]




