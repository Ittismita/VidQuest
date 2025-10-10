from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

def build_retriever(vectorstore, retriever_type="basic", nvidia_key=None):
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    
    print()
    if retriever_type == "rerank" and nvidia_key:
        reranker = NVIDIARerank(nvidia_api_key=nvidia_key)#max allowed token size is 512
        return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever, document_compressor_kwargs={"split_documents": True}, top_n=4)
    return retriever
