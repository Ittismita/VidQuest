from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings

def text_chunker(text: str, chunk_size=450, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def semantic_chunker(text: str):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    splitter = SemanticChunker(embedding_model, breakpoint_threshold_amount=65.0, min_chunk_size=50 )
    # lower threshold = more splits → smaller chunks allow smaller chunks
    return splitter.split_text(text)


