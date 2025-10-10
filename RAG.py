from dotenv import load_dotenv
load_dotenv()

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from youtube_transcript_api.formatters import JSONFormatter
from youtube_transcript_api.formatters import TextFormatter
import google.generativeai as genai
from youtube_transcript_api.proxies import WebshareProxyConfig
from youtube_transcript_api.proxies import GenericProxyConfig
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.embeddings import Embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate




#document ingestion/loading

def fetch_transcript(video_id):
  try:
    """ ytt_api = YouTubeTranscriptApi(
        proxy_config=GenericProxyConfig(
            http_url="http://mrnhrbyp:u2nkoaiif54z@my-custom-proxy.org:7030",
            https_url="https://mrnhrbyp:u2nkoaiif54z@my-custom-proxy.org:7030",
        )
    )

        all requests done by ytt_api will be proxied through Webshare
        ytt_api = YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username="mrnhrbyp",
                proxy_password="u2nkoaiif54z",
                filter_ip_locations=["de", "us"],
            )
        )"""


    ytt_api = YouTubeTranscriptApi()

    jsonFormatter=JSONFormatter()
    json_transcripts=jsonFormatter.format_transcript(transcript_list)

    textFormatter=TextFormatter()
    text_transcripts=textFormatter.format_transcript(transcript_list)

    transcript_list=ytt_api.fetch(video_id, languages=["en"])

    #print(transcript_list)#returns a list of FetchTranscriptSnippet objects
    #print(json_transcripts)

    #flattening to plain text
    # transcript=transcript_list.join(chunk["text"] for chunk in transcript_list)
    # print(transcript)

  except TranscriptsDisabled:
    print("No Transcripts available")

  return text_transcripts



#Chunking

#RecursiveCharacterTextSplitter
def rec_splitter(transcript):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    texts = text_splitter.split_text(transcript)

    rec_splitted_texts=[]
    for chunk in texts:
        rec_splitted_texts.append(Document(page_content=chunk))

    return rec_splitted_texts

#Semantic Chunker
def semantic_doc_splitter(docs):
  embedding_model=GeminiEmbeddings(api_key=GOOGLE_API_KEY)
  splitter=SemanticChunker(embedding_model)


  chunks=splitter.split_text(docs)
  return chunks


#Embedding

#NVIDIA Embedding
def nv_embeddings():
   embeddings = NVIDIAEmbeddings(nvidia_api_key=NVIDIA_API_KEY)
   return embeddings


#Gemini Embedding
class GeminiEmbeddings(Embeddings):
  def __init__(self, model_name="models/text-embedding-004", api_key=None):
    self.model_name=model_name
    genai.configure(api_key=api_key)

  def embed_documents(self, docs):
    embeddings=[]
    for doc in docs:
      response = genai.embed_content(
            model="models/text-embedding-004",
            content=doc,
            task_type="RETRIEVAL_DOCUMENT" # Important for retrieval tasks
        )

      embeddings.append(response['embedding'])
    return embeddings
  
  def embed_query(self, query):
    response=genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="RETRIEVAL_DOCUMENT" # Important for retrieval tasks

    )

    embedded_query=response["embedding"]
    return embedded_query


#Vector Store and Retriever

#FAISS Vector Store
def faiss_db(embeddings, docs, query):
   db = FAISS.from_documents(docs, embeddings)
   retriever= db.as_retriever(search_kwargs={"k": 45})

   retriever.invoke(query)


   
#Pinecone Vector Store
def pinecone_db(documents, embedding_model, query):
   #initializing pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Delete the index
    pc.delete_index("langchain-demo")

    index_name = "langchain-demo"

    # First, check if the index already exists. If it doesn't, create a new one.
    if index_name not in pc.list_indexes().names():
        # print("Creating index")
        pc.create_index(name=index_name,
                        metric="cosine",

                        dimension=768 or 4096,#should match the dimension of embeddings created by the embedding model

                        spec=ServerlessSpec(
                            cloud="aws",
                            region="us-east-1"
                            ),
        )
        # print(pc.describe_index(index_name))
    vectorstore = PineconeVectorStore.from_documents(documents, embedding_model, index_name=index_name)
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})
    retriever.invoke(query)


#Optimization

#Optimization 1: Re-ranking and contextual compression of retreived docs acc to relevance scores
def rerank_docs(retriever, query):
    reranker = NVIDIARerank(nvidia_api_key=NVIDIA_API_KEY)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=retriever
    )

    reranked_chunks = compression_retriever.get_relevant_documents(query)
    return reranked_chunks


def generate_prompt(retriever, query):
   prompt=PromptTemplate(template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,

    input_variables = ['context', 'question'])
   
   docs= retriever.invoke(query)
   context="\n\n".join([doc.page_content for doc in docs])

   final_prompt=prompt.invoke({"context": context, "question": query})