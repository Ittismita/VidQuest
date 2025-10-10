import streamlit as st
from config import GOOGLE_API_KEY, NVIDIA_API_KEY
from loaders import fetch_transcript
from chunkers import text_chunker, semantic_chunker
from embeddings import get_embedding_model
from vectorstores import get_vectorstore
from retrievers import build_retriever
from chains import build_chain
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

load_dotenv()

# Initialize session state---------------------------------------------------------------------------------------------- 
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'pipeline_built' not in st.session_state:
    st.session_state.pipeline_built = False
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore=[]



st.title("🎞️ YouTube Transcript RAG App")

# fetch videoid -> transcript from url----------------------------------------------------------------------------------
video_url = st.text_input("Enter YouTube Video URL:")
def get_video_id(url):
    if "v=" in url:
        vid_id=url.split("v=")[-1].split("&")[0]
        return vid_id
    elif "youtu.be/" in url:
        vid_id=url.split("youtu.be/")[-1].split("?")[0]
        return vid_id
    return None

video_id=get_video_id(video_url)
print(video_id)

if video_id and st.session_state.transcript is None:
    with st.spinner("Fetching transcript..."): 
        try:
            st.session_state.transcript = fetch_transcript(video_id=video_id)
            # if st.session_state.transcript:
                # st.success("Transcript fetched successfully!")
                # st.write(st.session_state.transcript[:200] + "...")
        except Exception as e:
            st.error("Transcript not available for this video")

if st.session_state.transcript:
    st.success("Transcript fetched successfully!")

# Pipeline configuration options----------------------------------------------------------------------------------------
# if st.session_state.transcript:
chunker_choice = st.selectbox("Select Chunker", ["text", "semantic"])
embedding_choice = st.selectbox("Select Embedding Model", ["gemini", "nvidia"])# for vector store
vectorstore_choice = st.selectbox("Select Vector Store", ["faiss", "pinecone"])
retriever_choice = st.selectbox("Retriever Type", ["basic", "rerank"])

llm_choice = st.selectbox("Select LLM", ["gemini", "nvidia"])


if st.button("Build Pipeline"):
    with st.spinner("Building Pipeline..."): 
        # divide transcript into chunks - text based or semantic based
        if chunker_choice == "text":
            chunks = text_chunker(st.session_state.transcript)#RecursiveCharacterTextSplitter
        else:
            chunks = semantic_chunker(st.session_state.transcript)#HuggingFaceEmbeddings + SemanticChunker

        #convert the chunks into document objects
        documents = [Document(page_content=c) for c in chunks]
        # print(f"Number of chunks created: {len(documents)}")

        # embedding model - gemini or nvidia -> vector store - faiss or pinecone -> retriever - basic or rerank(optimized)
        try:
            embedding_model = get_embedding_model(embedding_choice, GOOGLE_API_KEY, NVIDIA_API_KEY)
            st.session_state.vectorstore=get_vectorstore(vectorstore_choice, documents, embedding_model, embedding_choice)

            st.session_state.pipeline_built = True
            # st.success("Pipeline built successfully!")

        except Exception as e:
            st.error(f"Error building pipeline: {str(e)}")
            st.session_state.pipeline_built = False

if st.session_state.pipeline_built:
    st.success("Pipeline built successfully!")

# query the pipeline if built successfully---------------------------------------------------------------------------------
if hasattr(st.session_state, 'pipeline_built') and st.session_state.pipeline_built: 
    with st.form(key='my_form'):
        query = st.text_input("Ask a question about the video:")
        submit_button = st.form_submit_button(label='Generate')
      
    if submit_button:
        with st.spinner("Generating Response..."): 
            try:
                retriever = build_retriever(st.session_state.vectorstore, retriever_choice, NVIDIA_API_KEY)
                chain = build_chain(llm_choice, retriever, GOOGLE_API_KEY, NVIDIA_API_KEY)
                result=chain.invoke(query)
                # result = chain.invoke({"query": query})
                st.write("**Answer:**", result)
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                
                



 
 
 
# if vectorstore_choice == "faiss":
        # vectorstore = build_faiss(documents, embedding_model)
        # st.write("Number of docs in vectorstore:", vectorstore.index.ntotal)
# else:
        # vectorstore = build_pinecone(documents, embedding_model, INDEX_NAME, PINECONE_API_KEY)

#to do:
#deal with semantic chunking
#mutli turn conv
#ragas
#chat history store
