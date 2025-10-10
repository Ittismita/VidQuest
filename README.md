# 🎞️ YouTube Transcript RAG App

An end-to-end Retrieval-Augmented Generation (RAG) pipeline that allows users to interact with YouTube videos through natural language queries.
By simply entering a YouTube video link, users can chat with the content — asking questions, summarizing sections, or clarifying concepts — all powered by configurable AI components.

---

## 🚀 Features

•⁠  ⁠Dynamic Transcript Extraction

  * Automatically extracts the video ID and fetches the transcript using the YouTube Transcript API.
  * Notifies the user when a transcript is unavailable.

•⁠  ⁠Flexible Configuration Options
  Choose from multiple RAG pipeline components before launching your chatbot:

  * Chunking: ⁠ Text ⁠ or ⁠ Semantic ⁠ splitter
  * Embeddings: ⁠ Gemini ⁠ or ⁠ NVIDIA ⁠
  * Vector Store: ⁠ FAISS ⁠ (local) or ⁠ Pinecone ⁠ (cloud-based)
  * Retriever: ⁠ Basic ⁠ or ⁠ Re-rank ⁠
  * LLM: ⁠ Gemini ⁠ or ⁠ NVIDIA ⁠

•⁠  ⁠Interactive Chat Interface

  * Once the pipeline is built successfully, users are prompted to start asking questions about the video.
  * The model retrieves relevant chunks from the transcript and generates contextual answers.

---

## 🧠 Workflow Overview

1.⁠ ⁠Input: User provides a YouTube video link.
2.⁠ ⁠Transcript Extraction: The transcript is fetched using the YouTube Transcript API.
3.⁠ ⁠Chunking: The text is split using either text-based or semantic chunking.
4.⁠ ⁠Embedding Generation: Chosen model (Gemini or NVIDIA) converts chunks into vector embeddings.
5.⁠ ⁠Vector Storage: Embeddings are stored in FAISS or Pinecone.
6.⁠ ⁠Retrieval: A retriever (basic or reranked) fetches context-relevant chunks based on user queries.
7.⁠ ⁠Response Generation: The selected LLM (Gemini or NVIDIA) generates answers grounded in retrieved context.

---

## 🛠️ Tech Stack

•⁠  ⁠Language: Python
•⁠  ⁠Libraries & Frameworks:

  * ⁠ LangChain (for RAG orchestration)
  * ⁠ FAISS ⁠, ⁠ Pinecone ⁠ (for vector storage)
  * ⁠ YouTube Transcript API ⁠ (for transcript extraction)
  * ⁠ Streamlit ⁠ (for UI and interactivity)
  * ⁠ Gemini ⁠, ⁠ NVIDIA ⁠ APIs (for embeddings and LLMs)

---

## ⚙️ Configuration Options (User Interface)

| Component           | Options          | Description                         |
| ------------------- | ---------------- | ----------------------------------- |
| Chunker         | Text / Semantic  | Choose method for text segmentation |
| Embedding Model | Gemini / NVIDIA  | Select embedding generator          |
| Vector Store    | FAISS / Pinecone | Choose local or cloud storage       |
| Retriever Type  | Basic / Re-rank  | Retrieval strategy                  |
| LLM             | Gemini / NVIDIA  | Response generation model           |

---

## 🧩 Example Use Case

	⁠User Input:
	⁠“Summarize the main points of this video: [YouTube link]”

	⁠Bot Output:
	⁠“The video discusses the basics of reinforcement learning — including the concept of agents, environments, rewards, and policy optimization…”

---

## 📦 Future Enhancements

•⁠  ⁠Multi-language transcript support
•⁠  ⁠Automatic video summarization mode
•⁠  ⁠Caching mechanism for faster repeated queries
•⁠  ⁠Enhanced UI with conversation history and export options

---
