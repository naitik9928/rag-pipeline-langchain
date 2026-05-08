# 🔧 RAG-Based Technical Document Q&A System

A Retrieval-Augmented Generation (RAG) system that answers questions about automotive manufacturing and predictive maintenance using LangChain, Chroma, Jina Embeddings, and Groq's LLM.

## 🚀 Features
- 📄 Recursive text splitting with 500 char chunks & 50 char overlap
- 🔍 Vector search using Jina Embeddings and Chroma DB
- 🤖 LLM-powered answers with Groq (Llama 3.1 8B)
- ⚡ Retrieval pipeline with RunnableParallel architecture

## 🛠️ Tech Stack
- LangChain - RAG orchestration
- Chroma - Vector database
- Jina Embeddings - Text embedding model
- ChatGroq - LLM inference (Llama 3.1 8B)
- RecursiveCharacterTextSplitter - Document chunking

## 🔄 Workflow
Technical Document → Text Splitting → Embeddings → Chroma Vector Store → Retriever (k=4) → Context + Question → LLM → Answer
