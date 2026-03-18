---
title: RAG Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# RAG Chatbot with HuggingFace Models

A ChatGPT-like RAG (Retrieval-Augmented Generation) chatbot built with FastAPI, HuggingFace Inference API, and FAISS vector store. Designed for free deployment on HuggingFace Spaces.

## Architecture

| Component | Technology |
|-----------|-----------|
| LLM | HuggingFace Inference API (Mistral-7B-Instruct) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| Backend | FastAPI with streaming responses |
| Frontend | Vanilla HTML/CSS/JS |

## Local Setup

```bash
# Clone the repo
git clone https://github.com/upujari-bitSI/LLM_Bot_W_RAG_API.git
cd LLM_Bot_W_RAG_API

# Create .env from example
cp .env.example .env
# Edit .env and add your HuggingFace API token

# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

Open http://localhost:7860 in your browser.

## Deploy to HuggingFace Spaces

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Docker** as the SDK
3. Connect this GitHub repo
4. Add `HUGGINGFACEHUB_API_TOKEN` as a Space Secret
5. Deploy — you'll get a persistent URL like `https://your-username-rag-chatbot.hf.space`

## Usage

1. Upload a PDF or text document using the upload button
2. Ask questions about your documents in the chat
3. The bot retrieves relevant context and generates answers using Mistral-7B
