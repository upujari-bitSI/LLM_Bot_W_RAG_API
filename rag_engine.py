import asyncio
import os
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

VECTORSTORE_PATH = "vectorstore"


class RAGEngine:
    def __init__(self):
        self.hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
        self.model_id = os.getenv(
            "HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2"
        )
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cpu"},
        )

        self.client = InferenceClient(
            model=self.model_id,
            token=self.hf_token if self.hf_token else None,
        )

        self.vectorstore = None
        self._load_vectorstore()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

    def _load_vectorstore(self):
        index_path = Path(VECTORSTORE_PATH) / "index.faiss"
        if index_path.exists():
            self.vectorstore = FAISS.load_local(
                VECTORSTORE_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

    def has_vectorstore(self) -> bool:
        return self.vectorstore is not None

    def ingest_document(self, file_path: str) -> int:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)

        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)

        self.vectorstore.save_local(VECTORSTORE_PATH)
        return len(chunks)

    def _build_prompt(self, query: str, context: str) -> str:
        return (
            f"<s>[INST] You are a helpful assistant. Use the following context "
            f"to answer the question. If the context doesn't contain relevant "
            f"information, say so but still try to help.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query} [/INST]"
        )

    async def aquery(self, query: str) -> AsyncGenerator[str, None]:
        context = ""
        if self.vectorstore:
            docs = self.vectorstore.similarity_search(query, k=4)
            context = "\n\n".join(doc.page_content for doc in docs)

        prompt = self._build_prompt(query, context)

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.text_generation(
                    prompt,
                    max_new_tokens=512,
                    temperature=0.7,
                    repetition_penalty=1.1,
                ),
            )
            yield response
        except Exception as e:
            yield f"Error from model: {e}"
