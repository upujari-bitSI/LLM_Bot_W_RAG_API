import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rag_engine import RAGEngine

load_dotenv()

app = FastAPI(title="RAG Chatbot with HuggingFace")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

rag_engine = RAGEngine()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    num_chunks = rag_engine.ingest_document(str(file_path))
    return {
        "filename": file.filename,
        "chunks": num_chunks,
        "message": f"Successfully ingested {file.filename} ({num_chunks} chunks)",
    }


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("message", "")

    if not query:
        return {"error": "No message provided"}

    async def generate():
        async for token in rag_engine.aquery(query):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/documents")
async def list_documents():
    files = [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]
    return {"documents": files, "has_vectorstore": rag_engine.has_vectorstore()}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
