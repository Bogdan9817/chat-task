import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from store.knowledge_base import KnowledgeBase
from store.graph.knowledge_graph import KnowledgeGraph
from store.manager import Store
from chat_session.session_manager import SessionManager
from load_static import txt_folder_to_documents

import uvicorn

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.getenv("OPENAI_API_KEY")
    docs = txt_folder_to_documents("static_data")
    app.state.kb = KnowledgeBase(api_key)
    app.state.kg = KnowledgeGraph(
        api_key,
        url=os.getenv("NEO4J_URL"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
    )
    app.state.store_manager = Store(app.state.kb, app.state.kg)
    app.state.session_manager = SessionManager(app.state.kb, app.state.kg)
    await app.state.store_manager.add_documents(docs)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def read_root():
    return "OK"


@app.post("/upload_files")
async def upload_file(files: list[UploadFile]) -> Response:
    try:
        return await app.state.store_manager.upload_files(files)
    except Exception as e:
        return {"error": str(e)}


@app.post("/init_session")
async def initialize_agent(q: dict) -> Response:
    try:
        return app.state.session_manager.init_session(**q)
    except Exception as e:
        return {"error": str(e)}


@app.websocket("/stream")
async def chat(websocket: WebSocket) -> Response:
    try:
        await app.state.session_manager.handle_chat(websocket)
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
