import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, Response

from store.knowledge_base import KnowledgeBase
from store.knowledge_graph import KnowledgeGraph
from store.manager import Store
from load_static import txt_folder_to_documents
from agentic.agent import Agent

import uvicorn

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.getenv("OPENAI_API_KEY")
    docs = txt_folder_to_documents("static_data")
    print(f"Loaded {len(docs)} documents from static data.")
    kb = KnowledgeBase(api_key)
    kg = KnowledgeGraph(
        api_key,
        url=os.getenv("NEO4J_URL"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
    )
    store_manager = Store(kb, kg)

    await store_manager.add_documents(docs)
    agent = Agent(kb, kg, api_key)

    app.state.kb = kb
    app.state.kg = kg
    app.state.store_manager = store_manager
    app.state.agent = agent
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def read_root():
    return "OK"


@app.post("/upload_files")
async def upload_file(files: list[UploadFile]) -> Response:
    try:
        return await app.state.store_manager.upload_files(files)
    except Exception as e:
        return {"error": str(e)}


@app.post("/ask")
async def ask_agent(q: dict) -> Response:
    try:
        result = await app.state.agent.ask(q["question"])
        return result
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
