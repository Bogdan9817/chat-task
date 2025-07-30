from langchain_core.documents import Document
from fastapi import UploadFile

from store.knowledge_base import KnowledgeBase
from store.knowledge_graph import KnowledgeGraph
from utils.convert import convert_files_to_document


class Store:
    def __init__(self, kb: KnowledgeBase, kg: KnowledgeGraph):
        self.kb = kb
        self.kg = kg

    async def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("No documents provided to add.")
        self.kb.add_texts(documents)
        await self.kg.add_documents(documents)

    async def upload_files(self, files: list[UploadFile]) -> None:
        docs = convert_files_to_document(files)
        await self.add_documents(docs)
