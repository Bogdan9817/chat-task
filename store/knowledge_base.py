from typing import Iterable

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import faiss


class KnowledgeBase:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._vs = self._init_vs()
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_overlap=20, chunk_size=500, separators=["\n\n", "\n", ".", " ", ""]
        )
        self.retriever = self._vs.as_retriever(search_kwargs={"k": 5})

    def add_texts(self, stored_content: Iterable[Document]) -> None:
        splitted_text = self._text_splitter.split_documents(stored_content)
        self._vs.add_documents(splitted_text)

    def search(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def _init_vs(self) -> FAISS:
        index = faiss.IndexFlatIP(1536)
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", api_key=self.api_key
        )
        return FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
