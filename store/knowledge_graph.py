from typing import Any

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain


class KnowledgeGraph:
    def __init__(self, api_key: str, url: str, username: str, password: str) -> None:
        self.graph = Neo4jGraph(
            url=url, username=username, password=password, enhanced_schema=True
        )
        self.graph.refresh_schema()
        self.llm = ChatOpenAI(temperature=0, api_key=api_key)
        self.llm_transformer = LLMGraphTransformer(llm=self.llm)
        self.chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
            allow_dangerous_requests=True,
        )

    async def add_documents(self, documents: list[Document]) -> None:
        graph_documents = await self.prepare_graph_documents(documents=documents)
        self.graph.add_graph_documents(graph_documents=graph_documents)
        self.graph.refresh_schema()

    async def prepare_graph_documents(self, documents: list[Document]) -> list[Any]:
        return await self.llm_transformer.aconvert_to_graph_documents(documents)

    def graph_search(self, search_text: str) -> list[dict]:
        return self.chain.invoke({"query": search_text})
