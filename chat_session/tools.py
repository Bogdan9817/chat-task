from langchain.agents import Tool

from store.knowledge_base import KnowledgeBase
from store.graph import knowledge_graph


class Tools:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        knowledge_graph: knowledge_graph.KnowledgeGraph,
    ):
        self._tools = [
            Tool(
                name="knowledge_base_search",
                func=knowledge_base.search,
                description="""
                    Searches the FAISS knowledge base for documents similar to the query.
                    Returns the most relevant documents based on semantic similarity.
                    Useful for finding information related to a specific topic or question.
                """,
            ),
            Tool(
                name="knowledge_graph_search",
                func=knowledge_graph.graph_search,
                description="""
                    Performs semantic and relational search over the knowledge graph using natural language queries.
                    Best used for exploring entities, relationships, connections, and structured facts.
                    Returns relevant nodes, relationships, and facts from the graph based on the query context.
                    Useful when the question involves relationships, hierarchies, or connections between concepts.
                """,
            ),
        ]
