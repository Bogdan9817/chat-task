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
                    This tool performs semantic similarity search over a FAISS-powered vector knowledge base.
                    It finds and returns documents that are most similar in meaning to the user's query.

                    Use this tool when:
                    - The user is asking a **direct question** about a specific topic or concept.
                    - The user is looking for **facts**, **summaries**, **explanations**, or **descriptions**.
                    - You expect the answer to come from **text documents**, not from a structured graph.

                    Best for:
                    - Definitions and descriptions.
                    - Summarizing document content.
                    - Retrieving relevant document chunks.
                    - Topic-specific questions.

                    Examples:
                    - "What is LangChain?"
                    - "Tell me about FAISS and how it works."
                    - "Summarize what the documentation says about memory components."
                    - "Explain what agents are in the context of LLMs."
                """,
            ),
            Tool(
                name="knowledge_graph_search",
                func=knowledge_graph.graph_search,
                description="""
                    This tool performs semantic and relational search over a structured knowledge graph.
                    It explores entities, their relationships, hierarchies, and interconnections using graph traversal.

                    Use this tool when:
                    - The user's question involves relationships, connections, or hierarchies between concepts or entities.
                    - You need to identify how items are linked, dependent, or organized within a structured graph.

                    Best for:
                    - Finding direct and indirect relationships.
                    - Exploring dependencies and hierarchies.
                    - Navigating structured conceptual networks.

                    Examples:
                    - "How is LangChain related to FAISS?"
                    - "Which modules depend on vector databases?"
                    - "What tools are connected to OpenAI?"
                    - "Show the relationship between agents and memory in the architecture."

                    Note: Do not use this tool for retrieving unstructured document text or general-purpose summaries.
                """,
            ),
        ]
