import uuid

from store.knowledge_base import KnowledgeBase
from store.graph.knowledge_graph import KnowledgeGraph
from chat_session.agent import Agent


class Session:
    def __init__(self, id: uuid.UUID, kb: KnowledgeBase, kg: KnowledgeGraph, **kwargs):
        self.id = id
        self.agent = Agent(kb, kg, **kwargs)
        self.messages = []
