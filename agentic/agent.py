from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

from store.knowledge_graph import KnowledgeGraph
from store.knowledge_base import KnowledgeBase
from agentic.prompt import SystemPrompt
from agentic.tools import Tools


class Agent:
    def __init__(self, kb: KnowledgeBase, kg: KnowledgeGraph, api_key: str):
        tools = Tools(kb, kg)
        client = ChatOpenAI(api_key=api_key)
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self._agent = initialize_agent(
            tools._tools,
            client,
            AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            return_indeterminate_steps=True,
            verbose=True,
            system_prompt=SystemPrompt,
            handle_parsing_errors=True,
        )

    async def ask(self, user_question: str):
        return await self._agent.ainvoke(user_question)
