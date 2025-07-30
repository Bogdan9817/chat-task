import os
import asyncio
from dotenv import load_dotenv

from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI


from store.graph.knowledge_graph import KnowledgeGraph
from store.knowledge_base import KnowledgeBase
from chat_session.prompt import system_prompt
from chat_session.tools import Tools
from chat_session.custom_callback import CustomAsyncCallBackHandler


load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")


class Agent:
    def __init__(self, kb: KnowledgeBase, kg: KnowledgeGraph, **kwargs):
        tools = Tools(kb, kg)
        client = ChatOpenAI(api_key=API_KEY, model=kwargs.get("model"), streaming=True)
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.model_cfg = kwargs
        self._agent = initialize_agent(
            tools._tools,
            client,
            AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            agent_kwargs={
                "system_message": system_prompt,
            },
            verbose=True,
            streaming=True,
        )

    async def run_call(
        self, query: str, callback_handler: AsyncIteratorCallbackHandler
    ):
        config = {"callbacks": [callback_handler]}
        return await self._agent.ainvoke(input=query, config=config)

    async def create_gen(self, query: str):
        callback_handler = CustomAsyncCallBackHandler()
        task = asyncio.create_task(self.run_call(query, callback_handler))
        async for token in callback_handler.aiter():
            yield token
        await task
