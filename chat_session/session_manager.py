import uuid

from fastapi import WebSocket, WebSocketDisconnect

from chat_session.session import Session
from store.knowledge_base import KnowledgeBase
from store.graph.knowledge_graph import KnowledgeGraph


class SessionManager:
    def __init__(self, kb: KnowledgeBase, kg: KnowledgeGraph):
        self.kb = kb
        self.kg = kg
        self.sessions: dict[str, Session] = {}

    def init_session(self, **kwargs) -> str:
        id = uuid.uuid4()
        session = Session(id, self.kb, self.kg, **kwargs)
        self.sessions[str(id)] = session
        return str(id)

    def _get_session(self, id: str) -> Session:
        if id in self.sessions:
            return self.sessions[id]
        new_session_id = self.init_session()
        return self.sessions[new_session_id]

    def disconnect(self, session_id: uuid.UUID) -> None:
        session = self._get_session(session_id)
        session.disconnect()
        del self.sessions[session.id]

    async def handle_chat(self, ws: WebSocket):
        await ws.accept()
        session_id = ws.query_params.get("session_id")
        session = self._get_session(session_id)
        try:
            while True:
                user_q = await ws.receive_text()
                async for chunk in session.agent.create_gen(user_q):
                    await ws.send_text(chunk)
                await ws.send_text("MESSAGE_END")
        except WebSocketDisconnect:
            print("Stopped")
            self.disconnect(session.id)
        except Exception as e:
            print(f"Error in session {session.id}: {e}")
