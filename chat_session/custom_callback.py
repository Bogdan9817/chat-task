from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.schema import LLMResult

from utils.utilities import contains


class CustomAsyncCallBackHandler(AsyncIteratorCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[list[str], int] = {"used_tools": [], "used_tokens": 0}
        self.content: str = ""
        self.final_answer: bool = False

    async def on_chat_model_start(
        self,
        serialized,
        messages,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        return await super().on_chat_model_start(
            serialized,
            messages,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    async def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.content += token
        self.meta["used_tokens"] += 1
        if self.final_answer:
            if '"action_input": "' in self.content:
                if contains(token, ['"']):
                    self.queue.put_nowait(token.replace('"', ""))
                elif not contains(token, ["}", "```"]):
                    self.queue.put_nowait(token)
        elif "Final Answer" in self.content:
            self.final_answer = True
            self.content = ""

    async def on_llm_end(self, response: LLMResult, **kwargs: any) -> None:
        if self.final_answer:
            self.content = ""
            self.final_answer = False
            self.done.set()
        else:
            self.content = ""

    async def on_tool_start(
        self,
        serialized,
        input_str,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        inputs=None,
        **kwargs,
    ):
        self.meta["used_tools"].append(serialized["name"])
        return await super().on_tool_start(
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            inputs=inputs,
            **kwargs,
        )
