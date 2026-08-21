from typing import Protocol

from app.models.llm_response import LLMResponse


class LLMClient(Protocol):
    @property
    def model(self) -> str:
        ...

    async def generate(
        self,
        prompt: str,
    ) -> LLMResponse:
        ...

    async def close(self) -> None:
        ...