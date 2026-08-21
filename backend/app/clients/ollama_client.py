import logging

import httpx

from app.models.llm_response import LLMResponse


logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        temperature: float = 0.0,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")

        if not model.strip():
            raise ValueError("model must not be empty")

        if temperature < 0:
            raise ValueError("temperature must not be negative")

        self._model = model
        self._temperature = temperature
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
        )

    @property
    def model(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
    ) -> LLMResponse:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        logger.info(
            "Requesting generation from Ollama model %s",
            self._model,
        )

        response = await self._client.post(
            "/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self._temperature,
                },
            },
        )
        response.raise_for_status()

        data = response.json()

        result = LLMResponse(
            text=data["response"],
            model = data.get("model", self._model),
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
        )

        logger.info(
            "Received generation from Ollama model %s",
            result.model,
        )

        return result

    async def close(self) -> None:
        await self._client.aclose()