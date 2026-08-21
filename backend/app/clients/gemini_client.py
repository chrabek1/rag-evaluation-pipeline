import logging
from google import genai
from google.genai import types

from app.models.llm_response import LLMResponse


logger = logging.getLogger(__name__)


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty")

        if not model.strip():
            raise ValueError("model must not be empty")

        if temperature < 0:
            raise ValueError("temperature must not be negative")

        self._model = model
        self._temperature = temperature
        self._client = genai.Client(
            api_key=api_key,
        ).aio

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
            "Requesting generation from Gemini model %s",
            self._model,
        )

        response = await self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self._temperature,
            ),
        )

        if response.text is None:
            raise ValueError("Gemini response does not contain text")

        usage = response.usage_metadata

        result = LLMResponse(
            text=response.text,
            model=self._model,
            input_tokens=(
                usage.prompt_token_count
                if usage is not None
                else None
            ),
            output_tokens=(
                usage.candidates_token_count
                if usage is not None
                else None
            ),
        )

        logger.info(
            "Received generation from Gemini model %s",
            self._model,
        )

        return result

    async def close(self) -> None:
        await self._client.aclose()