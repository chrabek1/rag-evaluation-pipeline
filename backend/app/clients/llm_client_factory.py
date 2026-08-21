from app.clients.gemini_client import GeminiClient
from app.clients.llm_client import LLMClient
from app.clients.ollama_client import OllamaClient


def create_llm_client(
    provider: str,
    model: str,
    temperature: float,
    gemini_api_key: str | None = None,
    ollama_base_url: str | None = None,
) -> LLMClient:
    normalized_provider = provider.strip().lower()

    if normalized_provider == "gemini":
        if gemini_api_key is None:
            raise ValueError(
                "gemini_api_key is required for Gemini"
            )

        return GeminiClient(
            api_key=gemini_api_key,
            model=model,
            temperature=temperature,
        )

    if normalized_provider == "ollama":
        if ollama_base_url is None:
            raise ValueError("ollama_base_url is required for Ollama")

        return OllamaClient(
            base_url=ollama_base_url,
            model=model,
            temperature=temperature,
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider}"
    )