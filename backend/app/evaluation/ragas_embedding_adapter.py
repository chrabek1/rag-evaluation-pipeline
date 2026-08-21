from typing import Any

from ragas.embeddings.base import BaseRagasEmbedding

from app.clients.embedding_client import EmbeddingClient


class RagasEmbeddingAdapter(BaseRagasEmbedding):
    def __init__(
        self,
        embedding_client: EmbeddingClient,
    ) -> None:
        super().__init__()
        self._embedding_client = embedding_client

    def embed_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> list[float]:
        raise RuntimeError("Synchronous embedding is not supported")

    async def aembed_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> list[float]:
        if not text.strip():
            raise ValueError("text must not be empty")

        vectors = await self._embedding_client.embed([text])

        if len(vectors) != 1:
            raise ValueError("Embedding service must return one vector")

        return vectors[0]

    async def aembed_texts(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError("texts must not contain empty text")

        vectors = await self._embedding_client.embed(texts)

        if len(vectors) != len(texts):
            raise ValueError("Embedding count must match text count")

        return vectors