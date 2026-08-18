import logging

import httpx

from app.models.embedding_model_info import EmbeddingModelInfo


logger=logging.getLogger(__name__)

class EmbeddingClient:
    def __init__(
        self, 
        base_url: str,
        timeout: float = 60.0,
        ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
        )
        
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        
        logger.info("Requesting embeddings for %d texts", len(texts))
        
        response = await self._client.post(
            "/embed",
            json={"texts": texts},
        )
        response.raise_for_status()
        
        data = response.json()
        vectors = data["vectors"]
        
        logger.info("Received %d embeddings", len(vectors))
        
        return vectors
    
    async def get_info(self) -> EmbeddingModelInfo:
        logger.info("Requesting embedding model information")
        
        response = await self._client.get("/info")
        response.raise_for_status()
        
        model_info = EmbeddingModelInfo.model_validate(response.json())
        
        logger.info(
            "Embedding model: %s, dimension: %d",
            model_info.model,
            model_info.embedding_dimension,
        )
        
        return model_info
    
    async def close(self) -> None:
        await self._client.aclose()