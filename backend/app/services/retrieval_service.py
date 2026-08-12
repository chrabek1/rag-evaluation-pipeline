import logging

from app.clients.embedding_client import EmbeddingClient
from app.models.retrieved_chunk import RetrievedChunk
from app.repositories.chunk_repository import ChunkRepository


logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        chunk_repository: ChunkRepository,
    ) -> None:
        self._embedding_client = embedding_client
        self._chunk_repository = chunk_repository
        
    async def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("query must not be empty")
        
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        
        logger.info("Retrieving top %d chunks", top_k)
        
        embeddings = await self._embedding_client.embed([query])
        
        if len(embeddings) != 1:
            raise ValueError(
                "Expected exactly one embedding for query"
            )
            
        results = await self._chunk_repository.search(
            query_embedding=embeddings[0],
            top_k=top_k,
        )
        
        logger.info("Retrieved %d chunks", len(results))
        
        return results