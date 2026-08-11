from collections import defaultdict

from app.clients.embedding_client import EmbeddingClient
from app.models.chunk import Chunk
from app.models.embedded_chunk import EmbeddedChunk
from app.repositories.chunk_repository import ChunkRepository


class IndexingService:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        chunk_repository: ChunkRepository,
        batch_size: int = 16,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        
        self._embedding_client = embedding_client
        self._chunk_repository = chunk_repository
        self._batch_size = batch_size
        
    async def index(self, chunks: list[Chunk]) -> None:
        document_counters: dict[str, int] = defaultdict(int)
        
        for batch_start in range(0, len(chunks), self._batch_size):
            batch = chunks[batch_start : batch_start + self._batch_size]
        
            texts = [chunk.content for chunk in batch]
            embeddings = await self._embedding_client.embed(texts)
            
            if len(embeddings) != len(batch):
                raise ValueError("Number of embeddings does not match number of chunks")
            
            embedded_chunks: list[EmbeddedChunk] = []
            
            for chunk, embedding in zip(batch, embeddings):
                document_counters[chunk.filename] += 1
                
                chunk_id = (
                    f"{chunk.filename}_"
                    f"{document_counters[chunk.filename]:04d}"
                )
                
                embedded_chunks.append(
                    EmbeddedChunk(
                        chunk_id=chunk_id,
                        chunk=chunk,
                        embedding=embedding,
                    )
                )
                
            await self._chunk_repository.add_many(embedded_chunks)