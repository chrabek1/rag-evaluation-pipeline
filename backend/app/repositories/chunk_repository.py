import logging

import asyncpg

from app.models.chunk import Chunk
from app.models.embedded_chunk import EmbeddedChunk
from app.models.retrieved_chunk import RetrievedChunk


logger = logging.getLogger(__name__)

class ChunkRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
    
    async def add_many(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        if not embedded_chunks:
            return
        
        logger.info(
            "Writing %d chunks to database",
            len(embedded_chunks)
        )
        
        records = [
            (
                embedded_chunk.chunk_id,
                embedded_chunk.chunk.filename,
                embedded_chunk.chunk.content,
                embedded_chunk.embedding,
            )
            for embedded_chunk in embedded_chunks
        ]
        
        async with self._pool.acquire() as connection:
            await connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id,
                    filename,
                    content,
                    embedding
                )
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (chunk_id)
                DO UPDATE SET
                    filename = EXCLUDED.filename,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding
                """,
                records,
            )
            
    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        
        logger.info("Searching for top %d chunks", top_k)
        
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT
                    chunk_id,
                    filename,
                    content,
                    1 - (embedding <=> $1) AS score
                FROM chunks
                ORDER BY embedding <=> $1
                LIMIT $2
                """,
                query_embedding,
                top_k,
            )
        
        return [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                chunk=Chunk(
                    filename=row["filename"],
                    content=row["content"],
                ),
                score=float(row["score"]),
            )
            for row in rows
        ]