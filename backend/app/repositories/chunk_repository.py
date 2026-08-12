import logging

import asyncpg

from app.models.embedded_chunk import EmbeddedChunk


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