import asyncpg

from app.models.embedded_chunk import EmbeddedChunk


class ChunkRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        
    async def create_schema(self) -> None:
        async with self._pool.acquire() as connection:
            await connection.execute(
                """
                CREATE EXTENSION IF NOT EXISTS vector;
                
                CREATE TABLE IF NOT EXISTS chunks (
                    id BIGSERIAL PRIMARY KEY,
                    chunk_id TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR(1024) NOT NULL
                );
                """
            )
    
    async def add_many(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        if not embedded_chunks:
            return
        
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
                """,
                records,
            )