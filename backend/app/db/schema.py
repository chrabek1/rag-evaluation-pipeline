import asyncpg


async def initialize_schema(database_url: str) -> None:
    connection = await asyncpg.connect(database_url)
    
    try:
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
    finally:
        await connection.close()