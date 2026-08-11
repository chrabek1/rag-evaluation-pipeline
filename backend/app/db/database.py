import asyncpg
from pgvector.asyncpg import register_vector

async def create_pool(database_url: str) -> asyncpg.Pool:
    connection = await asyncpg.connect(database_url)
    
    try:
        await connection.execute(
            "CREATE EXTENSION IF NOT EXISTS vector"
        )
    finally:
        await connection.close()
        
    async def init_connection(connection: asyncpg.Connection) -> None:
        await register_vector(connection)
        
    return await asyncpg.create_pool(
        dsn=database_url,
        init=init_connection,
    )