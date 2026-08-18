import asyncpg


async def initialize_schema(
    database_url: str,
    embedding_dimension: int,
) -> None:
    if embedding_dimension <= 0:
        raise ValueError("embedding_dimension must be greater than 0")
    
    connection = await asyncpg.connect(database_url)
    
    try:
        await connection.execute(
            f"""
            CREATE EXTENSION IF NOT EXISTS vector;
            
            CREATE TABLE IF NOT EXISTS chunks (
                id BIGSERIAL PRIMARY KEY,
                chunk_id TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR({embedding_dimension}) NOT NULL
            );
            """
        )
        
        database_dimension = await connection.fetchval(
            """
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'chunks'::regclass
                AND attname = 'embedding'
                AND NOT attisdropped
            """
        )
        
        if database_dimension is None:
            raise RuntimeError(
                "Could not determine the dimension of the embedding column"
            )
        
        if database_dimension != embedding_dimension:
            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"model produces {embedding_dimension}-dimensional vectors, "
                f"but the database expects {database_dimension}"
            )
    finally:
        await connection.close()