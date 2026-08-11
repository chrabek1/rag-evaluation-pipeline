import asyncio
from pathlib import Path

from app.clients.embedding_client import EmbeddingClient
from app.core.config import settings
from app.db.database import create_pool
from app.db.schema import initialize_schema
from app.loaders.csv_chunk_loader import CsvChunkLoader
from app.repositories.chunk_repository import ChunkRepository
from app.services.indexing_service import IndexingService

async def main() -> None:
    loader = CsvChunkLoader()
    chunks = loader.load(Path("/data/dane.csv"))
    
    await initialize_schema(settings.database_url)
    
    pool = await create_pool(settings.database_url)
    embedding_client = EmbeddingClient(settings.embedding_service_url)
    repository = ChunkRepository(pool)
    
    try:
        indexing_service = IndexingService(
            embedding_client=embedding_client,
            chunk_repository=repository,
        )
        
        await indexing_service.index(chunks)
        
        print(f"Indexed {len(chunks)} chunks.")
        
    finally:
        await embedding_client.close()
        await pool.close()
        
if __name__ == "__main__":
    asyncio.run(main())