import asyncio
import logging
from pathlib import Path

from app.clients.embedding_client import EmbeddingClient
from app.core.config import settings
from app.core.logging import configure_logging
from app.db.database import create_pool
from app.db.schema import initialize_schema
from app.loaders.csv_chunk_loader import CsvChunkLoader
from app.repositories.chunk_repository import ChunkRepository
from app.services.indexing_service import IndexingService


logger = logging.getLogger(__name__)

async def run_indexing() -> None:
    corpus_path = Path(settings.corpus_path)
    
    loader = CsvChunkLoader()
    chunks = loader.load(corpus_path)
    
    logger.info("Loaded %d chunks from %s", len(chunks), corpus_path)
    
    embedding_client = EmbeddingClient(settings.embedding_service_url)
    
    try:
        model_info = await embedding_client.get_info()
        
        await initialize_schema(
            settings.database_url,
            model_info.embedding_dimension,
        )
        
        pool = await create_pool(settings.database_url)
        
        try:
            repository = ChunkRepository(pool)
            indexing_service = IndexingService(
                embedding_client=embedding_client,
                chunk_repository=repository,
            )
            
            await indexing_service.index(chunks)
            logger.info("Indexed %d chunks", len(chunks))
        finally:
            await pool.close()
    finally:
        await embedding_client.close()

async def main() -> None:
    configure_logging()
    
    logger.info("Start indexing pipeline")
    
    try: 
        await run_indexing()
    except Exception:
        logger.exception("Indexing pipeline failed")
        raise
    
    logger.info("Indexing pipeline completed successfully")
    
if __name__ == "__main__":
    asyncio.run(main())