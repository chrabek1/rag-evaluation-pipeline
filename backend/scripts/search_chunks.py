import argparse
import asyncio
import logging

from app.clients.embedding_client import EmbeddingClient
from app.core.config import settings
from app.core.logging import configure_logging
from app.db.database import create_pool
from app.repositories.chunk_repository import ChunkRepository
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve the most relevant chunks for a query",
    )
    parser.add_argument(
        "query",
        type=str,
        help="Query text used for retrieval.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve.",
    )
    
    return parser.parse_args()

async def run_search(
    query: str,
    top_k: int,
) -> None:
    pool = await create_pool(settings.database_url)
    embedding_client = EmbeddingClient(
        settings.embedding_service_url
    )
    
    try:
        repository = ChunkRepository(pool)
        retrieval_service = RetrievalService(
            embedding_client=embedding_client,
            chunk_repository=repository,
        )
        
        results = await retrieval_service.retrieve(
            query=query,
            top_k=top_k,
        )
        
        for rank, result in enumerate(results, start=1):
            print(
                f"\n#{rank}"
                f"\nchunk_id: {result.chunk_id}"
                f"\nfilename: {result.chunk.filename}"
                f"\nscore: {result.score:.4f}"
                f"\ncontent:\n{result.chunk.content}"
            )
    
    finally:
        await embedding_client.close()
        await pool.close()

async def main() -> None:
    configure_logging()
    
    args = parse_args()
    
    logger.info(
        "Starting retrieval for query with top_k=%d",
        args.top_k,
    )
    
    try:
        await run_search(
            query=args.query,
            top_k=args.top_k,
        )
    except Exception:
        logger.exception("Retrieval failed")
        raise
    
    logger.info("Retrieval completed successfully")
    
if __name__ == "__main__":
    asyncio.run(main())