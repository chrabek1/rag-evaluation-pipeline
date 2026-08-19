import argparse
import asyncio
import logging
from pathlib import Path

from app.clients.embedding_client import EmbeddingClient
from app.core.config import settings
from app.core.logging import configure_logging
from app.db.database import create_pool
from app.evaluation.retrieval_evaluation_pipeline import RetrievalEvaluationPipeline
from app.evaluation.retrieval_evaluator import RetrievalEvaluator
from app.evaluation.retrieval_metrics_aggregator import RetrievalMetricsAggregator
from app.evaluation.retrieval_result_writer import RetrievalResultWriter
from app.loaders.golden_dataset_loader import GoldenDatasetLoader
from app.repositories.chunk_repository import ChunkRepository
from app.services.retrieval_service import RetrievalService


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run retrieval evaluation against the golden dataset"),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks retrieved for every question.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/retrieval_evaluation.json"
        ),
        help="Path to the JSON result file."
    )
    
    return parser.parse_args()


async def run_evaluation(
    top_k: int,
    output_path: Path,
) -> None:
    dataset = GoldenDatasetLoader().load(
        settings.golden_dataset_path
    )
    
    logger.info(
        "Loaded %d golden dataset records",
        len(dataset.records),
    )
    
    embedding_client = EmbeddingClient(
        settings.embedding_service_url
    )
    
    try:
        model_info = await embedding_client.get_info()
        pool = await create_pool(settings.database_url)
        
        try:
            repository = ChunkRepository(pool)
            retrieval_service = RetrievalService(
                embedding_client=embedding_client,
                chunk_repository=repository,
            )
            
            pipeline = RetrievalEvaluationPipeline(
                retrieval_service=retrieval_service,
                retrieval_evaluator=RetrievalEvaluator(),
                metrics_aggregator=RetrievalMetricsAggregator(),
            )
            
            result = await pipeline.evaluate(
                dataset=dataset,
                top_k=top_k,
            )
            
            RetrievalResultWriter().write(
                result=result,
                output_path=output_path,
                embedding_model=model_info.model,
                embedding_dimension=model_info.embedding_dimension,
            )
            
            logger.info(
                "Evaluation completed for %d queries",
                result.summary.query_count,
            )
            logger.info(
                "Results saved to %s",
                output_path,
            )
        finally:
            await pool.close()
    finally:
        await embedding_client.close()
        
        
async def main() -> None:
    configure_logging()
    args = parse_args()
    
    logger.info(
        "Starting retrieval evaluation with top_k=%d",
        args.top_k,
    )
    
    try:
        await run_evaluation(
            top_k=args.top_k,
            output_path=args.output,
        )
    except Exception:
        logger.exception("Retrieval evaluation failed")
        raise
    

if __name__ == "__main__":
    asyncio.run(main())
