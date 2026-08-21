import argparse
import asyncio
import logging
from pathlib import Path

from app.clients.embedding_client import EmbeddingClient
from app.clients.llm_client import LLMClient
from app.clients.llm_client_factory import create_llm_client
from app.core.config import settings
from app.core.logging import configure_logging
from app.db.database import create_pool
from app.evaluation.generation_metrics_aggregator import (
    GenerationMetricsAggregator,
)
from app.evaluation.rag_evaluation_pipeline import (
    RAGEvaluationPipeline,
)
from app.evaluation.rag_result_writer import RAGResultWriter
from app.evaluation.ragas_evaluator_factory import (
    create_ragas_evaluator,
)
from app.evaluation.retrieval_evaluation_pipeline import (
    RetrievalEvaluationPipeline,
)
from app.evaluation.retrieval_evaluator import (
    RetrievalEvaluator,
)
from app.evaluation.retrieval_metrics_aggregator import (
    RetrievalMetricsAggregator,
)
from app.loaders.golden_dataset_loader import (
    GoldenDatasetLoader,
)
from app.repositories.chunk_repository import ChunkRepository
from app.services.generation_service import GenerationService
from app.services.retrieval_service import RetrievalService


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run retrieval and generation evaluation "
            "against the golden dataset."
        ),
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
        default=Path("results/rag_evaluation.json"),
        help="Path to the JSON result file.",
    )

    return parser.parse_args()


def get_gemini_api_key() -> str | None:
    if settings.gemini_api_key is None:
        return None

    return settings.gemini_api_key.get_secret_value()


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
    generation_client: LLMClient | None = None

    try:
        gemini_api_key = get_gemini_api_key()

        generation_client = create_llm_client(
            provider=settings.generation_provider,
            model=settings.generation_model,
            temperature=settings.generation_temperature,
            gemini_api_key=gemini_api_key,
            ollama_base_url=settings.ollama_base_url,
        )

        generation_evaluator = create_ragas_evaluator(
            provider=settings.evaluation_provider,
            model=settings.evaluation_model,
            embedding_client=embedding_client,
            temperature=settings.evaluation_temperature,
            gemini_api_key=gemini_api_key,
            ollama_base_url=settings.ollama_base_url,
        )

        model_info = await embedding_client.get_info()
        pool = await create_pool(settings.database_url)

        try:
            retrieval_service = RetrievalService(
                embedding_client=embedding_client,
                chunk_repository=ChunkRepository(pool),
            )

            retrieval_pipeline = RetrievalEvaluationPipeline(
                retrieval_service=retrieval_service,
                retrieval_evaluator=RetrievalEvaluator(),
                metrics_aggregator=(
                    RetrievalMetricsAggregator()
                ),
            )

            pipeline = RAGEvaluationPipeline(
                retrieval_pipeline=retrieval_pipeline,
                generation_service=GenerationService(
                    llm_client=generation_client,
                ),
                generation_evaluator=generation_evaluator,
                generation_metrics_aggregator=(
                    GenerationMetricsAggregator()
                ),
            )

            result = await pipeline.evaluate(
                dataset=dataset,
                top_k=top_k,
            )

            RAGResultWriter().write(
                result=result,
                output_path=output_path,
                embedding_model=model_info.model,
                embedding_dimension=(
                    model_info.embedding_dimension
                ),
                generation_provider=(
                    settings.generation_provider
                ),
                generation_model=settings.generation_model,
                evaluation_provider=(
                    settings.evaluation_provider
                ),
                evaluation_model=settings.evaluation_model,
            )

            logger.info(
                "Evaluation completed for %d queries",
                result.generation_summary.query_count,
            )
            logger.info(
                "Results saved to %s",
                output_path,
            )
        finally:
            await pool.close()
    finally:
        try:
            if generation_client is not None:
                await generation_client.close()
        finally:
            await embedding_client.close()


async def main() -> None:
    configure_logging()
    args = parse_args()

    logger.info(
        "Starting RAG evaluation with top_k=%d",
        args.top_k,
    )

    try:
        await run_evaluation(
            top_k=args.top_k,
            output_path=args.output,
        )
    except Exception:
        logger.exception("RAG evaluation failed")
        raise


if __name__ == "__main__":
    asyncio.run(main())