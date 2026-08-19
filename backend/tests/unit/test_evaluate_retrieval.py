from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.models.embedding_model_info import EmbeddingModelInfo
from scripts import evaluate_retrieval


@pytest.mark.asyncio
async def test_run_evaluation_builds_pipeline_and_writes_result() -> None:
    dataset = Mock()
    dataset.records = (Mock(),)

    loader = Mock()
    loader.load.return_value = dataset

    embedding_client = AsyncMock()
    embedding_client.get_info.return_value = (
        EmbeddingModelInfo(
            model="test-model",
            embedding_dimension=768,
        )
    )

    pool = AsyncMock()
    repository = Mock()
    retrieval_service = Mock()
    retrieval_evaluator = Mock()
    metrics_aggregator = Mock()

    evaluation_result = Mock()
    evaluation_result.summary.query_count = 1

    pipeline = AsyncMock()
    pipeline.evaluate.return_value = evaluation_result
    pipeline_factory = Mock(return_value=pipeline)

    writer = Mock()
    output_path = Path("results/test-result.json")

    with (
        patch.object(
            evaluate_retrieval,
            "GoldenDatasetLoader",
            return_value=loader,
        ),
        patch.object(
            evaluate_retrieval,
            "EmbeddingClient",
            return_value=embedding_client,
        ),
        patch.object(
            evaluate_retrieval,
            "create_pool",
            new=AsyncMock(return_value=pool),
        ) as create_pool_mock,
        patch.object(
            evaluate_retrieval,
            "ChunkRepository",
            return_value=repository,
        ),
        patch.object(
            evaluate_retrieval,
            "RetrievalService",
            return_value=retrieval_service,
        ),
        patch.object(
            evaluate_retrieval,
            "RetrievalEvaluator",
            return_value=retrieval_evaluator,
        ),
        patch.object(
            evaluate_retrieval,
            "RetrievalMetricsAggregator",
            return_value=metrics_aggregator,
        ),
        patch.object(
            evaluate_retrieval,
            "RetrievalEvaluationPipeline",
            new=pipeline_factory,
        ),
        patch.object(
            evaluate_retrieval,
            "RetrievalResultWriter",
            return_value=writer,
        ),
    ):
        await evaluate_retrieval.run_evaluation(
            top_k=5,
            output_path=output_path,
        )

    loader.load.assert_called_once_with(
        evaluate_retrieval.settings.golden_dataset_path
    )
    embedding_client.get_info.assert_awaited_once_with()
    create_pool_mock.assert_awaited_once_with(
        evaluate_retrieval.settings.database_url
    )

    pipeline_factory.assert_called_once_with(
        retrieval_service=retrieval_service,
        retrieval_evaluator=retrieval_evaluator,
        metrics_aggregator=metrics_aggregator,
    )

    pipeline.evaluate.assert_awaited_once_with(
        dataset=dataset,
        top_k=5,
    )

    writer.write.assert_called_once_with(
        result=evaluation_result,
        output_path=output_path,
        embedding_model="test-model",
        embedding_dimension=768,
    )

    embedding_client.close.assert_awaited_once_with()
    pool.close.assert_awaited_once_with()
