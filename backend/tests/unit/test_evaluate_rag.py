from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.models.embedding_model_info import EmbeddingModelInfo
from scripts import evaluate_rag


@pytest.mark.asyncio
async def test_run_evaluation_builds_pipeline_and_writes_result(
) -> None:
    dataset = Mock()
    dataset.records = (Mock(),)

    loader = Mock()
    loader.load.return_value = dataset

    embedding_client = AsyncMock()
    embedding_client.get_info.return_value = (
        EmbeddingModelInfo(
            model="test-embedding-model",
            embedding_dimension=768,
        )
    )

    generation_client = AsyncMock()
    generation_evaluator = Mock()
    pool = AsyncMock()

    repository = Mock()
    retrieval_service = Mock()
    retrieval_evaluator = Mock()
    retrieval_aggregator = Mock()
    retrieval_pipeline = Mock()

    generation_service = Mock()
    generation_aggregator = Mock()

    evaluation_result = Mock()
    evaluation_result.generation_summary.query_count = 1

    pipeline = Mock()
    pipeline.evaluate = AsyncMock(
        return_value=evaluation_result
    )

    writer = Mock()
    output_path = Path("results/test-rag.json")

    with (
        patch.object(
            evaluate_rag,
            "GoldenDatasetLoader",
            return_value=loader,
        ),
        patch.object(
            evaluate_rag,
            "EmbeddingClient",
            return_value=embedding_client,
        ),
        patch.object(
            evaluate_rag,
            "get_gemini_api_key",
            return_value="test-key",
        ),
        patch.object(
            evaluate_rag,
            "create_llm_client",
            return_value=generation_client,
        ) as create_llm_client_mock,
        patch.object(
            evaluate_rag,
            "create_ragas_evaluator",
            return_value=generation_evaluator,
        ) as create_ragas_evaluator_mock,
        patch.object(
            evaluate_rag,
            "create_pool",
            new=AsyncMock(return_value=pool),
        ) as create_pool_mock,
        patch.object(
            evaluate_rag,
            "ChunkRepository",
            return_value=repository,
        ),
        patch.object(
            evaluate_rag,
            "RetrievalService",
            return_value=retrieval_service,
        ),
        patch.object(
            evaluate_rag,
            "RetrievalEvaluator",
            return_value=retrieval_evaluator,
        ),
        patch.object(
            evaluate_rag,
            "RetrievalMetricsAggregator",
            return_value=retrieval_aggregator,
        ),
        patch.object(
            evaluate_rag,
            "RetrievalEvaluationPipeline",
            return_value=retrieval_pipeline,
        ),
        patch.object(
            evaluate_rag,
            "GenerationService",
            return_value=generation_service,
        ),
        patch.object(
            evaluate_rag,
            "GenerationMetricsAggregator",
            return_value=generation_aggregator,
        ),
        patch.object(
            evaluate_rag,
            "RAGEvaluationPipeline",
            return_value=pipeline,
        ) as pipeline_factory,
        patch.object(
            evaluate_rag,
            "RAGResultWriter",
            return_value=writer,
        ),
    ):
        await evaluate_rag.run_evaluation(
            top_k=5,
            output_path=output_path,
        )

    loader.load.assert_called_once_with(
        evaluate_rag.settings.golden_dataset_path
    )

    create_llm_client_mock.assert_called_once_with(
        provider=evaluate_rag.settings.generation_provider,
        model=evaluate_rag.settings.generation_model,
        temperature=(
            evaluate_rag.settings.generation_temperature
        ),
        gemini_api_key="test-key",
        ollama_base_url=(
            evaluate_rag.settings.ollama_base_url
        ),
    )

    create_ragas_evaluator_mock.assert_called_once_with(
        provider=evaluate_rag.settings.evaluation_provider,
        model=evaluate_rag.settings.evaluation_model,
        embedding_client=embedding_client,
        temperature=(
            evaluate_rag.settings.evaluation_temperature
        ),
        gemini_api_key="test-key",
        ollama_base_url=(
            evaluate_rag.settings.ollama_base_url
        ),
    )

    embedding_client.get_info.assert_awaited_once_with()
    create_pool_mock.assert_awaited_once_with(
        evaluate_rag.settings.database_url
    )

    pipeline_factory.assert_called_once_with(
        retrieval_pipeline=retrieval_pipeline,
        generation_service=generation_service,
        generation_evaluator=generation_evaluator,
        generation_metrics_aggregator=(
            generation_aggregator
        ),
    )

    pipeline.evaluate.assert_awaited_once_with(
        dataset=dataset,
        top_k=5,
    )

    writer.write.assert_called_once_with(
        result=evaluation_result,
        output_path=output_path,
        embedding_model="test-embedding-model",
        embedding_dimension=768,
        generation_provider=(
            evaluate_rag.settings.generation_provider
        ),
        generation_model=(
            evaluate_rag.settings.generation_model
        ),
        evaluation_provider=(
            evaluate_rag.settings.evaluation_provider
        ),
        evaluation_model=(
            evaluate_rag.settings.evaluation_model
        ),
    )

    pool.close.assert_awaited_once_with()
    generation_client.close.assert_awaited_once_with()
    embedding_client.close.assert_awaited_once_with()