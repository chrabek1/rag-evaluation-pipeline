import json
from pathlib import Path
from unittest.mock import AsyncMock

import asyncpg
import pytest

from app.evaluation.generation_metrics_aggregator import (
    GenerationMetricsAggregator,
)
from app.evaluation.rag_evaluation_pipeline import (
    RAGEvaluationPipeline,
)
from app.evaluation.rag_result_writer import RAGResultWriter
from app.evaluation.retrieval_evaluation_pipeline import (
    RetrievalEvaluationPipeline,
)
from app.evaluation.retrieval_evaluator import RetrievalEvaluator
from app.evaluation.retrieval_metrics_aggregator import (
    RetrievalMetricsAggregator,
)
from app.models.chunk import Chunk
from app.models.embedded_chunk import EmbeddedChunk
from app.models.generation_metrics_result import (
    GenerationMetricsResult,
)
from app.models.golden_dataset import (
    ChunkEvidenceIntervals,
    EvidenceInterval,
    GoldenDataset,
    GoldenDatasetMetadata,
    GoldenDatasetRecord,
    GoldenEvidence,
    GoldenRelevantChunk,
)
from app.models.llm_response import LLMResponse
from app.repositories.chunk_repository import ChunkRepository
from app.services.generation_service import GenerationService
from app.services.retrieval_service import RetrievalService


pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

EMBEDDING_DIMENSION = 1024


async def test_rag_evaluation_runs_complete_pipeline_and_writes_json(
    db_pool: asyncpg.Pool,
    tmp_path: Path,
) -> None:
    relevant_embedding = [1.0, 0.0] + [0.0] * 1022
    repository = ChunkRepository(db_pool)
    await repository.add_many(
        [
            EmbeddedChunk(
                chunk_id="document.pdf_0001",
                chunk=Chunk(
                    filename="document.pdf",
                    content="Paris is the capital of France.",
                ),
                embedding=relevant_embedding,
            ),
            EmbeddedChunk(
                chunk_id="document.pdf_0002",
                chunk=Chunk(
                    filename="document.pdf",
                    content="Unrelated context.",
                ),
                embedding=[0.0, 1.0] + [0.0] * 1022,
            ),
        ]
    )

    embedding_client = AsyncMock()
    embedding_client.embed.return_value = [
        relevant_embedding
    ]

    llm_client = AsyncMock()
    llm_client.generate.return_value = LLMResponse(
        text="Paris is the capital of France.",
        model="test-generation-model",
        input_tokens=20,
        output_tokens=7,
    )

    generation_evaluator = AsyncMock()
    generation_evaluator.evaluate.return_value = (
        GenerationMetricsResult(
            faithfulness=1.0,
            answer_relevancy=0.9,
        )
    )

    dataset = GoldenDataset(
        metadata=GoldenDatasetMetadata(
            schema_version=1,
            evidence_interval_gap_tolerance=3,
        ),
        records=(
            GoldenDatasetRecord(
                query_id="query-1",
                question="What is the capital of France?",
                expected_answer="Paris.",
                evidence=(
                    GoldenEvidence(
                        text=(
                            "Paris is the capital of France."
                        ),
                        normalized_length=31,
                    ),
                ),
                relevant_chunks=(
                    GoldenRelevantChunk(
                        chunk_id="document.pdf_0001",
                        evidence_coverage=1.0,
                        evidence_intervals=(
                            ChunkEvidenceIntervals(
                                evidence_index=0,
                                intervals=(
                                    EvidenceInterval(
                                        start=0,
                                        end=31,
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    retrieval_pipeline = RetrievalEvaluationPipeline(
        retrieval_service=RetrievalService(
            embedding_client=embedding_client,
            chunk_repository=repository,
        ),
        retrieval_evaluator=RetrievalEvaluator(),
        metrics_aggregator=RetrievalMetricsAggregator(),
    )
    pipeline = RAGEvaluationPipeline(
        retrieval_pipeline=retrieval_pipeline,
        generation_service=GenerationService(
            llm_client=llm_client,
        ),
        generation_evaluator=generation_evaluator,
        generation_metrics_aggregator=(
            GenerationMetricsAggregator()
        ),
    )

    result = await pipeline.evaluate(
        dataset=dataset,
        top_k=1,
    )
    output_path = tmp_path / "rag.json"
    RAGResultWriter().write(
        result=result,
        output_path=output_path,
        embedding_model="test-embedding-model",
        embedding_dimension=EMBEDDING_DIMENSION,
        generation_provider="test",
        generation_model="test-generation-model",
        evaluation_provider="test",
        evaluation_model="test-evaluation-model",
    )

    payload = json.loads(
        output_path.read_text(encoding="utf-8")
    )
    query_result = result.query_results[0]

    assert query_result.retrieval.retrieved_chunks[0].chunk_id == (
        "document.pdf_0001"
    )
    assert query_result.retrieval.metrics.precision_at_k == 1.0
    assert query_result.retrieval.metrics.recall_at_k == 1.0
    assert query_result.generation.answer == (
        "Paris is the capital of France."
    )
    assert query_result.generation_metrics.faithfulness == 1.0
    assert result.retrieval_summary.query_count == 1
    assert result.generation_summary.query_count == 1
    assert result.generation_summary.total_input_tokens == 20
    assert result.generation_summary.total_output_tokens == 7

    embedding_client.embed.assert_awaited_once_with(
        ["What is the capital of France?"]
    )
    generation_evaluator.evaluate.assert_awaited_once_with(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        contexts=["Paris is the capital of France."],
    )

    assert payload["summary"]["retrieval"]["query_count"] == 1
    assert payload["summary"]["generation"]["query_count"] == 1
    assert payload["queries"][0]["generation"]["answer"] == (
        "Paris is the capital of France."
    )
