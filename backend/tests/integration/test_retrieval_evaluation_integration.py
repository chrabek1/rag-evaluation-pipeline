import json
from pathlib import Path
from unittest.mock import AsyncMock

import asyncpg
import pytest

from app.evaluation.retrieval_evaluation_pipeline import (
    RetrievalEvaluationPipeline,
)
from app.evaluation.retrieval_evaluator import RetrievalEvaluator
from app.evaluation.retrieval_metrics_aggregator import (
    RetrievalMetricsAggregator,
)
from app.evaluation.retrieval_result_writer import (
    RetrievalResultWriter,
)
from app.models.chunk import Chunk
from app.models.embedded_chunk import EmbeddedChunk
from app.models.golden_dataset import (
    ChunkEvidenceIntervals,
    EvidenceInterval,
    GoldenDataset,
    GoldenDatasetMetadata,
    GoldenDatasetRecord,
    GoldenEvidence,
    GoldenRelevantChunk,
)
from app.repositories.chunk_repository import ChunkRepository
from app.services.retrieval_service import RetrievalService


pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

EMBEDDING_DIMENSION = 1024


async def test_retrieval_evaluation_writes_complete_json(
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
    retrieval_service = RetrievalService(
        embedding_client=embedding_client,
        chunk_repository=repository,
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
                        text="Paris is the capital of France.",
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
    pipeline = RetrievalEvaluationPipeline(
        retrieval_service=retrieval_service,
        retrieval_evaluator=RetrievalEvaluator(),
        metrics_aggregator=RetrievalMetricsAggregator(),
    )

    result = await pipeline.evaluate(
        dataset=dataset,
        top_k=1,
    )
    output_path = tmp_path / "retrieval.json"
    RetrievalResultWriter().write(
        result=result,
        output_path=output_path,
        embedding_model="test-embedding-model",
        embedding_dimension=EMBEDDING_DIMENSION,
    )

    payload = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert result.summary.query_count == 1
    assert result.summary.k == 1
    assert payload["configuration"] == {
        "top_k": 1,
        "embedding_model": "test-embedding-model",
        "embedding_dimension": EMBEDDING_DIMENSION,
    }
    assert payload["queries"][0]["query_id"] == "query-1"
    assert (
        payload["queries"][0]["retrieved_chunks"][0][
            "chunk_id"
        ]
        == "document.pdf_0001"
    )
    assert all(
        value == pytest.approx(1.0)
        for name, value in payload["queries"][0][
            "metrics"
        ].items()
        if name != "k"
    )
