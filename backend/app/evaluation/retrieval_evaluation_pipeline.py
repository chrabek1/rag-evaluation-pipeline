import logging

from app.evaluation.retrieval_evaluator import RetrievalEvaluator
from app.evaluation.retrieval_metrics_aggregator import RetrievalMetricsAggregator
from app.models.golden_dataset import (
    GoldenDataset,
    GoldenDatasetRecord,
)
from app.models.retrieval_evaluation_result import (
    RetrievalEvaluationRunResult,
    RetrievalQueryEvaluationResult,
)
from app.services.retrieval_service import RetrievalService


logger = logging.getLogger(__name__)


class RetrievalEvaluationPipeline:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        retrieval_evaluator: RetrievalEvaluator,
        metrics_aggregator: RetrievalMetricsAggregator,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._retrieval_evaluator = retrieval_evaluator
        self._metrics_aggregator = metrics_aggregator
        
    async def evaluate(
        self,
        dataset: GoldenDataset,
        top_k: int,
    ) -> RetrievalEvaluationRunResult:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        
        query_results = []
        
        for index, record in enumerate(
            dataset.records,
            start=1,
        ):
            logger.info(
                "Evaluating query %d/%d: %s",
                index,
                len(dataset.records),
                record.query_id,
            )
            
            retrieved_chunks = (
                await self._retrieval_service.retrieve(
                    query=record.question,
                    top_k=top_k,
                )
            )
            
            metrics = self._retrieval_evaluator.evaluate(
                relevance_by_chunk_id=(
                    self._build_relevance_mapping(record)
                ),
                evidence_lengths=[
                    item.normalized_length
                    for item in record.evidence
                ],
                evidence_intervals_by_chunk_id=(
                    self._build_evidence_intervals_mapping(
                        record
                    )
                ),
                interval_gap_tolerance=(
                    dataset.metadata.evidence_interval_gap_tolerance
                ),
                retrieved_chunk_ids=[
                    chunk.chunk_id
                    for chunk in retrieved_chunks
                ],
                k=top_k
            )
            
            query_results.append(
                RetrievalQueryEvaluationResult(
                    query_id=record.query_id,
                    question=record.question,
                    retrieved_chunks=tuple(retrieved_chunks),
                    metrics=metrics,
                )
            )
        
        summary = self._metrics_aggregator.aggregate(
            [
                result.metrics
                for result in query_results
            ]
        )
        return RetrievalEvaluationRunResult(
            query_results=tuple(query_results),
            summary=summary,
        )
    
    @staticmethod
    def _build_relevance_mapping(
        record: GoldenDatasetRecord,
    ) -> dict[str, float]:
        return {
            chunk.chunk_id: chunk.evidence_coverage
            for chunk in record.relevant_chunks
        }
    
    @staticmethod
    def _build_evidence_intervals_mapping(
        record: GoldenDatasetRecord
    ) -> dict[
        str,
        dict[int, list[tuple[int, int]]],
    ]:
        return {
            chunk.chunk_id: {
                group.evidence_index: [
                    (interval.start, interval.end)
                    for interval in group.intervals
                ]
                for group in chunk.evidence_intervals
            }
            for chunk in record.relevant_chunks
        }
