from app.evaluation.generation_evaluator import GenerationEvaluator
from app.evaluation.generation_metrics_aggregator import GenerationMetricsAggregator
from app.evaluation.retrieval_evaluation_pipeline import RetrievalEvaluationPipeline
from app.models.golden_dataset import GoldenDataset
from app.models.rag_evaluation_result import RAGEvaluationRunResult, RAGQueryEvaluationResult
from app.services.generation_service import GenerationService


class RAGEvaluationPipeline:
    def __init__(
        self,
        retrieval_pipeline: RetrievalEvaluationPipeline,
        generation_service: GenerationService,
        generation_evaluator: GenerationEvaluator,
        generation_metrics_aggregator: GenerationMetricsAggregator
    ) -> None:
        self._retrieval_pipeline = retrieval_pipeline
        self._generation_service = generation_service
        self._generation_evaluator = generation_evaluator
        self._generation_metrics_aggregator = generation_metrics_aggregator

    async def evaluate(
        self,
        dataset: GoldenDataset,
        top_k: int,
    ) -> RAGEvaluationRunResult:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        retrieval_run = await self._retrieval_pipeline.evaluate(
            dataset=dataset,
            top_k=top_k,
        )

        records_by_query_id = {
            record.query_id: record
            for record in dataset.records
        }

        query_results = []

        for retrieval_result in retrieval_run.query_results:
            record = records_by_query_id.get(
                retrieval_result.query_id
            )

            if record is None:
                raise ValueError("retrieval result contains an unknown query_id")

            generation = await self._generation_service.generate(
                question=record.question,
                retrieved_chunks=retrieval_result.retrieved_chunks
            )

            generation_metrics = (
                await self._generation_evaluator.evaluate(
                    question=record.question,
                    answer=generation.answer,
                    contexts=[
                        retrieved_chunk.chunk.content
                        for retrieved_chunk in retrieval_result.retrieved_chunks
                    ],
                )
            )

            query_results.append(
                RAGQueryEvaluationResult(
                    retrieval=retrieval_result,
                    expected_answer=record.expected_answer,
                    generation=generation,
                    generation_metrics=generation_metrics,
                )
            )

        generation_summary = (
            self._generation_metrics_aggregator.aggregate(
                query_results
            )
        )

        return RAGEvaluationRunResult(
            query_results=tuple(query_results),
            retrieval_summary=retrieval_run.summary,
            generation_summary=generation_summary,
        )