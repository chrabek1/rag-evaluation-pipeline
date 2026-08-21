import json
from dataclasses import asdict
from pathlib import Path

from app.models.rag_evaluation_result import RAGEvaluationRunResult


class RAGResultWriter:
    def write(
        self,
        result: RAGEvaluationRunResult,
        output_path: Path,
        embedding_model: str,
        embedding_dimension: int,
        generation_provider: str,
        generation_model: str,
        evaluation_provider: str,
        evaluation_model: str,
    ) -> None:
        configuration_values = {
            "embedding_model": embedding_model,
            "generation_provider": generation_provider,
            "generation_model": generation_model,
            "evaluation_provider": evaluation_provider,
            "evaluation_model": evaluation_model,
        }

        for name, value in configuration_values.items():
            if not value.strip():
                raise ValueError(
                    f"{name} must not be empty"
                )

        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be greater than 0")

        payload = {
            "configuration": {
                "top_k": result.retrieval_summary.k,
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
                "generation_provider": generation_provider,
                "generation_model": generation_model,
                "evaluation_provider": evaluation_provider,
                "evaluation_model": evaluation_model,
            },
            "summary": {
                "retrieval": asdict(result.retrieval_summary),
                "generation": asdict(result.generation_summary),
            },
            "queries": [
                self._serialize_query(query_result)
                for query_result in result.query_results
            ],
        }

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")

    @staticmethod
    def _serialize_query(
        query_result,
    ) -> dict:
        return {
            "query_id": query_result.query_id,
            "question": query_result.question,
            "expected_answer": query_result.expected_answer,
            "retrieval": {
                "retrieved_chunks": [
                    {
                        "rank": rank,
                        "chunk_id": chunk.chunk_id,
                        "filename": chunk.chunk.filename,
                        "content": chunk.chunk.content,
                        "score": chunk.score,
                    }
                    for rank, chunk in enumerate(
                        query_result.retrieval.retrieved_chunks,
                        start=1,
                    )
                ],
                "metrics": asdict(query_result.retrieval.metrics),
            },
            "generation": {
                "answer": query_result.generation.answer,
                "model": query_result.generation.response.model,
                "input_tokens": query_result.generation.response.input_tokens,
                "output_tokens": query_result.generation.response.output_tokens,
                "latency_seconds": query_result.generation.latency_seconds,
                "metrics": asdict(query_result.generation_metrics),
            },
        }