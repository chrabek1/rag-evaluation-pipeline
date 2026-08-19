import json
from dataclasses import asdict
from pathlib import Path

from app.models.retrieval_evaluation_result import RetrievalEvaluationRunResult


class RetrievalResultWriter:
    def write(
        self,
        result: RetrievalEvaluationRunResult,
        output_path: Path,
        embedding_model: str,
        embedding_dimension: int,
    ) -> None:
        if not embedding_model.strip():
            raise ValueError(
                "embedding_model must not be empty"
            )
        
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be greater than 0")
        
        payload = {
            "configuration": {
                "top_k": result.summary.k,
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
            },
            "summary": asdict(result.summary),
            "queries": [
                {
                    "query_id": query_result.query_id,
                    "question": query_result.question,
                    "retrieved_chunks": [
                        {
                            "rank": rank,
                            "chunk_id": chunk.chunk_id,
                            "filename": chunk.chunk.filename,
                            "score": chunk.score,
                        }
                        for rank, chunk in enumerate(
                            query_result.retrieved_chunks,
                            start=1,
                        )
                    ],
                    "metrics": asdict(query_result.metrics),
                }
                for query_result in result.query_results
            ]
        }
        
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        
        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2
            )
            file.write("\n")