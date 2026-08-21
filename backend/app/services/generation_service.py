from collections.abc import Sequence
from time import perf_counter

from app.clients.llm_client import LLMClient
from app.models.generation_result import GenerationResult
from app.models.retrieved_chunk import RetrievedChunk


class GenerationService:
    def __init__(
        self,
        llm_client: LLMClient,
    ) -> None:
        self._llm_client = llm_client

    async def generate(
        self,
        question: str,
        retrieved_chunks: Sequence[RetrievedChunk],
    ) -> GenerationResult:
        if not question.strip():
            raise ValueError("question must not be empty")

        if not retrieved_chunks:
            raise ValueError("retrieved_chunks must not be empty")

        prompt = self._build_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        started_at = perf_counter()

        response = await self._llm_client.generate(prompt)

        latency_seconds = perf_counter() - started_at

        return GenerationResult(
            response=response,
            latency_seconds=latency_seconds,
        )

    @staticmethod
    def _build_prompt(
        question: str,
        retrieved_chunks: Sequence[RetrievedChunk],
    ) -> str:
        contexts = "\n\n".join(
            (
                f"[Chunk {index}: {chunk.chunk_id}]\n"
                f"{chunk.chunk.content}"
            )
            for index, chunk in enumerate(
                retrieved_chunks,
                start=1,
            )
        )

        return (
            "Answer the question using only the provided context.\n"
            "Do not use external knowledge.\n"
            "If the context does not contain enough information, "
            "state that the answer cannot be determined from the "
            "provided context.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{contexts}\n\n"
            "Answer:"
        )