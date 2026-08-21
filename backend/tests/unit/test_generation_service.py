from unittest.mock import AsyncMock, Mock

import pytest

from app.models.chunk import Chunk
from app.models.llm_response import LLMResponse
from app.models.retrieved_chunk import RetrievedChunk
from app.services.generation_service import GenerationService


def create_retrieved_chunk(
    chunk_id: str,
    content: str,
    score: float,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        chunk=Chunk(
            filename="document.pdf",
            content=content,
        ),
        score=score,
    )


@pytest.mark.asyncio
async def test_generate_builds_prompt_and_returns_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm_client = Mock()
    response = LLMResponse(
        text="Generated answer",
        model="test-model",
        input_tokens=30,
        output_tokens=10,
    )
    llm_client.generate = AsyncMock(
        return_value=response
    )

    clock = Mock(
        side_effect=[10.0, 11.5],
    )
    monkeypatch.setattr(
        "app.services.generation_service.perf_counter",
        clock,
    )

    service = GenerationService(llm_client)

    chunks = [
        create_retrieved_chunk(
            chunk_id="document.pdf_0001",
            content="First context.",
            score=0.9,
        ),
        create_retrieved_chunk(
            chunk_id="document.pdf_0002",
            content="Second context.",
            score=0.8,
        ),
    ]

    result = await service.generate(
        question="Example question?",
        retrieved_chunks=chunks,
    )

    llm_client.generate.assert_awaited_once()

    prompt = llm_client.generate.await_args.args[0]

    assert "Example question?" in prompt
    assert "[Chunk 1: document.pdf_0001]" in prompt
    assert "First context." in prompt
    assert "[Chunk 2: document.pdf_0002]" in prompt
    assert "Second context." in prompt
    assert prompt.index("First context.") < prompt.index(
        "Second context."
    )

    assert result.response == response
    assert result.answer == "Generated answer"
    assert result.latency_seconds == 1.5


@pytest.mark.asyncio
async def test_generate_rejects_empty_question() -> None:
    llm_client = Mock()
    llm_client.generate = AsyncMock()

    service = GenerationService(llm_client)

    with pytest.raises(
        ValueError,
        match="question must not be empty",
    ):
        await service.generate(
            question=" ",
            retrieved_chunks=[
                create_retrieved_chunk(
                    chunk_id="document.pdf_0001",
                    content="Context.",
                    score=0.9,
                )
            ],
        )

    llm_client.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_rejects_empty_chunks() -> None:
    llm_client = Mock()
    llm_client.generate = AsyncMock()

    service = GenerationService(llm_client)

    with pytest.raises(
        ValueError,
        match="retrieved_chunks must not be empty",
    ):
        await service.generate(
            question="Example question?",
            retrieved_chunks=[],
        )

    llm_client.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_propagates_llm_error() -> None:
    llm_client = Mock()
    llm_client.generate = AsyncMock(
        side_effect=RuntimeError("LLM unavailable")
    )
    service = GenerationService(llm_client)

    with pytest.raises(
        RuntimeError,
        match="LLM unavailable",
    ):
        await service.generate(
            question="Example question?",
            retrieved_chunks=[
                create_retrieved_chunk(
                    chunk_id="document.pdf_0001",
                    content="Context.",
                    score=0.9,
                )
            ],
        )

    llm_client.generate.assert_awaited_once()


def test_build_prompt_instructs_model_to_use_only_context() -> None:
    prompt = GenerationService._build_prompt(
        question="Example question?",
        retrieved_chunks=[
            create_retrieved_chunk(
                chunk_id="document.pdf_0001",
                content="Example context.",
                score=0.9,
            )
        ],
    )

    assert (
        "Answer the question using only the provided context."
        in prompt
    )
    assert "Do not use external knowledge." in prompt
    assert (
        "the answer cannot be determined from the "
        "provided context"
        in prompt
    )
