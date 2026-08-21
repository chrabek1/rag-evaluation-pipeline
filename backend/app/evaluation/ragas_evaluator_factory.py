from instructor import Mode, from_litellm
from litellm import acompletion
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    Faithfulness,
)

from app.clients.embedding_client import EmbeddingClient
from app.evaluation.generation_evaluator import (
    GenerationEvaluator,
)
from app.evaluation.ragas_embedding_adapter import (
    RagasEmbeddingAdapter,
)


def create_ragas_evaluator(
    provider: str,
    model: str,
    embedding_client: EmbeddingClient,
    temperature: float = 0.0,
    answer_relevancy_strictness: int = 3,
    gemini_api_key: str | None = None,
    ollama_base_url: str | None = None,
) -> GenerationEvaluator:
    normalized_provider = provider.strip().lower()

    if not model.strip():
        raise ValueError("model must not be empty")

    if temperature < 0:
        raise ValueError(
            "temperature must not be negative"
        )

    if answer_relevancy_strictness <= 0:
        raise ValueError(
            "answer_relevancy_strictness must be positive"
        )

    if normalized_provider == "gemini":
        if (
            gemini_api_key is None
            or not gemini_api_key.strip()
        ):
            raise ValueError(
                "gemini_api_key is required for Gemini"
            )

        ragas_model = f"gemini/{model}"
        ragas_provider = "google"
        provider_arguments = {
            "api_key": gemini_api_key,
        }
        instructor_mode = None
    elif normalized_provider == "ollama":
        if (
            ollama_base_url is None
            or not ollama_base_url.strip()
        ):
            raise ValueError(
                "ollama_base_url is required for Ollama"
            )

        ragas_model = f"ollama/{model}"
        ragas_provider = "ollama"
        provider_arguments = {
            "api_base": ollama_base_url.rstrip("/"),
        }
        instructor_mode = Mode.JSON_SCHEMA
    else:
        raise ValueError(
            f"Unsupported evaluation provider: {provider}"
        )

    if instructor_mode is None:
        ragas_client = from_litellm(
            acompletion,
        )
    else:
        ragas_client = from_litellm(
            acompletion,
            mode=instructor_mode,
        )

    ragas_llm = llm_factory(
        model=ragas_model,
        provider=ragas_provider,
        client=ragas_client,
        adapter="litellm",
        temperature=temperature,
        **provider_arguments,
    )

    embeddings = RagasEmbeddingAdapter(
        embedding_client=embedding_client,
    )

    return GenerationEvaluator(
        faithfulness=Faithfulness(
            llm=ragas_llm,
        ),
        answer_relevancy=AnswerRelevancy(
            llm=ragas_llm,
            embeddings=embeddings,
            strictness=answer_relevancy_strictness,
        ),
    )
