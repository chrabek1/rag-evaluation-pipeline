from collections.abc import Sequence

from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from app.models.generation_metrics_result import GenerationMetricsResult


class GenerationEvaluator:
    def __init__(
        self,
        faithfulness: Faithfulness,
        answer_relevancy: AnswerRelevancy,
    ) -> None:
        self._faithfulness = faithfulness
        self._answer_relevancy = answer_relevancy

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: Sequence[str],
    ) -> GenerationMetricsResult:
        if not question.strip():
            raise ValueError("question must not be empty")

        if not answer.strip():
            raise ValueError("answer must not be empty")

        if not contexts:
            raise ValueError("contexts must not be empty")

        if any(not context.strip() for context in contexts):
            raise ValueError("contexts must not contain empty text")

        faithfulness_result = (
            await self._faithfulness.ascore(
                user_input=question,
                response=answer,
                retrieved_contexts=list(contexts),
            )
        )

        answer_relevancy_result = (
            await self._answer_relevancy.ascore(
                user_input=question,
                response=answer,
            )
        )

        return GenerationMetricsResult(
            faithfulness=float(faithfulness_result.value),
            answer_relevancy=float(answer_relevancy_result.value),
        )