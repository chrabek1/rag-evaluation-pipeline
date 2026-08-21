from dataclasses import dataclass

from math import isfinite


@dataclass(frozen=True, slots=True)
class GenerationMetricsResult:
    faithfulness: float
    answer_relevancy: float

    def __post_init__(self) -> None:
        self._validate_score(
            name="faithfulness",
            value=self.faithfulness,
            minimum=0.0,
            maximum=1.0,
        )
        self._validate_score(
            name="answer_relevancy",
            value=self.answer_relevancy,
            minimum=-1.0,
            maximum=1.0,
        )

    @staticmethod
    def _validate_score(
        name: str,
        value: float,
        minimum: float,
        maximum: float
    ) -> None:
        if not isfinite(value):
            raise ValueError(f"{name} must be finite")

        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")