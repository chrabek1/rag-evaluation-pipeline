from dataset_preparation.build_golden_dataset import (
    chunk_contains_evidence,
    contains_evidence_fragment,
)


def test_math_evidence_accepts_reordered_pdf_tokens() -> None:
    evidence = (
        r"$\lambda_{\min }\left(\mathrm{Q}_{i}^{(\text {nom })}\right) "
        r"\in[-3.2,-2.7]$"
    )
    chunk = "lambda min Q nom i in -3.2, -2.7"

    assert contains_evidence_fragment(evidence, chunk)


def test_math_evidence_rejects_different_numeric_result() -> None:
    evidence = (
        r"$\lambda_{\min }\left(\mathrm{Q}_{i}^{(\text {nom })}\right) "
        r"\in[-3.2,-2.7]$"
    )
    chunk = "lambda min Q nom i in -35.9, -25.9"

    assert not contains_evidence_fragment(evidence, chunk)


def test_math_evidence_rejects_related_but_different_formula() -> None:
    evidence = (
        r"U(\boldsymbol{r}, \omega)=-\frac{\operatorname{Re}"
        r"[\alpha(\omega)]}{2 \epsilon_{0} c} I(\boldsymbol{r})"
    )
    chunk = (
        "F(x, z, omega) = -d U(x, z, omega) / dx = "
        "Re[alpha(omega)] / (2 epsilon 0 c) d I(x, z) / dx"
    )

    assert not contains_evidence_fragment(evidence, chunk)


def test_scattered_common_phrases_do_not_make_chunk_relevant() -> None:
    evidence = (
        "In the limit of small infectivity, the expected probability of "
        "infection is a decreasing function of both dose variances."
    )
    chunk = (
        "The expected probability of infection is discussed here. "
        "Many unrelated observations and model assumptions separate the "
        "phrases. In another experiment dose variances were recorded."
    )

    assert not contains_evidence_fragment(evidence, chunk)


def test_evidence_split_at_chunk_boundary_is_retained() -> None:
    evidence = (
        "The empirical model estimates transition probabilities from the "
        "previous occupation and does not use any additional covariates."
    )
    chunk = (
        "The empirical model estimates transition probabilities from the "
        "previous occupation"
    )

    assert contains_evidence_fragment(evidence, chunk)


def test_weak_matches_from_multiple_evidence_do_not_accumulate() -> None:
    evidence_texts = [
        "Dose response models map a microbial dose to infection probability.",
        "They have been applied across diverse species and host populations.",
    ]
    chunk = (
        "The paper mentions dose response models in one section and host "
        "populations in a separate, unrelated discussion."
    )

    assert not chunk_contains_evidence(evidence_texts, chunk)
