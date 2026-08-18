import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PREPARATION_ROOT = Path(__file__).resolve().parent
DATA_DIR = PREPARATION_ROOT / "open_rag_data"
OPEN_RAGBENCH_DIR = DATA_DIR / "open_ragbench/pdf/arxiv"

CHUNKS_PATH = REPOSITORY_ROOT / "dane.csv"
QUESTIONS_PATH = DATA_DIR / "selected_questions.json"
EVIDENCE_PATH = DATA_DIR / "evidence_annotations.json"
ANSWERS_PATH = OPEN_RAGBENCH_DIR / "answers.json"
OUTPUT_PATH = REPOSITORY_ROOT / "golden_dataset.json"

# Krótsze dopasowania często są przypadkowymi wspólnymi frazami.
MIN_MATCH_CHARS = 15
MIN_RELEVANT_MATCH_CHARS = 30
MIN_FRAGMENT_COVERAGE = 0.40
MIN_BOUNDARY_FRAGMENT_COVERAGE = 0.35
MIN_STRONG_FRAGMENT_COVERAGE = 0.80
MAX_MATCH_GAP_CHARS = 60
MAX_COVERAGE_GAP_CHARS = 3
MIN_MATH_COVERAGE = 0.85
MIN_MATH_SEGMENT_TOKENS = 8
MATH_SPAN_RE = re.compile(
    r"\$\$(.*?)\$\$|\$(.*?)\$",
    re.DOTALL,
)
STRUCTURAL_MATH_RE = re.compile(
    r"[\\_^{}=<>≤≥≈≠∑√∞⊤∂∫∪]"
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


@lru_cache(maxsize=None)
def normalize_text(text: str) -> str:
    """Normalize prose and PDF/LaTeX math to a comparable form.

    PDF extraction turns, for example, ``\alpha`` into ``α`` and inserts
    spaces inside formulas and decimal numbers.  A whitespace-only
    normalization therefore cannot match evidence copied from Markdown
    against the corresponding PDF chunk.
    """
    # PDF extractors often split a word at the end of a line. This must be
    # repaired before all whitespace is collapsed.
    text = re.sub(
        r"(?<=[^\W\d_])-\s*\r?\n\s*(?=[^\W\d_])",
        "",
        text,
        flags=re.UNICODE,
    )
    text = unicodedata.normalize("NFKC", text).lower()

    # Normalize typographic variants before punctuation is tokenized.
    text = text.translate(
        str.maketrans(
            {
                "−": "-",
                "–": "-",
                "—": "-",
                "‐": "-",
                "·": " times ",
                "⋅": " times ",
                "×": " times ",
            }
        )
    )

    symbol_names = {
        "α": " alpha ",
        "β": " beta ",
        "γ": " gamma ",
        "δ": " delta ",
        "ε": " epsilon ",
        "ϵ": " epsilon ",
        "η": " eta ",
        "θ": " theta ",
        "κ": " kappa ",
        "λ": " lambda ",
        "μ": " mu ",
        "ν": " nu ",
        "ξ": " xi ",
        "π": " pi ",
        "ρ": " rho ",
        "σ": " sigma ",
        "τ": " tau ",
        "φ": " phi ",
        "ϕ": " phi ",
        "ψ": " psi ",
        "ω": " omega ",
        "ℓ": " ell ",
        "∆": " delta ",
        "Δ": " delta ",
        "∈": " in ",
        "≤": " le ",
        "≥": " ge ",
        "≈": " approx ",
        "⊤": " top ",
        "×": " times ",
        "·": " times ",
        "⋅": " times ",
        "√": " sqrt ",
        "∑": " sum ",
        "∞": " infinity ",
        "≃": " approx ",
        "≅": " approx ",
        "≠": " neq ",
        "∉": " notin ",
        "∂": " partial ",
        "∫": " integral ",
        "∪": " union ",
        "∼": " sim ",
        "≡": " equiv ",
        "±": " plusminus ",
        "∣": " mid ",
        "|": " mid ",
        "→": " to ",
        "↪": " to ",
        "↑": " up ",
        "↓": " down ",
        "′": " prime ",
        "″": " doubleprime ",
        "ˆ": " hat ",
        "˜": " tilde ",
        "¯": " bar ",
        "̂": " hat ",
        "̃": " tilde ",
        "̄": " bar ",
    }
    text = "".join(symbol_names.get(char, char) for char in text)

    latex_symbols = {
        "in": " in ",
        "le": " le ",
        "ge": " ge ",
        "approx": " approx ",
        "top": " top ",
        "times": " times ",
        "neq": " neq ",
        "notin": " notin ",
        "sum": " sum ",
        "sqrt": " sqrt ",
        "infty": " infinity ",
        "cdot": " times ",
        "partial": " partial ",
        "int": " integral ",
        "cup": " union ",
        "sim": " sim ",
        "simeq": " approx ",
        "equiv": " equiv ",
        "pm": " plusminus ",
        "mid": " mid ",
        "to": " to ",
        "rightarrow": " to ",
        "longrightarrow": " to ",
        "mapsto": " to ",
        "uparrow": " up ",
        "downarrow": " down ",
        "prime": " prime ",
        "hat": " hat ",
        "tilde": " tilde ",
        "bar": " bar ",
    }
    formatting_commands = {
        "boldsymbol",
        "mathbf",
        "mathbb",
        "mathcal",
        "mathfrak",
        "mathit",
        "mathsf",
        "mathtt",
        "bm",
        "mathrm",
        "operatorname",
        "text",
        "left",
        "right",
        "frac",
        "dfrac",
        "tfrac",
        "big",
        "bigl",
        "bigr",
        "bigg",
        "biggl",
        "biggr",
        "overline",
        "underline",
    }

    def replace_latex_command(match: re.Match) -> str:
        command = match.group(1).lower()
        if command in formatting_commands:
            return " "
        return latex_symbols.get(command, f" {command} ")

    text = re.sub(r"\\([a-zA-Z]+)", replace_latex_command, text)

    # PDF extraction frequently inserts spaces around the decimal point.
    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)

    # Keep numeric sign information while treating prose hyphens as token
    # separators. Optional whitespace accommodates PDF output such as "- 3.2".
    text = re.sub(r"-(?=\s*\d)", " minus ", text)

    # Tokenization preserves word boundaries and decimal values. Braces,
    # Markdown markers, equation layout and ordinary punctuation are ignored.
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?", text)

    return " ".join(tokens)


def load_chunks() -> dict[str, list[dict]]:
    chunks_by_filename: dict[str, list[dict]] = defaultdict(list)
    counters: dict[str, int] = defaultdict(int)

    with CHUNKS_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            filename = row["filename"]
            counters[filename] += 1

            chunk_id = (
                f"{filename}_"
                f"{counters[filename]:04d}"
            )

            chunks_by_filename[filename].append(
                {
                    "chunk_id": chunk_id,
                    "content": row["content"],
                }
            )

    return chunks_by_filename


@lru_cache(maxsize=None)
def is_math_evidence(text: str) -> bool:
    """Return whether mathematical notation dominates the evidence."""
    spans = MATH_SPAN_RE.findall(text)
    math_text = " ".join(
        display_math or inline_math
        for display_math, inline_math in spans
    )

    text_length = len(re.sub(r"\s+", "", text))
    math_length = len(re.sub(r"\s+", "", math_text))
    math_ratio = math_length / text_length if text_length else 0.0

    text_without_math_spans = MATH_SPAN_RE.sub(" ", text)
    structural_markers = len(
        STRUCTURAL_MATH_RE.findall(text_without_math_spans)
    )
    token_count = len(re.findall(r"[A-Za-z0-9]+", text))

    return math_ratio >= 0.5 or (
        not spans
        and structural_markers >= 3
        and token_count <= 30
    )


@lru_cache(maxsize=None)
def math_token_coverage(
    evidence_text: str,
    chunk_text: str,
) -> float:
    """Match local math tokens despite PDF sub/superscript reordering."""
    if not is_math_evidence(evidence_text):
        return 0.0

    evidence_tokens = normalize_text(evidence_text).split()
    chunk_tokens = normalize_text(chunk_text).split()
    if not evidence_tokens or not chunk_tokens:
        return 0.0

    evidence_counts = Counter(evidence_tokens)
    evidence_numbers = Counter(
        token
        for token in evidence_tokens
        if re.fullmatch(r"\d+(?:\.\d+)?", token)
    )
    anchor_tokens = Counter(evidence_tokens[:2])
    weights = {token: len(token) for token in evidence_counts}
    total_weight = sum(
        weights[token] * count
        for token, count in evidence_counts.items()
    )

    evidence_length = len(evidence_tokens)
    min_window = min(evidence_length, len(chunk_tokens))
    max_window = min(
        len(chunk_tokens),
        evidence_length + max(4, evidence_length // 3),
    )
    best_weight = 0

    for window_size in range(min_window, max_window + 1):
        window_counts = Counter(chunk_tokens[:window_size])
        window_count = len(chunk_tokens) - window_size + 1

        for start in range(window_count):
            numbers_match = all(
                window_counts[token] >= count
                for token, count in evidence_numbers.items()
            )
            anchor_matches = all(
                window_counts[token] >= count
                for token, count in anchor_tokens.items()
            )
            matched_weight = 0
            if numbers_match and anchor_matches:
                matched_weight = sum(
                    weights[token] * min(count, window_counts[token])
                    for token, count in evidence_counts.items()
                )
            best_weight = max(best_weight, matched_weight)
            if best_weight == total_weight:
                return 1.0

            if start + 1 >= window_count:
                continue

            outgoing_token = chunk_tokens[start]
            incoming_token = chunk_tokens[start + window_size]
            window_counts[outgoing_token] -= 1
            window_counts[incoming_token] += 1

    return best_weight / total_weight if total_weight else 0.0


@lru_cache(maxsize=None)
def match_details(
    evidence_text: str,
    chunk_text: str,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Return matching characters and intervals using one comparison pass."""
    evidence = normalize_text(evidence_text)
    chunk = normalize_text(chunk_text)

    if not evidence or not chunk:
        return 0, ()

    # Najpierw sprawdzamy idealne dopasowanie.
    if evidence in chunk:
        return len(evidence), ((0, len(evidence)),)

    matcher = SequenceMatcher(
        None,
        evidence,
        chunk,
        autojunk=False,
    )

    matching_blocks = [
        block
        for block in matcher.get_matching_blocks()
        if block.size >= MIN_MATCH_CHARS
    ]
    matching_groups = []
    current_group = []

    for block in matching_blocks:
        if not current_group:
            current_group = [block]
            continue

        previous = current_group[-1]
        evidence_gap = block.a - (previous.a + previous.size)
        chunk_gap = block.b - (previous.b + previous.size)

        if (
            evidence_gap <= MAX_MATCH_GAP_CHARS
            and chunk_gap <= MAX_MATCH_GAP_CHARS
        ):
            current_group.append(block)
            continue

        matching_groups.append(current_group)
        current_group = [block]

    if current_group:
        matching_groups.append(current_group)

    best_group = max(
        matching_groups,
        key=lambda group: sum(block.size for block in group),
        default=[],
    )
    sequence_match_chars = sum(block.size for block in best_group)
    intervals = tuple(
        (block.a, block.a + block.size)
        for block in best_group
    )

    math_coverage = math_token_coverage(evidence_text, chunk_text)
    if math_coverage >= MIN_MATH_COVERAGE:
        return len(evidence), ((0, len(evidence)),)

    return sequence_match_chars, intervals


def contains_evidence_fragment(
    evidence_text: str,
    chunk_text: str,
) -> bool:
    """Return whether a chunk contains a meaningful part of one evidence."""
    evidence = normalize_text(evidence_text)
    if not evidence:
        return False

    matched_chars, intervals = match_details(evidence_text, chunk_text)
    if is_math_evidence(evidence_text):
        math_matches = (
            matched_chars == len(evidence)
            and intervals == ((0, len(evidence)),)
        )
        if math_matches:
            return True

        prose_text = MATH_SPAN_RE.sub(" ", evidence_text)
        if normalize_text(prose_text) != evidence:
            return contains_evidence_fragment(prose_text, chunk_text)

        return False

    if matched_chars < min(MIN_RELEVANT_MATCH_CHARS, len(evidence)):
        return False

    coverage = matched_chars / len(evidence)
    longest_match = max(
        (end - start for start, end in intervals),
        default=0,
    )
    longest_match_coverage = longest_match / len(evidence)
    touches_evidence_boundary = bool(intervals) and (
        intervals[0][0] == 0
        or intervals[-1][1] == len(evidence)
    )

    return (
        coverage >= MIN_STRONG_FRAGMENT_COVERAGE
        or longest_match_coverage >= MIN_FRAGMENT_COVERAGE
        or (
            touches_evidence_boundary
            and coverage >= MIN_BOUNDARY_FRAGMENT_COVERAGE
        )
    )


def chunk_contains_evidence(
    evidence_texts: list[str],
    chunk_text: str,
) -> bool:
    """Return whether a chunk contains at least one evidence fragment."""
    return any(
        contains_evidence_fragment(
            evidence_text=evidence_text,
            chunk_text=chunk_text,
        )
        for evidence_text in evidence_texts
    )


def matched_char_count(
    evidence_text: str,
    chunk_text: str,
) -> int:
    matched_chars, _ = match_details(evidence_text, chunk_text)

    return matched_chars


def calculate_chunk_coverage(
    evidence_texts: list[str],
    chunk_text: str,
) -> float:
    normalized_evidence = [
        normalize_text(text)
        for text in evidence_texts
    ]

    total_evidence_chars = sum(
        len(text)
        for text in normalized_evidence
    )

    if total_evidence_chars == 0:
        return 0.0

    matched_chars = sum(
        matched_char_count(
            evidence_text=evidence,
            chunk_text=chunk_text,
        )
        for evidence in evidence_texts
    )

    return min(
        matched_chars / total_evidence_chars,
        1.0,
    )


def matching_evidence_intervals(
    evidence_text: str,
    chunk_text: str,
) -> list[tuple[int, int]]:
    """Return evidence character ranges supported by a single chunk."""
    matched_chars, intervals = match_details(evidence_text, chunk_text)

    evidence_length = len(normalize_text(evidence_text))
    if (
        intervals
        and evidence_length
        and matched_chars / evidence_length >= MIN_FRAGMENT_COVERAGE
    ):
        matched_intervals = [(intervals[0][0], intervals[-1][1])]
    else:
        matched_intervals = list(intervals)

    evidence = normalize_text(evidence_text)
    search_start = 0
    for math_span in MATH_SPAN_RE.finditer(evidence_text):
        span_text = math_span.group(0)
        normalized_span = normalize_text(span_text)
        span_tokens = normalized_span.split()
        if len(span_tokens) < MIN_MATH_SEGMENT_TOKENS:
            continue

        span_start = evidence.find(normalized_span, search_start)
        if span_start < 0:
            continue
        search_start = span_start + len(normalized_span)

        if math_token_coverage(span_text, chunk_text) < MIN_MATH_COVERAGE:
            continue

        matched_intervals.append(
            (span_start, span_start + len(normalized_span))
        )

    if matched_intervals:
        chunk = normalize_text(chunk_text)
        first_start = min(start for start, _ in matched_intervals)
        last_end = max(end for _, end in matched_intervals)
        prefix = evidence[:first_start].strip()
        suffix = evidence[last_end:].strip()

        if prefix and len(prefix) < MIN_MATCH_CHARS and prefix in chunk:
            matched_intervals.append((0, first_start))
        if suffix and len(suffix) < MIN_MATCH_CHARS and suffix in chunk:
            matched_intervals.append((last_end, len(evidence)))

    return matched_intervals


def calculate_combined_evidence_coverage(
    evidence_texts: list[str],
    chunk_texts: list[str],
) -> float:
    """Calculate evidence coverage by the union of all relevant chunks."""
    normalized_evidence = [
        normalize_text(text)
        for text in evidence_texts
    ]
    total_evidence_chars = sum(
        len(text)
        for text in normalized_evidence
    )

    if total_evidence_chars == 0:
        return 0.0

    covered_chars = 0

    for evidence_text, normalized_text in zip(
        evidence_texts,
        normalized_evidence,
    ):
        intervals = []
        for chunk_text in chunk_texts:
            intervals.extend(
                matching_evidence_intervals(
                    evidence_text=evidence_text,
                    chunk_text=chunk_text,
                )
            )

        if not intervals:
            continue

        intervals.sort()
        merged_start, merged_end = intervals[0]

        for start, end in intervals[1:]:
            if start <= merged_end + MAX_COVERAGE_GAP_CHARS:
                merged_end = max(merged_end, end)
                continue

            covered_chars += merged_end - merged_start
            merged_start, merged_end = start, end

        covered_chars += merged_end - merged_start

    return min(
        covered_chars / total_evidence_chars,
        1.0,
    )


def validate_evidence_annotations(
    questions: list[dict],
    annotations: list[dict],
) -> dict[str, list[str]]:
    question_by_id = {
        item["query_id"]: item
        for item in questions
    }
    annotation_ids = [item["query_id"] for item in annotations]
    annotation_by_id = {
        item["query_id"]: item.get("evidence_texts", [])
        for item in annotations
    }
    errors = []

    if len(question_by_id) != len(questions):
        errors.append("selected_questions.json contains duplicate query_id")
    if len(set(annotation_ids)) != len(annotation_ids):
        errors.append("evidence_annotations.json contains duplicate query_id")

    question_ids = set(question_by_id)
    annotated_ids = set(annotation_by_id)
    missing_ids = question_ids - annotated_ids
    extra_ids = annotated_ids - question_ids
    if missing_ids:
        errors.append(f"Missing evidence query_ids: {sorted(missing_ids)}")
    if extra_ids:
        errors.append(f"Unexpected evidence query_ids: {sorted(extra_ids)}")

    for query_id in question_ids & annotated_ids:
        evidence_texts = annotation_by_id[query_id]
        ground_truth = question_by_id[query_id]["ground_truth_text"]

        if not isinstance(evidence_texts, list) or not evidence_texts:
            errors.append(f"{query_id}: evidence_texts must be a non-empty list")
            continue
        if len(evidence_texts) != len(set(evidence_texts)):
            errors.append(f"{query_id}: duplicate evidence strings")

        for evidence_text in evidence_texts:
            if not isinstance(evidence_text, str) or not evidence_text:
                errors.append(f"{query_id}: evidence must be a non-empty string")
            elif evidence_text not in ground_truth:
                errors.append(
                    f"{query_id}: evidence is not an exact substring of "
                    f"ground_truth_text:\n{evidence_text}"
                )

    if errors:
        raise ValueError("Evidence validation failed:\n\n" + "\n\n".join(errors))

    return annotation_by_id


def main() -> None:
    questions = load_json(QUESTIONS_PATH)
    evidence_annotations = load_json(EVIDENCE_PATH)
    answers = load_json(ANSWERS_PATH)
    chunks_by_filename = load_chunks()

    if not isinstance(answers, dict):
        raise ValueError("answers.json must contain a query_id-to-answer object")

    answer_errors = []
    for question in questions:
        query_id = question["query_id"]
        answer = answers.get(query_id)
        if not isinstance(answer, str) or not answer.strip():
            answer_errors.append(
                f"Missing or empty answer for query_id {query_id}"
            )

    if answer_errors:
        raise ValueError(
            "Answer validation failed:\n\n" + "\n".join(answer_errors)
        )

    evidence_by_query_id = validate_evidence_annotations(
        questions,
        evidence_annotations,
    )

    golden_dataset = []

    for question in questions:
        query_id = question["query_id"]

        evidence_texts = evidence_by_query_id[query_id]

        filename = question["filename"]
        document_chunks = chunks_by_filename.get(
            filename,
            [],
        )

        relevant_chunks = []
        relevant_chunk_texts = []

        for chunk in document_chunks:
            coverage = calculate_chunk_coverage(
                evidence_texts=evidence_texts,
                chunk_text=chunk["content"],
            )

            if not chunk_contains_evidence(
                evidence_texts=evidence_texts,
                chunk_text=chunk["content"],
            ):
                continue

            relevant_chunks.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "evidence_coverage": round(
                        coverage,
                        4,
                    ),
                }
            )
            relevant_chunk_texts.append(chunk["content"])

        relevant_chunks.sort(
            key=lambda item: item["evidence_coverage"],
            reverse=True,
        )

        evidence_text = " ".join(evidence_texts)
        evidence_coverage_percentage = round(
            calculate_combined_evidence_coverage(
                evidence_texts=evidence_texts,
                chunk_texts=relevant_chunk_texts,
            )
            * 100,
            2,
        )

        golden_dataset.append(
            {
                "query_id": query_id,
                "question": question["question"],
                "expected_answer": answers[query_id],
                "type": question["type"],
                "source": question["source"],
                "doc_id": question["doc_id"],
                "filename": filename,
                "section_id": question["section_id"],
                "ground_truth_text": question[
                    "ground_truth_text"
                ],
                "evidence_text": evidence_text,
                "evidence_coverage_percentage": (
                    evidence_coverage_percentage
                ),
                "relevant_chunks": relevant_chunks,
            }
        )

    no_relevant_chunks = [
        item["query_id"]
        for item in golden_dataset
        if not item["relevant_chunks"]
    ]

    if no_relevant_chunks:
        raise ValueError(
            "No relevant chunks found for query_ids: "
            f"{no_relevant_chunks}"
        )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            golden_dataset,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    print(f"Golden dataset records: {len(golden_dataset)}")
    print(f"Evidence annotations: {len(evidence_annotations)}")
    print("Records without relevant chunks: 0")
    print("Validation: PASS")
    print(f"Saved to: {OUTPUT_PATH.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
