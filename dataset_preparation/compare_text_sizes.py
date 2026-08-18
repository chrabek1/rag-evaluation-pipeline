import csv
import json
import statistics
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PREPARATION_ROOT = Path(__file__).resolve().parent
DATA_DIR = PREPARATION_ROOT / "open_rag_data"

CHUNKS_PATH = REPOSITORY_ROOT / "dane.csv"
QUESTIONS_PATH = DATA_DIR / "selected_questions.json"
EVIDENCE_PATH = DATA_DIR / "evidence_annotations.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def summarize(values: list[int]) -> dict[str, float]:
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def print_stats(name: str, values: list[int]) -> None:
    stats = summarize(values)

    print(f"{name} (characters):")

    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")

    print()


def main() -> None:
    with CHUNKS_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        chunk_lengths = [
            len(row["content"])
            for row in reader
        ]

    questions = load_json(QUESTIONS_PATH)
    evidence_output = load_json(EVIDENCE_PATH)

    ground_truth_lengths = [
        len(item["ground_truth_text"])
        for item in questions
    ]

    evidence_lengths = [
        sum(
            len(evidence)
            for evidence in item["evidence_texts"]
        )
        for item in evidence_output
        if item["evidence_texts"]
    ]

    empty_evidence_count = sum(
        1
        for item in evidence_output
        if not item["evidence_texts"]
    )

    print_stats("Chunks", chunk_lengths)
    print_stats("Ground truth texts", ground_truth_lengths)
    print_stats("Evidence texts", evidence_lengths)

    chunk_median = statistics.median(chunk_lengths)
    ground_truth_median = statistics.median(
        ground_truth_lengths
    )
    evidence_median = statistics.median(evidence_lengths)

    print(
        "Median ground truth / median chunk ratio: "
        f"{ground_truth_median / chunk_median:.2f}x"
    )

    print(
        "Median evidence / median chunk ratio: "
        f"{evidence_median / chunk_median:.2f}x"
    )

    print(
        "Median ground truth / median evidence ratio: "
        f"{ground_truth_median / evidence_median:.2f}x"
    )

    print(f"Empty evidence records: {empty_evidence_count}")


if __name__ == "__main__":
    main()
