import json
from pathlib import Path

import pytest

from app.loaders.golden_dataset_loader import (
    GoldenDatasetLoader,
)
from app.models.golden_dataset import GoldenDataset


def build_valid_payload() -> dict:
    return {
        "metadata": {
            "schema_version": 1,
            "evidence_interval_gap_tolerance": 3,
        },
        "records": [
            {
                "query_id": "query-1",
                "question": "Example question?",
                "expected_answer": "Example answer.",
                "evidence": [
                    {
                        "text": "Relevant evidence.",
                        "normalized_length": 18,
                    }
                ],
                "relevant_chunks": [
                    {
                        "chunk_id": "document.pdf_0001",
                        "evidence_coverage": 1.0,
                        "evidence_intervals": [
                            {
                                "evidence_index": 0,
                                "intervals": [[0, 18]],
                            }
                        ],
                    }
                ],
            }
        ],
    }


def write_payload(
    tmp_path: Path,
    payload: object,
) -> Path:
    path = tmp_path / "golden_dataset.json"
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return path


def test_load_returns_golden_dataset(tmp_path: Path) -> None:
    path = write_payload(
        tmp_path,
        build_valid_payload(),
    )

    dataset = GoldenDatasetLoader().load(path)

    assert isinstance(dataset, GoldenDataset)
    assert dataset.metadata.schema_version == 1
    assert (
        dataset.metadata.evidence_interval_gap_tolerance
        == 3
    )
    assert len(dataset.records) == 1

    record = dataset.records[0]
    assert record.query_id == "query-1"
    assert record.question == "Example question?"
    assert record.expected_answer == "Example answer."
    assert record.evidence[0].text == "Relevant evidence."
    assert record.evidence[0].normalized_length == 18

    chunk = record.relevant_chunks[0]
    assert chunk.chunk_id == "document.pdf_0001"
    assert chunk.evidence_coverage == 1.0

    interval_group = chunk.evidence_intervals[0]
    assert interval_group.evidence_index == 0
    assert interval_group.intervals[0].start == 0
    assert interval_group.intervals[0].end == 18


def test_load_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "golden_dataset.json"
    path.write_text(
        "{invalid json",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="golden dataset must contain valid JSON",
    ):
        GoldenDatasetLoader().load(path)


def test_load_rejects_non_object_root(tmp_path: Path) -> None:
    path = write_payload(
        tmp_path,
        [],
    )

    with pytest.raises(
        ValueError,
        match="golden dataset root must be an object",
    ):
        GoldenDatasetLoader().load(path)


def test_load_rejects_missing_metadata(tmp_path: Path) -> None:
    payload = build_valid_payload()
    del payload["metadata"]

    path = write_payload(tmp_path, payload)

    with pytest.raises(
        ValueError,
        match="missing required field: metadata",
    ):
        GoldenDatasetLoader().load(path)


def test_load_rejects_unsupported_schema_version(
    tmp_path: Path,
) -> None:
    payload = build_valid_payload()
    payload["metadata"]["schema_version"] = 2

    path = write_payload(tmp_path, payload)

    with pytest.raises(
        ValueError,
        match=(
            "unsupported golden dataset schema_version: 2"
        ),
    ):
        GoldenDatasetLoader().load(path)


def test_load_rejects_non_list_records(tmp_path: Path) -> None:
    payload = build_valid_payload()
    payload["records"] = {}

    path = write_payload(tmp_path, payload)

    with pytest.raises(
        ValueError,
        match="records must be a list",
    ):
        GoldenDatasetLoader().load(path)


def test_load_rejects_invalid_interval_shape(
    tmp_path: Path,
) -> None:
    payload = build_valid_payload()
    payload["records"][0]["relevant_chunks"][0][
        "evidence_intervals"
    ][0]["intervals"] = [[0, 10, 18]]

    path = write_payload(tmp_path, payload)

    with pytest.raises(
        ValueError,
        match=(
            "evidence interval must contain start and end"
        ),
    ):
        GoldenDatasetLoader().load(path)