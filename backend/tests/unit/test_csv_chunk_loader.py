import pytest

from pathlib import Path

from app.loaders.csv_chunk_loader import CsvChunkLoader


def test_load_returns_chunks_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "chunks.csv"
    csv_path.write_text(
        "filename,content\n"
        "doc-a.pdf,First chunk\n"
        "doc-a.pdf,Second chunk\n"
        "doc-b.pdf,Third chunk\n",
        encoding="utf-8",
    )

    loader = CsvChunkLoader()

    chunks = loader.load(csv_path)

    assert len(chunks) == 3

    assert chunks[0].filename == "doc-a.pdf"
    assert chunks[0].content == "First chunk"

    assert chunks[1].filename == "doc-a.pdf"
    assert chunks[1].content == "Second chunk"

    assert chunks[2].filename == "doc-b.pdf"
    assert chunks[2].content == "Third chunk"
    
def test_load_preserves_multiline_content_with_commas(tmp_path: Path) -> None:
    csv_path = tmp_path / "chunks.csv"
    csv_path.write_text(
        'filename,content\n'
        'doc-a.pdf,"First line, with comma.\n'
        'Second line of the same chunk.\n'
        'Third line, also with comma."\n',
        encoding="utf-8",
    )

    loader = CsvChunkLoader()

    chunks = loader.load(csv_path)

    assert len(chunks) == 1
    assert chunks[0].filename == "doc-a.pdf"
    assert chunks[0].content == (
        "First line, with comma.\n"
        "Second line of the same chunk.\n"
        "Third line, also with comma."
    )
    
def test_load_raises_error_when_required_column_is_missing(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "chunks.csv"
    csv_path.write_text(
        "filename,text\n"
        "doc-a.pdf,Some text\n",
        encoding="utf-8",
    )

    loader = CsvChunkLoader()

    with pytest.raises(
        ValueError,
        match="CSV must contain columns: filename, content",
    ):
        loader.load(csv_path)

def test_load_returns_empty_list_for_empty_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "chunks.csv"
    csv_path.write_text(
        "filename,content\n",
        encoding="utf-8",
    )

    loader = CsvChunkLoader()

    chunks = loader.load(csv_path)

    assert chunks == []
    
def test_load_rejects_empty_content(tmp_path: Path) -> None:
    csv_path = tmp_path / "chunks.csv"
    csv_path.write_text(
        'filename,content\n'
        'doc-a.pdf,""\n',
        encoding="utf-8",
    )

    loader = CsvChunkLoader()

    with pytest.raises(ValueError):
        loader.load(csv_path)