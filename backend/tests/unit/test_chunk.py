import pytest

from app.models.chunk import Chunk


def test_chunk_stores_filename_and_content() -> None:
    chunk = Chunk(
        filename="document.pdf",
        content="Example chunk content",
    )

    assert chunk.filename == "document.pdf"
    assert chunk.content == "Example chunk content"


@pytest.mark.parametrize("filename", ["", " ", "\t", "\n"])
def test_chunk_rejects_empty_filename(filename: str) -> None:
    with pytest.raises(ValueError, match="filename cannot be empty"):
        Chunk(
            filename=filename,
            content="Valid content",
        )


@pytest.mark.parametrize("content", ["", " ", "\t", "\n"])
def test_chunk_rejects_empty_content(content: str) -> None:
    with pytest.raises(ValueError, match="content cannot be empty"):
        Chunk(
            filename="document.pdf",
            content=content,
        )