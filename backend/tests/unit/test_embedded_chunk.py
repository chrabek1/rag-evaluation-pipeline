import pytest

from app.models.chunk import Chunk
from app.models.embedded_chunk import EmbeddedChunk


def test_embedded_chunk_stores_data() -> None:
    chunk = Chunk(
        filename="document.pdf",
        content="Example content",
    )

    embedded_chunk = EmbeddedChunk(
        chunk_id="document.pdf_0001",
        chunk=chunk,
        embedding=[0.1, 0.2, 0.3],
    )

    assert embedded_chunk.chunk_id == "document.pdf_0001"
    assert embedded_chunk.chunk is chunk
    assert embedded_chunk.embedding == [0.1, 0.2, 0.3]


@pytest.mark.parametrize("chunk_id", ["", " ", "\t", "\n"])
def test_embedded_chunk_rejects_empty_chunk_id(chunk_id: str) -> None:
    chunk = Chunk(
        filename="document.pdf",
        content="Example content",
    )

    with pytest.raises(ValueError, match="chunk_id cannot be empty"):
        EmbeddedChunk(
            chunk_id=chunk_id,
            chunk=chunk,
            embedding=[0.1, 0.2],
        )


def test_embedded_chunk_rejects_empty_embedding() -> None:
    chunk = Chunk(
        filename="document.pdf",
        content="Example content",
    )

    with pytest.raises(ValueError, match="embedding cannot be empty"):
        EmbeddedChunk(
            chunk_id="document.pdf_0001",
            chunk=chunk,
            embedding=[],
        )