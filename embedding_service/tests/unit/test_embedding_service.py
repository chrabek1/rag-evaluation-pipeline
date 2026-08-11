import pytest
from unittest.mock import Mock

from app.services.embedding import EmbeddingService
from app.exceptions.embedding import EmbeddingError

def test_embed_returns_vectors_as_lists():
    mock_model = Mock()
    

    mock_model.encode.return_value = [
        Mock(tolist=lambda: [0.1, 0.2]),
        Mock(tolist=lambda: [0.3, 0.4]),
    ]
    
    service = EmbeddingService(embedding_model=mock_model)
    
    result = service.embed(["First", "Second"])
    
    assert result == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

def test_embed_raises_embedding_error_when_model_fails():
    mock_model = Mock()
    
    mock_model.encode.side_effect = RuntimeError("Model failure")
    
    service = EmbeddingService(embedding_model=mock_model)
    
    with pytest.raises(EmbeddingError):
        service.embed(["Test"])