from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.main import app
from app.api.dependencies import get_embedding_service

client = TestClient(app)

def test_embed_returns_vectors():
    mock_service = Mock()
    mock_service.embed.return_value = [[0.2, 0.2]]
    
    app.dependency_overrides[get_embedding_service] = lambda: mock_service
    
    response = client.post(
        "/embed",
        json={"texts": ["Test"]},
    )
    
    app.dependency_overrides.clear()
    
    assert response.status_code == 200
    assert response.json() == {
        "vectors": [[0.2, 0.2]]
    }