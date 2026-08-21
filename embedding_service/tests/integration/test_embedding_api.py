from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app import main as main_module
from app.exceptions.embedding import EmbeddingError
from app.main import app
from app.api.dependencies import get_embedding_service

client = TestClient(app)


def test_health_returns_service_status():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "embedding",
        "model": main_module.settings.embedding_model,
    }

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

def test_model_info_returns_model_name_and_embedding_dimension():
    with patch.object(
        main_module.model,
        "get_sentence_embedding_dimension",
        return_value=1024,
    ):
        response = client.get("/info")

    assert response.status_code == 200
    assert response.json() == {
        "model": main_module.settings.embedding_model,
        "embedding_dimension": 1024,
    }


def test_embed_rejects_empty_text_list():
    response = client.post(
        "/embed",
        json={"texts": []},
    )

    assert response.status_code == 422


def test_embed_rejects_empty_text():
    response = client.post(
        "/embed",
        json={"texts": ["Valid text", " "]},
    )

    assert response.status_code == 422


def test_embed_rejects_invalid_texts_type():
    response = client.post(
        "/embed",
        json={"texts": "not-a-list"},
    )

    assert response.status_code == 422


def test_embed_returns_500_for_embedding_error():
    mock_service = Mock()
    mock_service.embed.side_effect = EmbeddingError(
        "Failed to generate embeddings."
    )
    app.dependency_overrides[get_embedding_service] = (
        lambda: mock_service
    )

    try:
        response = client.post(
            "/embed",
            json={"texts": ["Test"]},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Failed to generate embeddings."
    }
