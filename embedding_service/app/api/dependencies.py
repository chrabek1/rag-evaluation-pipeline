from app.services.embedding import EmbeddingService

def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()