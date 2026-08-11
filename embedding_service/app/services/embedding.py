from app.exceptions.embedding import EmbeddingError
from app.model import model


class EmbeddingService:
    
    def __init__(self, embedding_model=None):
        self.model = embedding_model or model

    def embed(self, texts: list[str]) -> list[list[float]]:    
        try:
            vectors = self.model.encode(texts)
        except Exception as error:
            raise EmbeddingError(
                "Failed to generate embeddings."
            ) from error
        return [vector.tolist() for vector in vectors]