from sentence_transformers import SentenceTransformer

from app.core.config import settings

model = SentenceTransformer(settings.embedding_model)
