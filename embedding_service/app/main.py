from fastapi import FastAPI

from app.api.embedding import router as embedding_router
from app.schemas.embedding import ModelInfoResponse
from app.core.config import settings
from app.model import model

app = FastAPI(title="Embedding Service")

app.include_router(embedding_router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "embedding",
        "model": settings.embedding_model
    }
    
@app.get("/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    embedding_dimension = model.get_sentence_embedding_dimension()
    
    if embedding_dimension is None:
        raise RuntimeError(
            "Embedding model did not provide its embedding dimension"
        )
        
    return ModelInfoResponse(
        model=settings.embedding_model,
        embedding_dimension=embedding_dimension,
    )