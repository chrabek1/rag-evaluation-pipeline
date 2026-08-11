from fastapi import FastAPI

from app.api.embedding import router as embedding_router
from app.core.config import settings

app = FastAPI(title="Embedding Service")

app.include_router(embedding_router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "embedding",
        "model": settings.embedding_model
    }