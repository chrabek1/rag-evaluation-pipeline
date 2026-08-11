from fastapi import APIRouter, Depends, HTTPException, status

from app.exceptions.embedding import EmbeddingError
from app.schemas.embedding import EmbedRequest, EmbedResponse
from app.api.dependencies import get_embedding_service
from app.services.embedding import EmbeddingService


router = APIRouter(tags=["Embedding"])


@router.post("/embed", response_model=EmbedResponse)
def embed(
    request: EmbedRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> EmbedResponse:
    try:
        vectors = embedding_service.embed(request.texts)
    except EmbeddingError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        ) from error
        
    return EmbedResponse(vectors=vectors)