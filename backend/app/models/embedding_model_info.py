from pydantic import BaseModel, Field


class EmbeddingModelInfo(BaseModel):
    model: str
    embedding_dimension: int = Field(gt=0)