from pydantic import BaseModel, Field, field_validator


class EmbedRequest(BaseModel):
    texts: list[str] = Field(
        min_length=1,
        max_length=1000,
    )
    
    @field_validator("texts")
    @classmethod
    def validate_texts(cls, texts: list[str]) -> list[str]:
        if any(not text.strip() for text in texts):
            raise ValueError("Texts cannot contain empty strings.")
        
        return texts
    
    
class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    