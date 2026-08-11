import httpx


class EmbeddingClient:
    def __init__(self, base_url: str) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/")
        )
        
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []   
        
        response = await self._client.post(
            "/embed",
            json={"texts": texts},
        )
        response.raise_for_status()
        
        data = response.json()
        return data["vectors"]
    
    async def close(self) -> None:
        await self._client.aclose()