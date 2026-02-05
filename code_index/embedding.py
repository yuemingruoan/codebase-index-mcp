from __future__ import annotations

from typing import Iterable

import httpx

from .config import EmbeddingConfig
from .errors import EmbeddingError


class EmbeddingClient:
    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        timeout: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._client = http_client or httpx.Client(
            base_url=config.base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
        )
        self._owns_client = http_client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        payload = {"model": self._config.model, "input": list(texts)}
        if not payload["input"]:
            return []
        response = self._client.post("/v1/embeddings", json=payload)
        if response.status_code != 200:
            raise EmbeddingError(
                "Embedding request failed",
                status_code=response.status_code,
                detail=response.text,
            )
        data = response.json()
        items = data.get("data", [])
        embeddings = [item.get("embedding") for item in items]
        if len(embeddings) != len(payload["input"]):
            raise EmbeddingError(
                "Embedding response length mismatch",
                detail=f"expected {len(payload['input'])}, got {len(embeddings)}",
            )
        if any(vec is None for vec in embeddings):
            raise EmbeddingError("Embedding response missing vectors")
        return [list(map(float, vec)) for vec in embeddings]

    def embed_text(self, text: str) -> list[float]:
        embeddings = self.embed_texts([text])
        return embeddings[0] if embeddings else []
