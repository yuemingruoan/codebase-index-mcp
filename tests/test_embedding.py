import json

import httpx
import pytest

from code_index.config import EmbeddingConfig
from code_index.embedding import EmbeddingClient
from code_index.errors import EmbeddingError


def test_embedding_client_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model"] == "test-model"
        assert payload["input"] == ["hello", "world"]
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"embedding": [0.1, 0.2]},
                    {"embedding": [0.3, 0.4]},
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://example")
    emb = EmbeddingClient(
        EmbeddingConfig(base_url="http://example", api_key="test-key", model="test-model"),
        http_client=client,
    )
    result = emb.embed_texts(["hello", "world"])
    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_embedding_client_error():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://example")
    emb = EmbeddingClient(
        EmbeddingConfig(base_url="http://example", api_key="test-key", model="test-model"),
        http_client=client,
    )
    with pytest.raises(EmbeddingError) as exc:
        emb.embed_texts(["fail"])
    assert exc.value.status_code == 500
