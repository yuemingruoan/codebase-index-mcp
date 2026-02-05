import pytest

from code_index.config import MilvusConfig
from code_index.store import VectorRecord, VectorStore


pymilvus = pytest.importorskip("pymilvus")


def test_vector_store_roundtrip(tmp_path):
    config = MilvusConfig(
        uri=str(tmp_path / "milvus.db"),
        collection="code_index_test",
        dimension=None,
    )
    store = VectorStore(config)
    store.ensure_collection(2)
    inserted = store.insert(
        [VectorRecord(embedding=[0.1, 0.2], path="src/app.py", line_start=1, line_end=2, file_hash="hash")]
    )
    assert inserted == 1
    results = store.search([0.1, 0.2], top_k=1)
    assert results
    assert results[0]["path"] == "src/app.py"
    deleted = store.delete_by_paths(["src/app.py"])
    assert deleted >= 1
