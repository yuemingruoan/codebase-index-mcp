import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch required for vector store tests",
)

from code_index.config import ApproxConfig, VectorConfig
from code_index.store import VectorRecord, VectorStore


def _vector_config(**kwargs):
    return VectorConfig(
        device=kwargs.get("device", "cpu"),
        metric=kwargs.get("metric", "ip"),
        search_mode=kwargs.get("search_mode", "exact"),
        approx=kwargs.get("approx", ApproxConfig(sample_rate=1.0)),
        max_vram_mb=kwargs.get("max_vram_mb", 64),
    )


def test_vector_store_roundtrip(tmp_path):
    store = VectorStore(str(tmp_path), _vector_config())
    inserted = store.insert(
        [
            VectorRecord(
                embedding=[1.0, 0.0],
                path="a.py",
                line_start=1,
                line_end=2,
                file_hash="hash-a",
            ),
            VectorRecord(
                embedding=[0.0, 1.0],
                path="b.py",
                line_start=3,
                line_end=4,
                file_hash="hash-b",
            ),
        ]
    )
    assert inserted == 2
    results = store.search([1.0, 0.0], top_k=1)
    assert results
    assert results[0]["path"] == "a.py"
    deleted = store.delete_by_paths(["a.py"])
    assert deleted == 1


def test_vector_store_approx_sampling(tmp_path):
    config = _vector_config(search_mode="approx", approx=ApproxConfig(sample_rate=0.5))
    store = VectorStore(str(tmp_path), config)
    store.insert(
        [
            VectorRecord(
                embedding=[1.0, 0.0],
                path="a.py",
                line_start=1,
                line_end=2,
                file_hash="hash-a",
            ),
            VectorRecord(
                embedding=[0.0, 1.0],
                path="b.py",
                line_start=3,
                line_end=4,
                file_hash="hash-b",
            ),
            VectorRecord(
                embedding=[0.5, 0.5],
                path="c.py",
                line_start=5,
                line_end=6,
                file_hash="hash-c",
            ),
            VectorRecord(
                embedding=[-1.0, 0.0],
                path="d.py",
                line_start=7,
                line_end=8,
                file_hash="hash-d",
            ),
        ]
    )
    results = store.search([0.0, 1.0], top_k=1)
    assert results
    assert results[0]["path"] in {"a.py", "c.py"}


def test_vector_store_respects_max_vram_mb(tmp_path):
    store = VectorStore(str(tmp_path), _vector_config(max_vram_mb=1))
    records = []
    for idx in range(50):
        records.append(
            VectorRecord(
                embedding=[float(idx), float(idx + 1)],
                path=f"file-{idx}.py",
                line_start=1,
                line_end=2,
                file_hash=f"hash-{idx}",
            )
        )
    store.insert(records)
    results = store.search([1.0, 2.0], top_k=3)
    assert len(results) == 3
