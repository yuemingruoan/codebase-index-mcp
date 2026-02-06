from code_index.config import (
    ApproxConfig,
    ChunkingConfig,
    EmbeddingConfig,
    RepoConfig,
    RepoFileMeta,
    VectorConfig,
)
from code_index.indexer import compute_incremental_plan


def test_compute_incremental_plan(tmp_path):
    repo_root = tmp_path
    (repo_root / "a.py").write_text("print('hi')\n")
    (repo_root / "new.py").write_text("x = 1\n")
    (repo_root / "b.bin").write_bytes(b"\x00\x01\x02")

    config = RepoConfig(
        version=1,
        repo_root=str(repo_root),
        repo_hash="hash",
        index_dir=str(repo_root),
        embedding=EmbeddingConfig(base_url="http://example", api_key="k", model="m"),
        chunking=ChunkingConfig(chunk_lines=10, overlap_lines=0),
        vector=VectorConfig(
            device="auto",
            metric="ip",
            search_mode="exact",
            approx=ApproxConfig(sample_rate=1.0),
            max_vram_mb=None,
        ),
        files={
            "a.py": RepoFileMeta(hash="old", line_count=1),
            "removed.py": RepoFileMeta(hash="gone", line_count=1),
        },
    )

    plan = compute_incremental_plan(str(repo_root), ["a.py", "new.py", "b.bin"], config)
    assert set(plan.new_or_changed) == {"a.py", "new.py"}
    assert plan.removed == ["removed.py"]
    assert plan.skipped_binary == ["b.bin"]
