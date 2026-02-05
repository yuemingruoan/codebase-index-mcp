import os

from code_index.config import (
    ChunkingConfig,
    EmbeddingConfig,
    MilvusConfig,
    RepoConfig,
    RepoFileMeta,
    load_repo_config,
    save_repo_config,
)
from code_index.paths import hash_repo_path, index_dir, repo_config_path, server_config_path


def test_hash_repo_path_stable():
    path = os.path.join(os.sep, "tmp", "repo")
    first = hash_repo_path(path)
    second = hash_repo_path(path)
    assert first == second
    assert len(first) == 64


def test_paths_helpers():
    idx_dir = index_dir("/persist", "abcd")
    assert idx_dir.endswith(os.path.join("persist", "abcd"))
    assert repo_config_path(idx_dir).endswith(os.path.join("persist", "abcd", "config.json"))
    assert server_config_path("/persist").endswith(os.path.join("persist", "server.json"))


def test_repo_config_roundtrip(tmp_path):
    embedding = EmbeddingConfig(base_url="http://example", api_key="k", model="m")
    chunking = ChunkingConfig(chunk_lines=80, overlap_lines=10)
    milvus = MilvusConfig(uri=str(tmp_path / "milvus.db"), collection="code_index", dimension=None)
    config = RepoConfig(
        version=1,
        repo_root="/repo",
        repo_hash="hash",
        index_dir=str(tmp_path),
        embedding=embedding,
        chunking=chunking,
        milvus=milvus,
        files={"src/app.py": RepoFileMeta(hash="abc", line_count=12)},
        chunks_indexed=7,
        last_indexed_at="2024-01-01T00:00:00Z",
        last_indexed_commit="deadbeef",
    )
    save_repo_config(config)
    loaded = load_repo_config(str(tmp_path))
    assert loaded.repo_root == config.repo_root
    assert loaded.embedding.model == config.embedding.model
    assert loaded.chunking.chunk_lines == config.chunking.chunk_lines
    assert loaded.files["src/app.py"].hash == "abc"
    assert loaded.chunks_indexed == 7
