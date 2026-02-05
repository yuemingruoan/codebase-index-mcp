from __future__ import annotations

import os

from .config import EmbeddingConfig, load_repo_config
from .errors import IndexingError
from .indexer import IndexSummary, incremental_update, rebuild_repo_index
from .paths import hash_repo_path, index_dir, normalize_repo_path, repo_config_path
from .embedding import EmbeddingClient
from .git_utils import get_repo_root, is_git_repo
from .store import VectorStore


def _load_config(repo_path: str, persist_dir: str):
    if not is_git_repo(repo_path):
        raise IndexingError("NOT_GIT_REPO")
    repo_root = normalize_repo_path(get_repo_root(repo_path))
    repo_hash = hash_repo_path(repo_root)
    idx_dir = index_dir(persist_dir, repo_hash)
    config_path = repo_config_path(idx_dir)
    if not os.path.exists(config_path):
        raise IndexingError("NOT_INITIALIZED")
    config = load_repo_config(idx_dir)
    return repo_root, repo_hash, idx_dir, config, config_path


def status_repo(repo_path: str, persist_dir: str) -> dict:
    repo_root, repo_hash, idx_dir, config, config_path = _load_config(repo_path, persist_dir)
    return {
        "repo_root": repo_root,
        "repo_hash": repo_hash,
        "index_dir": idx_dir,
        "config_path": config_path,
        "files_indexed": len(config.files),
        "chunks_indexed": config.chunks_indexed,
        "embedding": {"model": config.embedding.model},
        "chunking": config.chunking.to_dict(),
        "last_indexed_at": config.last_indexed_at,
        "last_indexed_commit": config.last_indexed_commit,
    }


def search_repo(
    repo_path: str,
    persist_dir: str,
    query: str,
    *,
    top_k: int = 10,
    refresh: bool = True,
) -> dict:
    if refresh:
        config, _ = incremental_update(repo_path, persist_dir)
    else:
        _, _, _, config, _ = _load_config(repo_path, persist_dir)
    if config.milvus.dimension is None:
        raise IndexingError("CONFIG_INVALID")
    embedding_client = EmbeddingClient(config.embedding)
    vector = embedding_client.embed_text(query)
    embedding_client.close()
    store = VectorStore(config.milvus)
    store.ensure_collection(config.milvus.dimension)
    results = store.search(vector, top_k=top_k)
    return {"results": [{"path": r["path"], "line_start": r["line_start"], "line_end": r["line_end"]} for r in results]}


def update_repo(
    repo_path: str,
    persist_dir: str,
    embedding: EmbeddingConfig,
) -> IndexSummary:
    _, summary = rebuild_repo_index(repo_path, persist_dir, embedding)
    return summary
