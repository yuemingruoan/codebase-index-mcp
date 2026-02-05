from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from .chunking import chunk_text
from .config import (
    ChunkingConfig,
    EmbeddingConfig,
    MilvusConfig,
    RepoConfig,
    RepoFileMeta,
    ensure_server_config,
    load_repo_config,
    new_repo_config,
    save_repo_config,
)
from .embedding import EmbeddingClient
from .errors import IndexingError
from .files import read_text_file, sha256_file
from .git_utils import get_head_commit, get_repo_root, is_git_repo, list_tracked_files
from .paths import hash_repo_path, index_dir, normalize_repo_path
from .store import VectorRecord, VectorStore
from .text_detect import is_text_file


DEFAULT_CHUNK_LINES = 80
DEFAULT_CHUNK_OVERLAP = 10
DEFAULT_BATCH_SIZE = 64


@dataclass(frozen=True)
class IndexSummary:
    repo_root: str
    repo_hash: str
    index_dir: str
    config_path: str
    files_indexed: int
    chunks_indexed: int


@dataclass(frozen=True)
class IncrementalPlan:
    new_or_changed: list[str]
    removed: list[str]
    skipped_binary: list[str]


@dataclass(frozen=True)
class IncrementalSummary:
    repo_root: str
    repo_hash: str
    index_dir: str
    config_path: str
    files_indexed: int
    chunks_indexed: int
    files_removed: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def choose_chunking() -> ChunkingConfig:
    return ChunkingConfig(
        chunk_lines=DEFAULT_CHUNK_LINES,
        overlap_lines=DEFAULT_CHUNK_OVERLAP,
    )


def _batched(items: list, batch_size: int) -> Iterable[list]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def init_repo_index(
    repo_path: str,
    persist_dir: str,
    embedding: EmbeddingConfig,
) -> tuple[RepoConfig, IndexSummary]:
    if not is_git_repo(repo_path):
        raise IndexingError("NOT_GIT_REPO")
    repo_root = get_repo_root(repo_path)
    repo_hash = hash_repo_path(repo_root)
    repo_root = normalize_repo_path(repo_root)
    idx_dir = index_dir(persist_dir, repo_hash)
    vectors_dir = os.path.join(idx_dir, "vectors")
    os.makedirs(vectors_dir, exist_ok=True)
    ensure_server_config(persist_dir)

    chunking = choose_chunking()
    milvus = MilvusConfig(
        uri=os.path.join(vectors_dir, "milvus.db"),
        collection="code_index",
        dimension=None,
    )
    config = new_repo_config(repo_root, repo_hash, idx_dir, embedding, chunking, milvus)

    tracked_files = list_tracked_files(repo_root)
    chunks: list[tuple[str, int, int, str, str]] = []
    file_metas: dict[str, RepoFileMeta] = {}
    for rel_path in tracked_files:
        abs_path = os.path.join(repo_root, rel_path)
        if not os.path.isfile(abs_path):
            continue
        if not is_text_file(abs_path):
            continue
        text = read_text_file(abs_path)
        file_hash = sha256_file(abs_path)
        file_chunks = chunk_text(text, chunking.chunk_lines, chunking.overlap_lines)
        if not file_chunks:
            continue
        file_metas[rel_path] = RepoFileMeta(hash=file_hash, line_count=len(text.splitlines()))
        for chunk in file_chunks:
            chunks.append((rel_path, chunk.line_start, chunk.line_end, chunk.text, file_hash))

    embedding_client = EmbeddingClient(embedding)
    store = VectorStore(milvus)

    total_inserted = 0
    dimension: int | None = None
    for batch in _batched(chunks, DEFAULT_BATCH_SIZE):
        texts = [item[3] for item in batch]
        embeddings = embedding_client.embed_texts(texts)
        if embeddings and dimension is None:
            dimension = len(embeddings[0])
            store.ensure_collection(dimension)
        records = [
            VectorRecord(
                embedding=embeddings[idx],
                path=batch[idx][0],
                line_start=batch[idx][1],
                line_end=batch[idx][2],
                file_hash=batch[idx][4],
            )
            for idx in range(len(batch))
        ]
        total_inserted += store.insert(records)
    embedding_client.close()

    config.milvus = MilvusConfig(
        uri=milvus.uri,
        collection=milvus.collection,
        dimension=dimension,
        metric_type=milvus.metric_type,
    )
    config.files = file_metas
    config.last_indexed_at = _utc_now_iso()
    config.last_indexed_commit = get_head_commit(repo_root)
    config_path = save_repo_config(config)

    summary = IndexSummary(
        repo_root=repo_root,
        repo_hash=repo_hash,
        index_dir=idx_dir,
        config_path=config_path,
        files_indexed=len(file_metas),
        chunks_indexed=total_inserted,
    )
    return config, summary


def compute_incremental_plan(
    repo_root: str, tracked_files: list[str], config: RepoConfig
) -> IncrementalPlan:
    tracked_set = set(tracked_files)
    removed = [path for path in config.files.keys() if path not in tracked_set]
    new_or_changed: list[str] = []
    skipped_binary: list[str] = []
    for rel_path in tracked_files:
        abs_path = os.path.join(repo_root, rel_path)
        if not os.path.isfile(abs_path):
            continue
        if not is_text_file(abs_path):
            if rel_path in config.files:
                removed.append(rel_path)
            else:
                skipped_binary.append(rel_path)
            continue
        file_hash = sha256_file(abs_path)
        previous = config.files.get(rel_path)
        if previous is None or previous.hash != file_hash:
            new_or_changed.append(rel_path)
    removed = sorted(set(removed))
    return IncrementalPlan(new_or_changed=new_or_changed, removed=removed, skipped_binary=skipped_binary)


def incremental_update(
    repo_path: str,
    persist_dir: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[RepoConfig, IncrementalSummary]:
    if not is_git_repo(repo_path):
        raise IndexingError("NOT_GIT_REPO")
    repo_root = get_repo_root(repo_path)
    repo_hash = hash_repo_path(repo_root)
    repo_root = normalize_repo_path(repo_root)
    idx_dir = index_dir(persist_dir, repo_hash)
    if not os.path.exists(idx_dir):
        raise IndexingError("NOT_INITIALIZED")
    config = load_repo_config(idx_dir)

    tracked_files = list_tracked_files(repo_root)
    plan = compute_incremental_plan(repo_root, tracked_files, config)
    to_delete = sorted(set(plan.removed + plan.new_or_changed))

    store = VectorStore(config.milvus)
    if config.milvus.dimension is not None:
        store.ensure_collection(config.milvus.dimension)
    if to_delete:
        store.delete_by_paths(to_delete)
        for path in to_delete:
            config.files.pop(path, None)

    chunks: list[tuple[str, int, int, str, str]] = []
    for rel_path in plan.new_or_changed:
        abs_path = os.path.join(repo_root, rel_path)
        if not os.path.isfile(abs_path):
            continue
        if not is_text_file(abs_path):
            continue
        text = read_text_file(abs_path)
        file_hash = sha256_file(abs_path)
        file_chunks = chunk_text(text, config.chunking.chunk_lines, config.chunking.overlap_lines)
        if not file_chunks:
            continue
        config.files[rel_path] = RepoFileMeta(hash=file_hash, line_count=len(text.splitlines()))
        for chunk in file_chunks:
            chunks.append((rel_path, chunk.line_start, chunk.line_end, chunk.text, file_hash))

    embedding_client = EmbeddingClient(config.embedding)
    total_inserted = 0
    dimension = config.milvus.dimension
    for batch in _batched(chunks, batch_size):
        texts = [item[3] for item in batch]
        embeddings = embedding_client.embed_texts(texts)
        if embeddings and dimension is None:
            dimension = len(embeddings[0])
            store.ensure_collection(dimension)
        records = [
            VectorRecord(
                embedding=embeddings[idx],
                path=batch[idx][0],
                line_start=batch[idx][1],
                line_end=batch[idx][2],
                file_hash=batch[idx][4],
            )
            for idx in range(len(batch))
        ]
        total_inserted += store.insert(records)
    embedding_client.close()

    if dimension != config.milvus.dimension:
        config.milvus = MilvusConfig(
            uri=config.milvus.uri,
            collection=config.milvus.collection,
            dimension=dimension,
            metric_type=config.milvus.metric_type,
        )
    config.last_indexed_at = _utc_now_iso()
    config.last_indexed_commit = get_head_commit(repo_root)
    config_path = save_repo_config(config)

    summary = IncrementalSummary(
        repo_root=repo_root,
        repo_hash=repo_hash,
        index_dir=idx_dir,
        config_path=config_path,
        files_indexed=len(plan.new_or_changed),
        chunks_indexed=total_inserted,
        files_removed=len(plan.removed),
    )
    return config, summary
