from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from .chunking import chunk_text
from .config import (
    ChunkingConfig,
    EmbeddingConfig,
    RepoConfig,
    RepoFileMeta,
    VectorConfig,
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


def _collect_chunks(
    repo_root: str, tracked_files: list[str], chunking: ChunkingConfig
) -> tuple[list[tuple[str, int, int, str, str]], dict[str, RepoFileMeta]]:
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
    return chunks, file_metas


def _embed_and_insert(
    chunks: list[tuple[str, int, int, str, str]],
    embedding_client: EmbeddingClient,
    store: VectorStore,
    *,
    batch_size: int,
) -> int:
    total_inserted = 0
    for batch in _batched(chunks, batch_size):
        texts = [item[3] for item in batch]
        embeddings = embedding_client.embed_texts(texts)
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
    return total_inserted


def init_repo_index(
    repo_path: str,
    persist_dir: str,
    embedding: EmbeddingConfig,
    vector: VectorConfig,
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
    config = new_repo_config(repo_root, repo_hash, idx_dir, embedding, chunking, vector)

    tracked_files = list_tracked_files(repo_root)
    chunks, file_metas = _collect_chunks(repo_root, tracked_files, chunking)

    embedding_client = EmbeddingClient(embedding)
    store = VectorStore(idx_dir, vector)

    total_inserted = _embed_and_insert(chunks, embedding_client, store, batch_size=DEFAULT_BATCH_SIZE)
    embedding_client.close()

    config.files = file_metas
    config.chunks_indexed = total_inserted
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


def rebuild_repo_index(
    repo_path: str,
    persist_dir: str,
    embedding: EmbeddingConfig,
    vector: VectorConfig,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[RepoConfig, IndexSummary]:
    if not is_git_repo(repo_path):
        raise IndexingError("NOT_GIT_REPO")
    repo_root = get_repo_root(repo_path)
    repo_hash = hash_repo_path(repo_root)
    repo_root = normalize_repo_path(repo_root)
    idx_dir = index_dir(persist_dir, repo_hash)
    if not os.path.exists(idx_dir):
        raise IndexingError("NOT_INITIALIZED")
    config = load_repo_config(idx_dir)
    store = VectorStore(idx_dir, vector)
    store.drop()

    config.embedding = embedding
    config.vector = vector
    tracked_files = list_tracked_files(repo_root)
    chunks, file_metas = _collect_chunks(repo_root, tracked_files, config.chunking)

    embedding_client = EmbeddingClient(embedding)
    total_inserted = _embed_and_insert(chunks, embedding_client, store, batch_size=batch_size)
    embedding_client.close()

    config.files = file_metas
    config.chunks_indexed = total_inserted
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

    store = VectorStore(idx_dir, config.vector)
    deleted_count = 0
    if to_delete:
        deleted_count = store.delete_by_paths(to_delete)
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
    for batch in _batched(chunks, batch_size):
        texts = [item[3] for item in batch]
        embeddings = embedding_client.embed_texts(texts)
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

    config.chunks_indexed = max(config.chunks_indexed - deleted_count, 0) + total_inserted
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
