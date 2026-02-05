import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .paths import repo_config_path, server_config_path


SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class EmbeddingConfig:
    base_url: str
    api_key: str
    model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "EmbeddingConfig":
        return EmbeddingConfig(
            base_url=str(data["base_url"]),
            api_key=str(data["api_key"]),
            model=str(data["model"]),
        )


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_lines: int
    overlap_lines: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_lines": self.chunk_lines,
            "overlap_lines": self.overlap_lines,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ChunkingConfig":
        return ChunkingConfig(
            chunk_lines=int(data["chunk_lines"]),
            overlap_lines=int(data["overlap_lines"]),
        )


@dataclass(frozen=True)
class MilvusConfig:
    uri: str
    collection: str
    dimension: int | None
    metric_type: str = "IP"

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "collection": self.collection,
            "dimension": self.dimension,
            "metric_type": self.metric_type,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "MilvusConfig":
        dimension = data.get("dimension")
        return MilvusConfig(
            uri=str(data["uri"]),
            collection=str(data["collection"]),
            dimension=None if dimension is None else int(dimension),
            metric_type=str(data.get("metric_type", "IP")),
        )


@dataclass(frozen=True)
class RepoFileMeta:
    hash: str
    line_count: int

    def to_dict(self) -> dict[str, Any]:
        return {"hash": self.hash, "line_count": self.line_count}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "RepoFileMeta":
        return RepoFileMeta(hash=str(data["hash"]), line_count=int(data["line_count"]))


@dataclass
class RepoConfig:
    version: int
    repo_root: str
    repo_hash: str
    index_dir: str
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    milvus: MilvusConfig
    files: dict[str, RepoFileMeta] = field(default_factory=dict)
    last_indexed_at: str | None = None
    last_indexed_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "repo_root": self.repo_root,
            "repo_hash": self.repo_hash,
            "index_dir": self.index_dir,
            "embedding": self.embedding.to_dict(),
            "chunking": self.chunking.to_dict(),
            "milvus": self.milvus.to_dict(),
            "files": {path: meta.to_dict() for path, meta in self.files.items()},
            "last_indexed_at": self.last_indexed_at,
            "last_indexed_commit": self.last_indexed_commit,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "RepoConfig":
        files = data.get("files", {})
        return RepoConfig(
            version=int(data["version"]),
            repo_root=str(data["repo_root"]),
            repo_hash=str(data["repo_hash"]),
            index_dir=str(data["index_dir"]),
            embedding=EmbeddingConfig.from_dict(data["embedding"]),
            chunking=ChunkingConfig.from_dict(data["chunking"]),
            milvus=MilvusConfig.from_dict(data["milvus"]),
            files={path: RepoFileMeta.from_dict(meta) for path, meta in files.items()},
            last_indexed_at=data.get("last_indexed_at"),
            last_indexed_commit=data.get("last_indexed_commit"),
        )


@dataclass
class ServerConfig:
    version: int
    persist_dir: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "persist_dir": self.persist_dir,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ServerConfig":
        return ServerConfig(
            version=int(data["version"]),
            persist_dir=str(data["persist_dir"]),
            created_at=str(data["created_at"]),
        )


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_repo_config(index_dir: str) -> RepoConfig:
    path = repo_config_path(index_dir)
    return RepoConfig.from_dict(_read_json(path))


def save_repo_config(config: RepoConfig) -> str:
    path = repo_config_path(config.index_dir)
    _write_json(path, config.to_dict())
    return path


def load_server_config(persist_dir: str) -> ServerConfig | None:
    path = server_config_path(persist_dir)
    if not os.path.exists(path):
        return None
    return ServerConfig.from_dict(_read_json(path))


def ensure_server_config(persist_dir: str) -> ServerConfig:
    existing = load_server_config(persist_dir)
    if existing is not None:
        return existing
    created = ServerConfig(
        version=SCHEMA_VERSION,
        persist_dir=persist_dir,
        created_at=_utc_now_iso(),
    )
    _write_json(server_config_path(persist_dir), created.to_dict())
    return created


def new_repo_config(
    repo_root: str,
    repo_hash: str,
    index_dir: str,
    embedding: EmbeddingConfig,
    chunking: ChunkingConfig,
    milvus: MilvusConfig,
) -> RepoConfig:
    return RepoConfig(
        version=SCHEMA_VERSION,
        repo_root=repo_root,
        repo_hash=repo_hash,
        index_dir=index_dir,
        embedding=embedding,
        chunking=chunking,
        milvus=milvus,
        files={},
        last_indexed_at=_utc_now_iso(),
        last_indexed_commit=None,
    )
