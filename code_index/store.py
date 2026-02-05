from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import MilvusConfig
from .errors import CodeIndexError

try:
    from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
    from pymilvus.milvus_client.index import IndexParams
except Exception as exc:  # pragma: no cover - exercised when pymilvus missing
    MilvusClient = None  # type: ignore[assignment]
    DataType = None  # type: ignore[assignment]
    FieldSchema = None  # type: ignore[assignment]
    CollectionSchema = None  # type: ignore[assignment]
    IndexParams = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class StorageError(CodeIndexError):
    pass


@dataclass(frozen=True)
class VectorRecord:
    embedding: list[float]
    path: str
    line_start: int
    line_end: int
    file_hash: str


class VectorStore:
    def __init__(self, config: MilvusConfig) -> None:
        if _IMPORT_ERROR is not None:
            raise StorageError(f"pymilvus is required: {_IMPORT_ERROR}")
        if MilvusClient is None:
            raise StorageError("pymilvus is required")
        self._config = config
        self._client = MilvusClient(config.uri)

    def ensure_collection(self, dimension: int) -> None:
        if self._client.has_collection(self._config.collection):
            return
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="line_start", dtype=DataType.INT64),
                FieldSchema(name="line_end", dtype=DataType.INT64),
                FieldSchema(name="file_hash", dtype=DataType.VARCHAR, max_length=64),
            ],
            description="code index vectors",
        )
        self._client.create_collection(
            collection_name=self._config.collection,
            schema=schema,
            metric_type=self._config.metric_type,
        )
        index_params = IndexParams()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type=self._config.metric_type,
        )
        self._client.create_index(
            collection_name=self._config.collection,
            index_params=index_params,
        )

    def insert(self, records: Iterable[VectorRecord]) -> int:
        payload = [
            {
                "embedding": record.embedding,
                "path": record.path,
                "line_start": record.line_start,
                "line_end": record.line_end,
                "file_hash": record.file_hash,
            }
            for record in records
        ]
        if not payload:
            return 0
        result = self._client.insert(collection_name=self._config.collection, data=payload)
        return len(result.get("ids", []))

    def search(self, embedding: list[float], top_k: int) -> list[dict]:
        self._client.load_collection(self._config.collection)
        results = self._client.search(
            collection_name=self._config.collection,
            data=[embedding],
            limit=top_k,
            output_fields=["path", "line_start", "line_end"],
        )
        hits = results[0] if results else []
        return [
            {
                "path": hit.get("path"),
                "line_start": hit.get("line_start"),
                "line_end": hit.get("line_end"),
                "score": hit.get("score"),
            }
            for hit in hits
        ]

    def delete_by_paths(self, paths: Iterable[str]) -> int:
        total_deleted = 0
        for path in paths:
            expr = f'path == "{path}"'
            result = self._client.delete(collection_name=self._config.collection, filter=expr)
            delete_count = 0
            if isinstance(result, dict):
                delete_count = int(result.get("delete_count", 0))
            elif isinstance(result, list):
                if result and all(isinstance(item, dict) for item in result):
                    delete_count = sum(int(item.get("delete_count", 0)) for item in result)
                else:
                    delete_count = len(result)
            total_deleted += delete_count
        return total_deleted

    def drop(self) -> None:
        if self._client.has_collection(self._config.collection):
            self._client.drop_collection(self._config.collection)
