from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable

from .config import VectorConfig
from .device import resolve_device
from .errors import CodeIndexError

try:
    import torch
except Exception as exc:  # pragma: no cover - exercised when torch missing
    torch = None  # type: ignore[assignment]
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
    def __init__(self, index_dir: str, config: VectorConfig) -> None:
        if _IMPORT_ERROR is not None or torch is None:
            raise StorageError(f"torch is required: {_IMPORT_ERROR}")
        self._index_dir = index_dir
        self._config = config
        self._vectors_dir = os.path.join(index_dir, "vectors")
        self._embeddings_path = os.path.join(self._vectors_dir, "embeddings.pt")
        self._meta_path = os.path.join(self._vectors_dir, "meta.json")
        self._embeddings: torch.Tensor | None = None
        self._records: list[dict] | None = None
        self._dimension: int | None = None
        os.makedirs(self._vectors_dir, exist_ok=True)

    def _load(self) -> None:
        if self._records is not None:
            return
        records: list[dict] = []
        dimension: int | None = None
        if os.path.exists(self._meta_path):
            with open(self._meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            dimension = meta.get("dimension")
            if dimension is not None:
                dimension = int(dimension)
            records = list(meta.get("records", []))
        embeddings = None
        if os.path.exists(self._embeddings_path):
            embeddings = torch.load(self._embeddings_path, map_location="cpu")
            if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
                raise StorageError("embeddings file invalid")
            embeddings = embeddings.to(dtype=torch.float32, device="cpu")
            if dimension is None and embeddings.numel() > 0:
                dimension = int(embeddings.shape[1])
        if embeddings is not None and records and embeddings.shape[0] != len(records):
            raise StorageError("embeddings/meta length mismatch")
        self._records = records
        self._embeddings = embeddings
        self._dimension = dimension

    def _save(self) -> None:
        if self._records is None:
            return
        payload = {
            "version": 1,
            "dimension": self._dimension,
            "records": self._records,
        }
        with open(self._meta_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if self._embeddings is None or self._embeddings.numel() == 0:
            if os.path.exists(self._embeddings_path):
                os.remove(self._embeddings_path)
            return
        torch.save(self._embeddings, self._embeddings_path)

    def _ensure_loaded(self) -> None:
        self._load()

    def _sample_indices(self, total: int, sample_rate: float) -> list[int]:
        if total <= 0:
            return []
        if sample_rate >= 1:
            return list(range(total))
        step = max(int(1 / sample_rate), 1)
        return list(range(0, total, step))

    def _chunk_size(self, total: int, dim: int, max_vram_mb: int | None) -> int:
        if total <= 0:
            return 0
        if max_vram_mb is None:
            return total
        bytes_per_vector = dim * 4
        if bytes_per_vector <= 0:
            return total
        budget = max_vram_mb * 1024 * 1024
        max_vectors = max(int(budget / (bytes_per_vector * 1.2)), 1)
        return min(total, max_vectors)

    def insert(self, records: Iterable[VectorRecord]) -> int:
        batch = list(records)
        if not batch:
            return 0
        self._ensure_loaded()
        embeddings = torch.tensor([record.embedding for record in batch], dtype=torch.float32)
        if embeddings.ndim != 2:
            raise StorageError("embeddings must be 2D")
        dimension = int(embeddings.shape[1])
        if self._dimension is None:
            self._dimension = dimension
        elif dimension != self._dimension:
            raise StorageError("embedding dimension mismatch")
        if self._embeddings is None or self._embeddings.numel() == 0:
            self._embeddings = embeddings
        else:
            self._embeddings = torch.cat([self._embeddings, embeddings], dim=0)
        if self._records is None:
            self._records = []
        self._records.extend(
            {
                "path": record.path,
                "line_start": record.line_start,
                "line_end": record.line_end,
                "file_hash": record.file_hash,
            }
            for record in batch
        )
        self._save()
        return len(batch)

    def search(
        self,
        embedding: list[float],
        top_k: int,
        *,
        device: str | None = None,
        search_mode: str | None = None,
        approx_sample_rate: float | None = None,
        max_vram_mb: int | None = None,
        metric: str | None = None,
    ) -> list[dict]:
        self._ensure_loaded()
        if not self._records or self._embeddings is None or self._embeddings.numel() == 0:
            return []
        query = torch.tensor(embedding, dtype=torch.float32)
        if query.ndim != 1:
            raise StorageError("query embedding must be 1D")
        dimension = int(self._embeddings.shape[1])
        if query.numel() != dimension:
            raise StorageError("query embedding dimension mismatch")
        metric = (metric or self._config.metric).lower()
        if metric not in {"ip", "l2"}:
            raise StorageError("metric must be ip or l2")
        search_mode = (search_mode or self._config.search_mode).lower()
        if search_mode not in {"exact", "approx"}:
            raise StorageError("search_mode must be exact or approx")
        if search_mode == "approx":
            sample_rate = approx_sample_rate if approx_sample_rate is not None else self._config.approx.sample_rate
            if not 0 < sample_rate <= 1:
                raise StorageError("approx.sample_rate must be in (0, 1]")
            indices = self._sample_indices(len(self._records), sample_rate)
        else:
            indices = list(range(len(self._records)))
        if not indices:
            return []
        resolved_device = resolve_device(device or self._config.device)
        max_vram_mb = max_vram_mb if max_vram_mb is not None else self._config.max_vram_mb
        chunk_size = self._chunk_size(len(indices), dimension, max_vram_mb)
        if chunk_size <= 0:
            return []
        query = query.to(resolved_device)
        best: list[tuple[float, int]] = []
        with torch.inference_mode():
            for start in range(0, len(indices), chunk_size):
                chunk_indices = indices[start : start + chunk_size]
                chunk = self._embeddings[chunk_indices].to(resolved_device)
                if metric == "ip":
                    scores = chunk @ query
                else:
                    diff = chunk - query
                    scores = -(diff * diff).sum(dim=1)
                k = min(top_k, int(scores.shape[0]))
                if k == 0:
                    continue
                chunk_scores, local_idx = torch.topk(scores, k=k)
                chunk_scores = chunk_scores.detach().cpu().tolist()
                local_idx = local_idx.detach().cpu().tolist()
                for score, rel in zip(chunk_scores, local_idx):
                    best.append((float(score), chunk_indices[rel]))
                best = sorted(best, key=lambda item: item[0], reverse=True)[:top_k]
        return [
            {
                "path": self._records[idx]["path"],
                "line_start": self._records[idx]["line_start"],
                "line_end": self._records[idx]["line_end"],
                "score": score,
            }
            for score, idx in best
        ]

    def delete_by_paths(self, paths: Iterable[str]) -> int:
        self._ensure_loaded()
        if not self._records:
            return 0
        remove = set(paths)
        keep_indices: list[int] = []
        new_records: list[dict] = []
        for idx, record in enumerate(self._records):
            if record.get("path") in remove:
                continue
            keep_indices.append(idx)
            new_records.append(record)
        deleted = len(self._records) - len(new_records)
        if deleted == 0:
            return 0
        if self._embeddings is not None and self._embeddings.numel() > 0:
            if keep_indices:
                self._embeddings = self._embeddings[torch.tensor(keep_indices, dtype=torch.long)]
            else:
                self._embeddings = None
                self._dimension = None
        self._records = new_records
        self._save()
        return deleted

    def drop(self) -> None:
        if os.path.exists(self._embeddings_path):
            os.remove(self._embeddings_path)
        if os.path.exists(self._meta_path):
            os.remove(self._meta_path)
        self._embeddings = None
        self._records = None
        self._dimension = None
