from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    text: str
    line_start: int
    line_end: int


def chunk_text(text: str, chunk_lines: int, overlap_lines: int) -> list[TextChunk]:
    if chunk_lines <= 0:
        raise ValueError("chunk_lines must be positive")
    if overlap_lines < 0:
        raise ValueError("overlap_lines must be >= 0")
    if overlap_lines >= chunk_lines:
        overlap_lines = max(chunk_lines - 1, 0)
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    chunks: list[TextChunk] = []
    start = 0
    total = len(lines)
    while start < total:
        end = min(start + chunk_lines, total)
        chunk_text = "".join(lines[start:end])
        chunks.append(TextChunk(text=chunk_text, line_start=start + 1, line_end=end))
        if end >= total:
            break
        start = end - overlap_lines if overlap_lines else end
    return chunks
