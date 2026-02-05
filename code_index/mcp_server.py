from __future__ import annotations

from typing import Any

import anyio
from mcp import types
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from .config import EmbeddingConfig
from .errors import CodeIndexError, EmbeddingError, GitError, IndexingError
from .indexer import init_repo_index
from .operations import search_repo, status_repo, update_repo
from .store import StorageError


class ServerState:
    def __init__(self) -> None:
        self.persist_dir: str | None = None


state = ServerState()
server = Server("code-index")


def _error_response(code: str, message: str, detail: Any | None = None) -> dict:
    return {"ok": False, "error": {"code": code, "message": message, "detail": detail}}


def _handle_exception(exc: Exception) -> dict:
    if isinstance(exc, IndexingError):
        return _error_response(str(exc), "indexing error")
    if isinstance(exc, EmbeddingError):
        return _error_response(
            "EMBEDDING_ERROR",
            "embedding error",
            {"status": exc.status_code, "detail": exc.detail},
        )
    if isinstance(exc, StorageError):
        return _error_response("STORAGE_ERROR", "storage error", str(exc))
    if isinstance(exc, GitError):
        return _error_response("NOT_GIT_REPO", "git error", str(exc))
    if isinstance(exc, CodeIndexError):
        return _error_response("INDEX_ERROR", "index error", str(exc))
    return _error_response("UNKNOWN", "unexpected error", str(exc))


def _require_persist_dir() -> str:
    if not state.persist_dir:
        raise IndexingError("CONFIG_INVALID")
    return state.persist_dir


EMBEDDING_SCHEMA = {
    "type": "object",
    "properties": {
        "base_url": {"type": "string"},
        "api_key": {"type": "string"},
        "model": {"type": "string"},
    },
    "required": ["base_url", "api_key", "model"],
    "additionalProperties": False,
}

VECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "device": {"type": "string"},
        "search_mode": {"type": "string"},
        "metric": {"type": "string"},
        "approx_sample_rate": {"type": "number", "minimum": 0, "maximum": 1},
        "max_vram_mb": {"type": "integer", "minimum": 1},
    },
    "required": ["device", "search_mode", "metric", "approx_sample_rate"],
    "additionalProperties": False,
}

TOOLS = [
    types.Tool(
        name="init",
        description="create a new repo index",
        inputSchema={
            "type": "object",
            "properties": {
                "repo_path": {"type": "string"},
                "persist_dir": {"type": "string"},
                "embedding": EMBEDDING_SCHEMA,
                "vector": VECTOR_SCHEMA,
            },
            "required": ["repo_path", "persist_dir", "embedding"],
            "additionalProperties": False,
        },
    ),
    types.Tool(
        name="search",
        description="semantic search in indexed repo",
        inputSchema={
            "type": "object",
            "properties": {
                "repo_path": {"type": "string"},
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
                "refresh": {"type": "boolean"},
                "device": {"type": "string"},
                "search_mode": {"type": "string"},
                "approx_sample_rate": {"type": "number", "minimum": 0, "maximum": 1},
                "max_vram_mb": {"type": "integer", "minimum": 1},
            },
            "required": ["repo_path", "query"],
            "additionalProperties": False,
        },
    ),
    types.Tool(
        name="status",
        description="get index status",
        inputSchema={
            "type": "object",
            "properties": {"repo_path": {"type": "string"}},
            "required": ["repo_path"],
            "additionalProperties": False,
        },
    ),
    types.Tool(
        name="update",
        description="update config and rebuild index",
        inputSchema={
            "type": "object",
            "properties": {
                "repo_path": {"type": "string"},
                "embedding": EMBEDDING_SCHEMA,
                "vector": VECTOR_SCHEMA,
            },
            "required": ["repo_path", "embedding"],
            "additionalProperties": False,
        },
    ),
]


@server.list_tools()
async def list_tools():
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "init":
            embedding_config = EmbeddingConfig.from_dict(arguments["embedding"])
            _, summary = init_repo_index(
                arguments["repo_path"],
                arguments["persist_dir"],
                embedding_config,
            )
            return {
                "ok": True,
                "data": {
                    "repo_root": summary.repo_root,
                    "repo_hash": summary.repo_hash,
                    "index_dir": summary.index_dir,
                    "config_path": summary.config_path,
                    "files_indexed": summary.files_indexed,
                    "chunks_indexed": summary.chunks_indexed,
                },
            }
        if name == "search":
            persist_dir = _require_persist_dir()
            payload = search_repo(
                arguments["repo_path"],
                persist_dir,
                arguments["query"],
                top_k=arguments.get("top_k", 10),
                refresh=arguments.get("refresh", True),
            )
            return {"ok": True, "data": payload}
        if name == "status":
            persist_dir = _require_persist_dir()
            payload = status_repo(arguments["repo_path"], persist_dir)
            return {"ok": True, "data": payload}
        if name == "update":
            persist_dir = _require_persist_dir()
            embedding_config = EmbeddingConfig.from_dict(arguments["embedding"])
            summary = update_repo(arguments["repo_path"], persist_dir, embedding_config)
            return {
                "ok": True,
                "data": {
                    "repo_root": summary.repo_root,
                    "repo_hash": summary.repo_hash,
                    "index_dir": summary.index_dir,
                    "config_path": summary.config_path,
                    "files_indexed": summary.files_indexed,
                    "chunks_indexed": summary.chunks_indexed,
                },
            }
        return _error_response("UNKNOWN_TOOL", "unknown tool", name)
    except Exception as exc:  # pragma: no cover - error path
        return _handle_exception(exc)


async def _serve() -> None:
    capabilities = types.ServerCapabilities(tools=types.ToolsCapability(listChanged=False))
    init_opts = InitializationOptions(
        server_name="code-index",
        server_version="0.1.0",
        capabilities=capabilities,
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, initialization_options=init_opts)


def run(persist_dir: str) -> None:
    state.persist_dir = persist_dir
    anyio.run(_serve)


__all__ = ["run", "server"]
