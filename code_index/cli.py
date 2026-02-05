import argparse
import json
import os
import sys

from .config import EmbeddingConfig
from .errors import EmbeddingError, GitError, IndexingError
from .indexer import init_repo_index
from .mcp_server import run as run_mcp_server
from .operations import search_repo, status_repo, update_repo
from .store import StorageError


def _write_json(payload: dict) -> None:
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _error_json(code: str, message: str, detail: str | None = None) -> None:
    _write_json({"ok": False, "error": {"code": code, "message": message, "detail": detail}})


def _resolve_persist_dir(persist_dir: str | None) -> str:
    resolved = persist_dir or os.environ.get("CODE_INDEX_PERSIST_DIR")
    if not resolved:
        raise IndexingError("CONFIG_INVALID")
    return resolved


def cmd_init(args: argparse.Namespace) -> int:
    try:
        embedding = EmbeddingConfig(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
        )
        _, summary = init_repo_index(args.repo_path, args.persist_dir, embedding)
        _write_json(
            {
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
        )
        return 0
    except Exception as exc:
        return _handle_cli_error(exc)


def cmd_search(args: argparse.Namespace) -> int:
    try:
        persist_dir = _resolve_persist_dir(args.persist_dir)
        payload = search_repo(
            args.repo_path,
            persist_dir,
            args.query,
            top_k=args.top_k,
            refresh=not args.no_refresh,
        )
        _write_json({"ok": True, "data": payload})
        return 0
    except Exception as exc:
        return _handle_cli_error(exc)


def cmd_status(args: argparse.Namespace) -> int:
    try:
        persist_dir = _resolve_persist_dir(args.persist_dir)
        payload = status_repo(args.repo_path, persist_dir)
        _write_json({"ok": True, "data": payload})
        return 0
    except Exception as exc:
        return _handle_cli_error(exc)


def cmd_update(args: argparse.Namespace) -> int:
    try:
        persist_dir = _resolve_persist_dir(args.persist_dir)
        embedding = EmbeddingConfig(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
        )
        summary = update_repo(args.repo_path, persist_dir, embedding)
        _write_json(
            {
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
        )
        return 0
    except Exception as exc:
        return _handle_cli_error(exc)


def cmd_serve(args: argparse.Namespace) -> int:
    persist_dir = _resolve_persist_dir(args.persist_dir)
    run_mcp_server(persist_dir)
    return 0


def _handle_cli_error(exc: Exception) -> int:
    if isinstance(exc, IndexingError):
        _error_json(str(exc), "indexing error")
    elif isinstance(exc, EmbeddingError):
        _error_json("EMBEDDING_ERROR", "embedding error", exc.detail)
    elif isinstance(exc, StorageError):
        _error_json("STORAGE_ERROR", "storage error", str(exc))
    elif isinstance(exc, GitError):
        _error_json("NOT_GIT_REPO", "git error", str(exc))
    else:
        _error_json("UNKNOWN", "unexpected error", str(exc))
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="code-index")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="initialize a repo index")
    init_parser.add_argument("repo_path", help="path to a git repository")
    init_parser.add_argument("--persist-dir", required=True, help="base directory for indexes")
    init_parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL")
    init_parser.add_argument("--api-key", required=True, help="OpenAI-compatible API key")
    init_parser.add_argument("--model", required=True, help="embedding model name")
    init_parser.set_defaults(func=cmd_init)

    search_parser = subparsers.add_parser("search", help="search indexed repo")
    search_parser.add_argument("repo_path", help="path to a git repository")
    search_parser.add_argument("query", help="query text")
    search_parser.add_argument("--persist-dir", help="base directory for indexes")
    search_parser.add_argument("--top-k", type=int, default=10, help="max results")
    search_parser.add_argument("--no-refresh", action="store_true", help="skip incremental refresh")
    search_parser.set_defaults(func=cmd_search)

    status_parser = subparsers.add_parser("status", help="show index status")
    status_parser.add_argument("repo_path", help="path to a git repository")
    status_parser.add_argument("--persist-dir", help="base directory for indexes")
    status_parser.set_defaults(func=cmd_status)

    update_parser = subparsers.add_parser("update", help="update config and rebuild index")
    update_parser.add_argument("repo_path", help="path to a git repository")
    update_parser.add_argument("--persist-dir", help="base directory for indexes")
    update_parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL")
    update_parser.add_argument("--api-key", required=True, help="OpenAI-compatible API key")
    update_parser.add_argument("--model", required=True, help="embedding model name")
    update_parser.set_defaults(func=cmd_update)

    serve_parser = subparsers.add_parser("serve", help="start MCP server")
    serve_parser.add_argument("--persist-dir", help="base directory for indexes")
    serve_parser.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
