import argparse
import json
import os
import sys

from .config import ChunkingConfig, EmbeddingConfig, MilvusConfig, ensure_server_config, new_repo_config, save_repo_config
from .paths import hash_repo_path, index_dir, normalize_repo_path


DEFAULT_CHUNK_LINES = 80
DEFAULT_CHUNK_OVERLAP = 10


def _write_json(payload: dict) -> None:
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def cmd_init(args: argparse.Namespace) -> int:
    repo_root = normalize_repo_path(args.repo_path)
    repo_hash = hash_repo_path(repo_root)
    idx_dir = index_dir(args.persist_dir, repo_hash)
    os.makedirs(idx_dir, exist_ok=True)

    embedding = EmbeddingConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
    )
    chunking = ChunkingConfig(
        chunk_lines=DEFAULT_CHUNK_LINES,
        overlap_lines=DEFAULT_CHUNK_OVERLAP,
    )
    milvus = MilvusConfig(
        uri=os.path.join(idx_dir, "milvus.db"),
        collection="code_index",
        dimension=None,
    )
    ensure_server_config(args.persist_dir)
    config = new_repo_config(repo_root, repo_hash, idx_dir, embedding, chunking, milvus)
    config_path = save_repo_config(config)

    _write_json(
        {
            "ok": True,
            "data": {
                "repo_root": repo_root,
                "repo_hash": repo_hash,
                "index_dir": idx_dir,
                "config_path": config_path,
                "files_indexed": 0,
                "chunks_indexed": 0,
            },
        }
    )
    return 0


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
