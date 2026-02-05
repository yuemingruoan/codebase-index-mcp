import argparse
import json
import sys

from .config import EmbeddingConfig
from .indexer import init_repo_index


def _write_json(payload: dict) -> None:
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def cmd_init(args: argparse.Namespace) -> int:
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
