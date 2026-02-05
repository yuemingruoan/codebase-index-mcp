import hashlib
import os


def normalize_repo_path(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def hash_repo_path(path: str) -> str:
    normalized = normalize_repo_path(path)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def index_dir(persist_dir: str, repo_hash: str) -> str:
    return os.path.join(persist_dir, repo_hash)


def repo_config_path(index_dir_path: str) -> str:
    return os.path.join(index_dir_path, "config.json")


def server_config_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, "server.json")
