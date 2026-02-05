# codebase-index-mcp

Codebase indexing MCP server powered by vector similarity. Uses OpenAI-compatible embeddings and Milvus Lite for local persistence.

## Features
- MCP tools: `init`, `search`, `status`, `update`
- OpenAI API compatible embeddings (`base_url`, `api_key`, `model`)
- Milvus Lite local storage (no external vector DB)
- Git tracked files only; binary files filtered
- Line-based chunking with overlap
- Incremental refresh on `search` (optional)

## Requirements
- Python >= 3.12
- Git
- OpenAI-compatible embedding endpoint

## Install
```bash
python3 -m pip install -e .
```

## CLI Usage
Initialize a repo index:
```bash
code-index init /abs/path/to/repo \
  --persist-dir /abs/path/to/persist \
  --base-url https://api.openai.com \
  --api-key $OPENAI_API_KEY \
  --model text-embedding-3-small
```

Search (default incremental refresh):
```bash
code-index search /abs/path/to/repo "query text" --persist-dir /abs/path/to/persist
```

Skip refresh:
```bash
code-index search /abs/path/to/repo "query text" --persist-dir /abs/path/to/persist --no-refresh
```

Status:
```bash
code-index status /abs/path/to/repo --persist-dir /abs/path/to/persist
```

Update embedding config (full rebuild):
```bash
code-index update /abs/path/to/repo \
  --persist-dir /abs/path/to/persist \
  --base-url https://api.openai.com \
  --api-key $OPENAI_API_KEY \
  --model text-embedding-3-small
```

You can also set `CODE_INDEX_PERSIST_DIR` to avoid passing `--persist-dir` each time.

## MCP Server
Start the MCP server:
```bash
code-index serve --persist-dir /abs/path/to/persist
```

Tools:
- `init`: create a new index
- `search`: semantic search (returns relative path + line range)
- `status`: current index info
- `update`: update config and rebuild index

## Persistence Layout
```
/persist
  server.json
  <repo_hash>/
    config.json
    vectors/
      milvus.db
```

## Tests
```bash
python3 -m pytest -q
```

## Troubleshooting
- `NOT_GIT_REPO`: ensure `repo_path` is inside a git repository
- `NOT_INITIALIZED`: run `code-index init` first
- `EMBEDDING_ERROR`: check `base_url`, `api_key`, and network access
- `STORAGE_ERROR`: ensure `pymilvus` is installed and writable
