# codebase-index-mcp

Codebase indexing MCP server powered by vector similarity. Uses OpenAI-compatible embeddings and a Torch-based local vector store with CUDA/MPS acceleration.

## Features
- MCP tools: `init`, `search`, `status`, `update`
- OpenAI API compatible embeddings (`base_url`, `api_key`, `model`)
- Torch-based local storage (CUDA/MPS when available, CPU fallback)
- Git tracked files only; binary files filtered
- Line-based chunking with overlap
- Incremental refresh on `search` (optional)
- Configurable device, search mode, and max VRAM budget

## Requirements
- Python >= 3.12
- Git
- OpenAI-compatible embedding endpoint
- Torch (CPU or GPU build)

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
  --model text-embedding-3-small \
  --device auto \
  --search-mode exact \
  --metric ip \
  --approx-sample-rate 0.2 \
  --max-vram-mb 2048
```

Search (default incremental refresh):
```bash
code-index search /abs/path/to/repo "query text" --persist-dir /abs/path/to/persist
```

Skip refresh:
```bash
code-index search /abs/path/to/repo "query text" --persist-dir /abs/path/to/persist --no-refresh
```

Override device/search settings for one query:
```bash
code-index search /abs/path/to/repo "query text" \
  --persist-dir /abs/path/to/persist \
  --device mps \
  --search-mode approx \
  --approx-sample-rate 0.3 \
  --max-vram-mb 1024
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
  --model text-embedding-3-small \
  --device auto \
  --search-mode exact \
  --metric ip \
  --approx-sample-rate 0.2 \
  --max-vram-mb 2048
```

You can also set `CODE_INDEX_PERSIST_DIR` to avoid passing `--persist-dir` each time.
Set `CODE_INDEX_MAX_VRAM_MB` to define a default VRAM budget for searches/indexing.

Notes:
- `--device auto` selects CUDA first, then MPS, then CPU; explicit CUDA/MPS falls back to CPU if unavailable.
- `--search-mode approx` samples candidates based on `--approx-sample-rate` (lower is faster, lower recall).
- `--metric` supports `ip` (default) or `l2`.

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
      embeddings.pt
      meta.json
```

## Tests
```bash
python3 -m pytest -q
```

## Troubleshooting
- `NOT_GIT_REPO`: ensure `repo_path` is inside a git repository
- `NOT_INITIALIZED`: run `code-index init` first
- `EMBEDDING_ERROR`: check `base_url`, `api_key`, and network access
- `STORAGE_ERROR`: ensure `torch` is installed and the vectors directory is writable
