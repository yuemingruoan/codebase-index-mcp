# codebase-index-mcp

[English](#english) | [中文](#中文)

## English

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

## 中文

使用向量相似度驱动的代码库索引 MCP 服务，基于 OpenAI 兼容的 embedding，并提供支持 CUDA/MPS 加速的 Torch 本地向量存储。

## 功能
- MCP 工具：`init`、`search`、`status`、`update`
- OpenAI API 兼容 embedding（`base_url`、`api_key`、`model`）
- Torch 本地存储（可用时启用 CUDA/MPS，加速不可用时回退到 CPU）
- 仅索引 Git 跟踪文件，过滤二进制
- 按行切分并支持重叠
- `search` 可选增量刷新
- 支持设备、检索模式与显存预算配置

## 环境要求
- Python >= 3.12
- Git
- OpenAI 兼容的 embedding 接口
- Torch（CPU 或 GPU 版本）

## 安装
```bash
python3 -m pip install -e .
```

## CLI 用法
初始化仓库索引：
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

搜索（默认会做增量刷新）：
```bash
code-index search /abs/path/to/repo "query text" --persist-dir /abs/path/to/persist
```

跳过刷新：
```bash
code-index search /abs/path/to/repo "query text" --persist-dir /abs/path/to/persist --no-refresh
```

单次查询覆盖设备/检索参数：
```bash
code-index search /abs/path/to/repo "query text" \
  --persist-dir /abs/path/to/persist \
  --device mps \
  --search-mode approx \
  --approx-sample-rate 0.3 \
  --max-vram-mb 1024
```

状态查看：
```bash
code-index status /abs/path/to/repo --persist-dir /abs/path/to/persist
```

更新 embedding 配置（全量重建）：
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

可通过 `CODE_INDEX_PERSIST_DIR` 避免每次传入 `--persist-dir`。
通过 `CODE_INDEX_MAX_VRAM_MB` 设置默认显存预算。

说明：
- `--device auto` 优先选 CUDA，其次 MPS，再回退 CPU；显式 CUDA/MPS 不可用时会回退 CPU。
- `--search-mode approx` 通过 `--approx-sample-rate` 采样候选（值越小越快但召回更低）。
- `--metric` 支持 `ip`（默认）或 `l2`。

## MCP Server
启动 MCP 服务：
```bash
code-index serve --persist-dir /abs/path/to/persist
```

工具：
- `init`：创建新索引
- `search`：语义检索（返回相对路径与行号范围）
- `status`：索引状态
- `update`：更新配置并重建索引

## 持久化目录结构
```
/persist
  server.json
  <repo_hash>/
    config.json
    vectors/
      embeddings.pt
      meta.json
```

## 测试
```bash
python3 -m pytest -q
```

## 故障排查
- `NOT_GIT_REPO`：确保 `repo_path` 在 git 仓库内
- `NOT_INITIALIZED`：先执行 `code-index init`
- `EMBEDDING_ERROR`：检查 `base_url`、`api_key` 与网络访问
- `STORAGE_ERROR`：确认已安装 `torch` 且向量目录可写
