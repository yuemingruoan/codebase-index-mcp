# 代码库索引 MCP 任务

## 标题与目标
基于 Milvus Lite 构建代码库索引 MCP Server，使用 OpenAI API 兼容 Embedding，支持 git 仓库增量更新并以 JSON 返回相对路径与行号范围。完成定义：提供 `code-index` CLI 与 MCP 工具（init/search/status/update），可在用户指定持久化目录中创建、更新和查询索引。

## 交付物
- Python MCP Server 实现与工具注册
- `code-index` CLI 入口与使用说明
- 持久化目录结构与配置 JSON（embedding 配置 + 项目索引信息 + chunk 策略）
- Milvus Lite 向量存储与元数据管理
- 测试用例与 README 更新

## 范围 / 非范围
范围:
- OpenAI API 格式 embedding（base_url/api_key/model）
- Milvus Lite 本地持久化，不依赖外部向量 DB
- 仅 git 仓库，增量更新仅对已跟踪文件
- 文本检测排除二进制文件
- 按行切分 chunk，search 返回相对路径 + 行号范围 JSON

非范围:
- 非 git 仓库索引
- UI/前端界面
- 多 provider 适配（除 OpenAI API 兼容）
- rerank、语义聚类或跨库检索
- 权限/多租户/远端服务运维

## MCP 工具规范
通用约定:
- 采用 JSON 输入输出，成功返回 `{"ok": true, "data": ...}`，失败返回 `{"ok": false, "error": {"code": "...", "message": "...", "detail": ...}}`
- `repo_path` 为 git 仓库的绝对路径；返回的 `path` 为相对仓库根目录路径
- 行号为 1-based 且闭区间（含 `line_start` 与 `line_end`）
- 仅处理 git tracked 文件；search 可在检索前做增量同步（见工具说明）
- 统一错误码示例：`NOT_GIT_REPO`、`NOT_INITIALIZED`、`CONFIG_INVALID`、`EMBEDDING_ERROR`、`STORAGE_ERROR`、`INDEX_ERROR`

工具: init
- 作用：创建新索引并写入配置，自动确定 chunk 行数/重叠策略
- 输入:
```json
{
  "repo_path": "/abs/path/to/repo",
  "persist_dir": "/abs/path/to/persist",
  "embedding": {"base_url": "...", "api_key": "...", "model": "..."}
}
```
- 输出（示例字段）:
```json
{
  "ok": true,
  "data": {
    "repo_root": "/abs/path/to/repo",
    "repo_hash": "xxxx",
    "index_dir": "/abs/path/to/persist/xxxx",
    "config_path": "/abs/path/to/persist/xxxx/config.json",
    "files_indexed": 123,
    "chunks_indexed": 456
  }
}
```

工具: search
- 作用：语义检索；默认在查询前进行增量同步（可通过参数关闭）
- 输入:
```json
{
  "repo_path": "/abs/path/to/repo",
  "query": "string",
  "top_k": 10,
  "refresh": true
}
```
- 输出（仅返回相对路径与行号范围）:
```json
{
  "ok": true,
  "data": {
    "results": [
      {"path": "src/foo.py", "line_start": 34, "line_end": 86}
    ]
  }
}
```

工具: status
- 作用：获取当前库信息
- 输入:
```json
{
  "repo_path": "/abs/path/to/repo"
}
```
- 输出至少包含：`repo_root`、`repo_hash`、`index_dir`、`config_path`、`files_indexed`、`chunks_indexed`、`embedding.model`、`chunking` 摘要（如可用再包含 `last_indexed_at`、`last_indexed_commit`）

工具: update
- 作用：修改当前库配置并触发全量重建
- 输入:
```json
{
  "repo_path": "/abs/path/to/repo",
  "embedding": {"base_url": "...", "api_key": "...", "model": "..."}
}
```
- 输出：同 init 的索引摘要字段

## 工作项清单
| ID | 内容 | 完成状态 | 负责人 | 实施要点 | 验证方式 |
| --- | --- | --- | --- | --- | --- |
| T1 | 项目结构与配置模型 | [x] | AI | 规划 `code_index/` 包与模块边界；定义配置 JSON 结构（全局 embedding + 项目索引信息 + chunk 参数）；实现路径 hash 与持久化目录结构；在 `pyproject.toml` 增加依赖与 `code-index` 入口；优先用 apply_patch 修改 | 运行 `python -m code_index --help`；执行 `code-index init` 后生成配置文件与目录 |
| T2 | OpenAI API 兼容 embedding 客户端 | [x] | AI | 使用 `httpx`/标准库实现 `/v1/embeddings` 调用；支持 base_url/api_key/model；实现批量、超时、错误处理；提供单测与 MockTransport | `pytest -q` 通过；手测返回向量长度与数量一致 |
| T3 | Milvus Lite 存储层 | [x] | AI | 以 `pymilvus` Lite 模式创建 collection；设计 schema（vector + path + line_start + line_end + file_hash）；实现 insert/search/delete；支持按文件删除 | `pytest -q` 通过；手测 insert/search 返回结果 |
| T4 | 索引流水线（初始化） | [ ] | AI | 使用 `git ls-files` 枚举已跟踪文件；文本检测过滤二进制；按行切分并记录行号范围；初始化时自动决定 chunk 策略并写入配置；批量 embedding 与入库 | 在小仓库执行 `code-index init <repo>`，索引数与 chunk 数合理 |
| T5 | 增量更新与删除处理 | [ ] | AI | 基于 git 变更检测（tracked 文件列表 + 内容 hash）识别新增/修改/删除；删除已移除文件的向量记录；更新元数据；增量更新与 search 入口联动 | 修改/删除文件后执行 `code-index search` 或增量入口，结果更新且无旧记录 |
| T6 | MCP 工具与 CLI | [ ] | AI | 使用 MCP Python SDK 注册工具：init/search/status/update；按 MCP 工具规范实现输入输出与错误码；update 修改配置并触发全量重建；search 输出相对路径与行号范围 | `code-index serve` 启动；MCP 工具调用返回 JSON 符合约定 |
| T7 | 测试与文档 | [ ] | AI | 增加单测与简单集成测试；更新 README（安装、init、search、update、持久化目录说明）；补充故障排查 | `pytest -q` 通过；README 示例可复现 |

## 里程碑与顺序
- M1：脚手架与配置（T1）
- M2：存储与 embedding 基础能力（T2, T3）
- M3：索引与增量更新（T4, T5）
- M4：MCP/CLI 接入与文档测试（T6, T7）

## 风险与缓解
- Embedding API 兼容性问题导致请求失败；缓解：严格按 OpenAI 规范封装、加入超时与可读错误
- Milvus Lite 本地文件损坏或并发写入问题；缓解：单进程写入、启动时健康检查、失败时提示清理
- 大仓库索引耗时/内存；缓解：批量处理、chunk 策略可配置、增量更新优先
- 文本检测误判导致漏索引；缓解：保守策略、记录被跳过文件并可配置白名单
- git 变更检测不完整；缓解：基于文件 hash + git tracked 列表双重校验，删除缺失记录

## 验收与测试
- `code-index init <repo>` 可在持久化目录创建项目索引与配置
- `code-index search "<query>"` 返回 JSON，包含相对路径与行号范围
- 修改/删除 tracked 文件后搜索不返回旧片段
- `code-index update` 修改配置后触发全量重建
- `pytest -q` 通过且关键模块有覆盖

## 回滚与清理
- 删除对应项目 hash 目录与集合文件，回滚到上一次可用配置
- 移除新建分支与未合并提交（按团队流程）
- 清理临时索引与日志文件

## 工具与命令
- `apply_patch`：修改文件；注意覆盖 `.codex/task.md` 与配置更新
- `git switch -c <branch>` / `git status`：分支管理与查看变更
- `code-index init/search/status/update/serve`：CLI 入口，验证功能
- `pytest -q`：运行测试，要求全部通过
- 进度同步：每完成一项在 `.codex/task.md` 勾选；按汇报清单更新用户

## 测试计划
- 配置与路径 hash：`pytest -q tests/test_config.py`，验证配置持久化与 hash 稳定
- 文本检测与切分：`pytest -q tests/test_chunking.py`，验证二进制过滤与行号范围
- Embedding 客户端：`pytest -q tests/test_embedding.py`，Mock OpenAI API 返回
- Milvus Lite 存储：`pytest -q tests/test_store.py`，验证插入/搜索/删除
- 增量更新：`pytest -q tests/test_incremental.py`，验证新增/修改/删除处理
- 集成验证：`pytest -q tests/test_cli.py` 或手测 `code-index init`/`search`

## 汇报清单
- 计划确认要点与假设
- 当前分支名称
- 已完成任务 / 剩余任务
- 测试与验证结果（命令与关键日志）
- 阻塞或待决事项
