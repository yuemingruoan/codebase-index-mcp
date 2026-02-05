# 向量 GPU 加速任务

## 标题与目标
为向量索引与检索增加 GPU 加速（CUDA/MPS）并自动回退 CPU，支持按参数选择精确/近似检索。完成定义：CLI/MCP 可配置设备与检索模式，索引/搜索在可用 GPU 上运行，测试通过。

## 交付物
- GPU 设备检测与向量后端路由实现（CUDA/MPS/CPU）
- 新向量存储/检索后端（Torch）与持久化格式
- 最大显存占用配置（安装时可设置并写入索引配置）
- 配置与 CLI/MCP 参数（device、search_mode、metric、max_vram_mb）
- 索引/增量更新流程适配新后端
- 单测与 README 更新（GPU 说明、显存上限、参数说明）

## 范围 / 非范围
范围:
- 向量索引/检索使用 GPU（CUDA/MPS）并支持自动回退 CPU
- 默认自动检测设备，可通过参数强制指定
- 安装时可设置最大显存占用（环境变量，写入索引配置）
- 近似/精确检索由 tools 参数决定
- 仅使用 Torch 作为向量后端（不引入 Faiss）
- 允许调整持久化格式与配置结构（无需向前兼容）

非范围:
- 外部向量数据库或独立服务进程
- 本地 embedding 模型与权重管理（继续使用 OpenAI 兼容 API）
- 分布式/多机索引或多租户能力
- UI 或可视化管理界面
- 旧索引文件迁移/兼容适配
- Faiss 或其他第三方 ANN 库支持

## 工作项清单
| ID | 内容 | 完成状态 | 负责人 | 实施要点 | 验证方式 |
| --- | --- | --- | --- | --- | --- |
| T1 | 配置与参数设计 | [ ] | AI | 更新 `code_index/config.py`：新增 VectorConfig(device/metric/search_mode/approx_params/max_vram_mb)，替换 MilvusConfig；支持 `CODE_INDEX_MAX_VRAM_MB` 默认值；调整 RepoConfig 序列化；更新 `mcp_server.py`/`cli.py` 输入 schema 与参数（`--device`/`--search-mode`/`--metric`/`--approx-sample-rate`/`--max-vram-mb`） | `pytest -q tests/test_config.py`；`code-index init --help` 包含新参数 |
| T2 | 设备检测与路由 | [ ] | AI | 新增 `code_index/device.py`：检测 CUDA/MPS；实现 auto 选择与 CPU 回退；提供统一 `resolve_device()` 给向量后端使用 | 新增/更新 `tests/test_device.py`；`pytest -q tests/test_device.py` |
| T3 | 向量存储后端 | [ ] | AI | 实现 `VectorStore` 抽象；新增 TorchVectorStore（CPU/CUDA/MPS，矩阵乘法+topk，依据 max_vram_mb 分块计算）；定义持久化格式：`vectors/embeddings.pt`、`vectors/meta.json` | 新增 `tests/test_vector_store.py`；`pytest -q tests/test_vector_store.py` |
| T4 | 索引/增量流程改造 | [ ] | AI | 更新 `code_index/indexer.py`/`operations.py` 调用新 VectorStore；处理 delete/insert 与必要的 index 重建；search_mode=approx 时按采样参数缩减候选并记录 | `pytest -q tests/test_incremental.py`；手测 `code-index init/search` |
| T5 | CLI/MCP 兼容与默认策略 | [ ] | AI | search 工具增加 `search_mode`/`device` 参数；init/update 增加 vector 配置参数；默认 auto 检测 GPU；出错时返回清晰错误码 | `pytest -q tests/test_cli.py`；MCP 工具调用参数校验 |
| T6 | 依赖与打包 | [ ] | AI | 更新 `pyproject.toml` 依赖（`torch` 必需）；补充安装时显存上限说明与回退策略 | `python -m pip install -e .` 无错误（后续执行） |
| T7 | 文档与示例 | [ ] | AI | 更新 `README.md`：GPU 支持、设备/检索参数、回退策略、示例命令 | 目视检查 README 示例可执行 |

## 里程碑与顺序
- M1：配置与设备选择（T1, T2）
- M2：向量后端与持久化（T3）
- M3：索引/CLI/MCP 接入（T4, T5）
- M4：依赖、测试与文档（T6, T7）

## 风险与缓解
- GPU 可用性与平台差异导致运行失败；缓解：严格 auto 检测与 CPU 回退，清晰报错
- 大索引显存不足或超出用户设定上限；缓解：依据 max_vram_mb 分块、限制 top_k、必要时 CPU 回退
- 近似检索可能带来召回下降；缓解：默认 exact，明确标注 approx 预期
- 结果度量不一致（IP/L2）；缓解：统一 metric 配置并在配置中显式保存

## 验收与测试
- `code-index init/search/status/update` 在有 CUDA/MPS 时走 GPU，在无 GPU 时自动回退
- 通过环境变量/参数设置 max_vram_mb 后索引/检索遵循上限
- search 支持 `search_mode=exact|approx` 且按参数生效
- 增量更新后搜索结果不包含已删除文件
- `pytest -q` 通过（必要时对 GPU 测试做 skip）

## 回滚与清理
- 删除新索引目录 `vectors/` 下的持久化文件并重新 `init`
- 回滚代码分支（按团队流程）
- 清理新增依赖或锁文件变更（按发布流程）

## 工具与命令
- `apply_patch`：修改代码与配置（覆盖 `.codex/task.md`）
- `git switch -c <branch>` / `git status`：分支管理与变更查看
- `pytest -q` / `pytest -q tests/<file>`：运行测试并查看通过信号
- `code-index init/search/status/update`：手测索引与检索
- `CODE_INDEX_MAX_VRAM_MB=<int>`：安装/运行时设置显存上限并写入配置
- 进度同步：完成任务后更新 `.codex/task.md` 勾选

## 测试计划
- 配置序列化：`pytest -q tests/test_config.py`，覆盖 VectorConfig round-trip、默认值与 `CODE_INDEX_MAX_VRAM_MB`、非法参数校验
- 设备选择：`pytest -q tests/test_device.py`，mock torch 设备能力，覆盖 CUDA/MPS/CPU 选择与回退
- 向量后端：`pytest -q tests/test_vector_store.py`，覆盖 insert/search/delete、exact 与 approx 采样、max_vram_mb 分块策略
- 增量更新：`pytest -q tests/test_incremental.py`，覆盖新增/修改/删除、索引重建、维度变化后搜索
- CLI 集成：`pytest -q tests/test_cli.py`，覆盖新参数解析、环境变量覆盖、错误码与输出结构
- 全量回归：`pytest -q`，GPU 相关测试在无设备时自动 skip

## 汇报清单
- 计划确认要点与关键假设
- 当前分支名称
- 已完成任务 / 剩余任务
- 测试与验证结果（命令与关键日志）
- 阻塞或待决事项
