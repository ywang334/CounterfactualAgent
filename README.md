# 层次化预算与通信控制器（监督训练基础框架）

这是一个轻量、可审计的研究骨架。Solver、Critic、Refiner 始终通过推理 API 调用，不进入优化器；只训练 `BudgetAllocator` 和 `ActionController`。当前阶段没有 RL、工具调用、动态拓扑、benchmark 下载或分布式训练。

## 核心语义

LangGraph 图为：

```text
START -> budget_allocator -> solver -> action_controller
                                     -> critic_or_refiner -> action_controller ... -> END
```

- Solver 每个 query 只运行一次，且不计入额外协作预算。
- 协作预算同时限制额外 completion tokens 和额外 LLM 调用次数。
- `SHORT/MEDIUM/FULL` 默认上限为 64/192/512。动作只有在完整上限及一次调用均可容纳时才会通过 mask；backend 返回的实际 completion tokens 用于记账。
- `SKIP` 只推进角色，`STOP` 立即结束；二者都不调用模型。
- 默认档位是 `ZERO=(0,0)`、`LOW=(128,2)`、`MEDIUM=(384,2)`、`HIGH=(1024,2)`，可在 [configs/default.json](configs/default.json) 修改。
- 所有反事实分支都从 `deepcopy` 的同一状态开始，在统一的大采集上限下执行；部署预算只在生成训练标签时绑定。

预算标签对同一 Solver 状态运行四个静态工作流，在样本 `max_budget` 上限内选择达到该 query 观测最佳质量的最低档位；所有可用档位都失败的样本记为 `unsolved`，默认不写入预算训练集。四档 rollout 仍全部保留以便离线重绑预算。动作标签在每个参考轨迹状态强制运行五个动作，再按每个部署预算筛选可行分支，先最大化终局质量，质量相同时最小化未来调用与 token 成本。

## 环境

本仓库提供 [environment.yml](environment.yml)。本机已创建的环境名是 `counterfactual-agent`：

```bash
conda activate counterfactual-agent
python -m pip check
```

正式训练会实例化冻结的 `sentence-transformers/all-MiniLM-L6-v2`。Mock smoke 使用确定性 `HashingEncoder`，它没有语义能力，绝不能用于正式实验；训练命令若要使用它必须显式添加 `--allow-mock-encoder`。

## 统一 CLI

输入是 JSONL，每行至少包含 `{"id": ..., "query": ...}`。`ExactMatchEvaluator` 还要求 `expected_answer`；真正实验通常应在 `hierarchical_control/evaluator.py` 中接入 benchmark 官方 evaluator。

按顺序运行：

```bash
hierarchical-control collect-budget \
  --input data/train.jsonl --output-dir artifacts/run1 \
  --backend openai --base-url http://localhost:8000/v1 \
  --api-key local --model localmodel --evaluator exact

hierarchical-control train-budget \
  --output-dir artifacts/run1 --encoder minilm --device cpu

hierarchical-control collect-counterfactual \
  --input data/train.jsonl --output-dir artifacts/run1 \
  --backend openai --base-url http://localhost:8000/v1 \
  --api-key local --model localmodel --evaluator exact

hierarchical-control train-action \
  --output-dir artifacts/run1 --encoder minilm --device cpu
```

在线 OpenAI-compatible 服务只需替换 `--base-url`、`--api-key` 和 `--model`（也可用 `OPENAI_API_KEY`）。代码使用普通 chat completions，不发送工具定义。为保证 evaluator 有意义，接入真实服务时应针对数据集替换 Solver/Critic/Refiner prompt，并使用公开 benchmark 的官方答案抽取和评分逻辑；本仓库不会自动下载 benchmark。

## 离线 smoke 与测试

```bash
conda run -n counterfactual-agent python -m pytest
conda run -n counterfactual-agent hierarchical-control smoke \
  --output-dir artifacts/smoke
```

Smoke 全程 CPU、无网络，只使用 `MockBackend + MockEvaluator + HashingEncoder`，并在每条数据和 `metrics.json` 中标记 `mock_only=true`。MockBackend 会把请求的生成上限作为确定性实际用量，以覆盖预算边界；它不是语言模型，也不能用于正式实验。

Smoke 检查预算不超限、SKIP/STOP 零调用、反事实分支隔离、预算与动作标签选择、两个控制器的前向/反向与 checkpoint 保存，以及 LangGraph 循环执行。输出包括：

```text
budget_rollouts.jsonl
budget_training.jsonl
counterfactual_rollouts.jsonl
action_training.jsonl
budget_allocator.pt
action_controller.pt
metrics.json
```

`budget_rollouts.jsonl` 保存一次 Solver 状态和四档结果；`counterfactual_rollouts.jsonl` 保存每个采集状态的五动作终局。两个 `*_training.jsonl` 是预算绑定后的监督样本。checkpoint 只保存各自控制头参数与编码器/配置元数据；共享 MiniLM 不随 checkpoint 重复保存，加载时必须提供同名冻结编码器。

## LogiQA 2.0 真实模型 Pilot

Pilot 只比较同一次 Solver 输出与后续 `Critic(FULL) -> Refiner` 的结果，不训练控制器，也不允许 Mock backend。输入是官方 MRC `logiqa/DATA/LOGIQA/dev.txt` JSONL：

```bash
hierarchical-control pilot-logiqa \
  --data-path /path/to/LogiQA2.0/logiqa/DATA/LOGIQA/dev.txt \
  --limit 50 --seed 20260810 \
  --output-dir artifacts/pilot_logiqa \
  --base-url http://localhost:8000/v1 --api-key local --model localmodel
```

默认 `temperature=0`；Solver/Critic/Refiner 上限由 `configs/default.json` 的 `pilot_generation_caps` 配置。针对 Qwen3 vLLM，`pilot_request_extra_body` 默认传入 `chat_template_kwargs.enable_thinking=false`，避免思考内容耗尽上限；其他模型可在自定义配置中设为空对象。输出 `predictions.jsonl` 和 `summary.json`，token 数严格使用服务返回的 usage；若 endpoint 不可访问或不返回 usage，命令明确失败且不会切换到 Mock。

已有 Pilot 可纯离线审计，不会初始化或调用模型：

```bash
hierarchical-control audit-logiqa-pilot \
  --predictions artifacts/pilot_logiqa/predictions.jsonl \
  --output-dir artifacts/pilot_logiqa/audit
```

审计保留 strict 结果，同时用仅识别显式 `FINAL_ANSWER:` 标记的 tolerant parser 复核格式失败，并输出 `audit_summary.json`、`audit_cases.jsonl` 和 `audit_report.md`。报告中的 Oracle 明确标记 `posthoc_oracle=true`，使用 gold 后验选择，不能作为可部署策略。

已有 Pilot 的 Solver 状态可用版本化 prompt 回放 Critic/Refiner；`minimal_v1` 保持原有行为且仍为默认版本，`structured_v2` 必须显式指定：

```bash
hierarchical-control replay-logiqa-prompts \
  --predictions artifacts/pilot_logiqa/predictions.jsonl \
  --prompt-version structured_v2 \
  --output-dir artifacts/pilot_logiqa/prompt_dev_structured_v2
```

回放从 Pilot 的 `summary.json` 继承 backend、模型、温度、生成上限和 `extra_body`，不会重新调用 Solver。每个协作阶段保存 checkpoint，完整样本立即追加到新目录的 `predictions.jsonl`，中断恢复不会重复已完成调用。输出同时包含 `summary.json` 和 `report.md`；这 50 条明确标记为 prompt development set（`deployable_result=false`），不能视为最终测试结果。

两版已保存 continuation 可进行一次纯离线稳定性审计；命令不会初始化 backend、调用模型或训练控制器：

```bash
hierarchical-control audit-prompt-stability \
  --minimal-predictions artifacts/pilot_logiqa/predictions.jsonl \
  --structured-predictions artifacts/pilot_logiqa/prompt_dev_structured_v2/predictions.jsonl \
  --output-dir artifacts/pilot_logiqa/prompt_stability_audit
```

审计比较纠错、退化、实际成本、Critic 错误检测和 corrected/degraded 标签集合稳定性，并为两版分别计算明确标记 `deployable=false` 的最小成本 posthoc Oracle。它不会自动选择、冻结或修改 prompt。
