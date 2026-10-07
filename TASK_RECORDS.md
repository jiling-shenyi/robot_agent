# Agent 执行记录 v2

更新日期：2026-10-07。指令 Agent 和环境修改 Agent 共用 schema v2。一次自然语言任务对应一个任务目录，同目标的重规划保留在尝试链中。实际模型调用、查询、提案校验、动作、环境保存和重载分别留证。

事实源是任务 `manifest.json`、追加式 `events.jsonl` 和引用的工件。`task.json` 与 SQLite 索引是可重建视图；`results` 只保存统计、结果摘要和引用。记录、独立评价、导出已实现，训练器和策略更新尚未接入。

## 运行与唯一记录根

在项目目录启动可视 Demo，使用正在执行物理步进的同一仿真实例：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode free --planner stub --environment-planner rules
```

专用自动化评测显式使用 `--headless`；下面短案例用于检查真实记录链：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode batch --headless --planner stub --environment-planner rules --case home_room_only --output .\results\demo\v2_check
```

指令入口同样默认可视化：

```powershell
.\.venv\Scripts\python.exe .\scripts\instruction_agent.py --instruction "Move the cube to the green target area."
```

`stub/rules` 是有限的离线解析器；真实模型采样使用 `llm`。离线计划保留来源，不伪装成模型调用。

默认主根固定为 `PROJECT_ROOT/records`，与当前工作目录和 `--output` 无关。Demo 和指令入口均支持 `--records-dir <主根>`，Python 会话使用 `records_dir=Path(...)`。参数指向包含 `runs/tasks/artifacts/evaluations` 的根，不能再传旧 `records/tasks` 或运行目录的 `task_records`。

`--output` 指定新的报告目录，非空目录拒绝覆盖。Demo 批量还在其中保存私有地图副本。`summary.json` 只保存摘要和引用；Session 的 `finish()` 可返回完整内存结果，磁盘报告不重复模型上下文、动作详情或轨迹。M1/M2 独立物理诊断仍使用自身格式。

```text
records/
  runs/<run_id>/
    manifest.json
    events.jsonl                  # 会话审计，含迟到回复
  tasks/<YYYYMMDD>/<task_uuid>/
    manifest.json                 # 不可变任务身份和初始引用
    events.jsonl                  # 唯一任务事件日志
    task.json                     # 可重建视图
    seal.json                     # 收尾后绑定事实与终态
    recovery.json                 # 仅显式恢复生成
    task.recovered.json           # 仅显式恢复生成
  artifacts/sha256/<前两位>/<sha256>.json
  evaluations/<eval_run_id>/<task_uuid>.json
  index.sqlite3                   # 派生索引、日序号分配
results/demo/<run_id>/
  manifest.json
  summary.json                    # 报告与引用
  maps/                           # 批量的地图副本
```

日期固定为任务开始的 Asia/Shanghai 日期，跨午夜不改目录。UUID 是任务身份；上海日序号保留在 manifest 中，在同一根跨地图、会话和进程分配，允许空洞。旧 `YYYYMMDD_<地图简写>_<序号>.json` 文件名不再使用。

## 事件与数据契约

任务事件包含 schema、连续序号、`event_id/event_type`、时间、哈希链、身份、来源、可见性及 payload/工件引用。身份区分 `run_id/session_id/task_id/agent_run_id/agent_role/attempt_id/decision_id/model_call_id/tool_call_id/action_id`。`attempt_id` 是任务内整数；模型提供的工具 ID 单独保留为 `native_tool_call_id`。

机器字段名为 `seq/operation/previous_hash/event_hash`，同时保存 `parent_event_ids/caused_by_event_ids`、采集进程 monotonic 时间及可得的模拟时间／步数。框架验证、工具执行和物理事实的 source 为 executor；真实模型输出、规则和注册例程来源分别保留。

| 事件或字段 | 实际内容和解释 |
| --- | --- |
| `model.requested/dispatch.started/responded/failed` | 实际 SDK messages/tools/参数、调用边界、原始响应或错误；同一次调用共用本地 `model_call_id` |
| `components.selected`、`context.assembled` | 实际提示词正文/版本/hash、工具 schemas/目录指纹、当次上下文；区分意图解释和规划 stage |
| `memory.read/write`、`retrieval.finished` | 实际可见内容、namespace/revision、文档版本与真实结果写入；默认禁用不产生读事件 |
| `tool.requested/finished` | 实际工具名/参数、原始 handler 结果与最终模型可见结果，本地/提供方 ID 分开 |
| `proposal.created/validation.started/validated/rejected` | 原始输出、校验与规范化结果；执行器修正计划不能替代模型原始回答 |
| `feedback.created/included_in_request/delivered` | 反馈产生、实际 messages 位置、本地交付；SDK dispatch 不代表远端已经读取 |
| `decision.*`、`action.reviewed/started/finished/failed/skipped/not_started`、`goal.evaluated` | 真实决策、审查、物理动作和判定；本地 dispatch 与物理技能边界分开 |
| `commit.started/finished/rejected`、`scene.reload.started/finished/failed` | 提交与重载的独立结果、提交前后地图和实际效果 |
| `natural_language`、`initial_state/final_state` | 原始任务和请求前后真实观测；场景尚未创建时初态可缺失并说明 availability |
| `versions/metadata` | 实际配置、组件/源码/资产 hash 与工件、行为 bundle、任务类型、实验来源和可信规格 |

累计 token_usage 从模型事件按 model_call_id 汇总。未知总量为 null，并保留 known_reported_subtotal、字段 availability 和来源 event IDs；预算器中的已报告计数不代表完整消耗。无模型调用为已知 0。
| `attempts[].actual_execution` | 任务独占的动作/物理事件、轨迹及起止切片，不混入相邻任务 |
| `attempts[].feedback` | 真实返回结果与确实交给 Agent 的反馈；未送达反馈不能填入 `sent_to_agent` |
| `outcome/costs/termination` | 实际结果、已有调用/时间/物理成本、`terminated/truncated`、结束原因和 bootstrap 状态可用性 |

`read_events()` 使用 dotted 规范名称。派生 `attempts[].events` 保留原 `event/detail`，便于检查已有执行链，属于同一事实源的视图。未发生的模型调用、查询或动作不产生成功事件；检查和预测须实际发生才可记录，记录设施不提供隔离预演。

自由任务复用 live episode，各自从当时状态开始并保存独立证据。选择地图/重置不伪造语言任务。`run_agent(..., record=False)` 只控制是否加入汇总，仍保存任务证据。任务结束或放弃后的迟到回复进入 run 审计，例如 `late.model.responded` 且 `consumed=false`，不修改任务终态或下一任务输入。

`record_status` 表示记录生命周期，`outcome.status` 表示真实结果。正常收尾的失败仍是失败；预期拒绝通过用例测试也不改成任务成功。`SUCCESS` 本身不证明用户意图正确，缺少 `goal_check` 时目标验证为 `null`，不能由正常返回补造验证。

当前观察不是可恢复 checkpoint，默认 `restorable=false`。DeepSeek 保存实际返回的 usage；未返回的 usage 为未知，token IDs 不可获得，未请求/未返回的 logprobs 为空并说明 availability。未显式设置的采样参数标为提供方默认未知。当前还没有经过验证的重置恢复协议、tokenizer 版本和完整行为策略 token 概率。

## 持久化、校验与恢复

工件按内容 SHA-256 寻址，先原子发布并核验字节数/hash，再追加引用。事实逐条追加、flush/fsync；序号和哈希链衔接 manifest。终态 seal 绑定 manifest/journal hash、事件数量、末事件 hash 和由事实重建的投影 hash。终态禁止追加任务事实。

关键模型请求和地图提交边界先记录再允许副作用。日志失败不重新调用模型、重做物理动作或重复地图保存。保存成功但重载失败保留 `persisted=true`、提交效果和独立重载错误。突然退出可能留下未封存任务/未结束调用，不能从缺少结束事件推断副作用没有发生。

`TaskRecordStore.load()` 校验并从事实源返回重建视图，不信任缓存 JSON。缓存丢失或陈旧可重建；事实损坏、缺工件或 seal 不匹配拒绝严格读取。`verify()` 分开报告 `valid/sealed`：健康 RUNNING 任务可以 valid，未封存仍不能导出为完整轨迹。

报告中的 `task_record` 引用绑定 manifest/journal/seal 的 SHA-256 和完整性结果。`sha256/projection_sha256` 仅表示 task.json 缓存的 hash，读取失败时为 null；缓存落后不否定完整事实。run.finished 后仅允许 consumed=false 的 late.* 审计；追加失败或损坏后停止写入，保留原日志。

查询、评价与导出仅依赖标准库，不加载 MuJoCo/模型客户端。在项目根运行：

```powershell
.\.venv\Scripts\python.exe .\scripts\task_records.py --date 2026-10-07 --map home_living_room --status FAILED
.\.venv\Scripts\python.exe .\scripts\task_records.py --record-status RUNNING
.\.venv\Scripts\python.exe .\scripts\task_records.py --agent-role environment --task-kind map_edit

$taskId = (Get-Content .\results\demo\v2_check\summary.json -Raw -Encoding UTF8 | ConvertFrom-Json).results[0].actions[0].task_id
.\.venv\Scripts\python.exe .\scripts\task_records.py --task $taskId
.\.venv\Scripts\python.exe .\scripts\task_records.py --task $taskId --events
.\.venv\Scripts\python.exe .\scripts\task_records.py --task $taskId --verify

# 只重建派生缓存，不修改日志
.\.venv\Scripts\python.exe .\scripts\task_records.py --task $taskId --rebuild
.\.venv\Scripts\python.exe .\scripts\task_records.py --rebuild-index
```

`--task` 接受 UUID、任务目录或 `task.json` 路径；`--date` 支持两种格式。自定义主根时提供相同的 `--records-dir`。`--integrity complete/valid/invalid/unsealed` 筛选完整性；坏事实通过 `errors` 报告并返回非零退出码，不当成合法数据。`--verified-only` 仅筛选执行器验证的实际成功，不能代替训练资格。不存在的根报参数错误，查询不创建目录。

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))
from embodied_agent.recording import TaskRecordStore

store = TaskRecordStore(Path("records"))
failed = store.query(status="FAILED", agent_role="environment")
records = store.query()
if records:
    task = store.load(records[0]["task_id"])
    events = store.read_events(task["task_id"], resolve=True)
    check = store.verify(task["task_id"])
    print(task["natural_language"], check)
print(len(failed), store.errors)
# 人工确认需要检查损坏日志的有效前缀后才显式调用：
# recovery = store.rebuild(task["task_id"], recover=True)
```

显式恢复保留损坏日志，另写 `task.recovered.json/recovery.json`。恢复不补造丢失事件、回复、奖励或终态，也不自动获得导出资格。CLI `--rebuild` 只严格重建，恢复需 Python 的 `recover=True`。

## 可信规格与独立评价

模型解释目标、执行器验证目标和用户意图正确性分别保存。可信规格来自用户或受控 fixture，独立于模型输出；有效 JSON、`expected_pass` 不能代替它。Session 支持 `run_agent(..., trusted_task_spec=...)`，用例/步骤也可提供 `trusted_task_spec`。事后评价可显式提供规格，但保留 retrospective 来源。

规格要求 `trusted=true`、`source="user"` 或 `"fixture"` 和 `kind`，支持 `robot_goal/map_edit/query/stop/clarification/rejection`。机器人任务需 `goals`、地图编辑需 `expected_operations`，查询需明确 `query_names`。下面只适用于用户确实要求的房间改名：

```json
{
  "trusted": true,
  "source": "user",
  "kind": "map_edit",
  "expected_operations": [{"op": "set_name", "value": "测试客厅"}]
}
```

将规格保存为自己的 UTF-8 文件，先把 `$editTaskId` 设为对应编辑任务 UUID，再评价：

```powershell
.\.venv\Scripts\python.exe .\scripts\task_records.py --task $editTaskId --assess --eval-run-id map-review-v1 --trusted-task-spec .\configs\my-map-task-spec.json
```

评价写不可覆盖的 `evaluations/<eval_run_id>/<task_id>.json`，保存评价器版本、代码/config hash、源任务 manifest/journal/seal hash、证据 event IDs/hash、规格和分项结果。重新评价用新 `eval_run_id`，不覆盖旧评价或任务事实；读取时重新核验任务与证据 hash。

hash/seal 正确不保证所有业务证据都成功采集。recording_errors 或 RECORDING_FAILED 会使内置评价无效、奖励为 null；SFT／Preference／task_pool 再次检查这类缺口，拒绝仅靠调用方正评补资格。分析导出保留事实并显式标记 recording_evidence_complete=false。

`intent_correctness` 与 `execution_success` 独立。机器人意图对照可信 goals，执行成功需真实目标证据。`map_edit` 执行成功是已持久化地图是否实现 `expected_operations` 的效果，忽略正常 revision 增量；`commit_status/reload_status` 分开保存。正确地图已保存但显示重载失败，可以同时有执行效果成功与重载失败；保存成功或计划通过本身不保证效果正确。

`scalar_reward` 默认 `null`。`--scalar-reward <有限数值>` 只保存调用方明确选择的奖励，不从 `SUCCESS/verified/expected_pass` 推算。缺可信规格、有效评价或完整证据时保持空奖励。当前不自动决定奖励权重，也不把离线评价回填成当时送给模型的反馈。

## 六种导出与训练资格

导出到全新目录，包含 `manifest.json/samples.json/qualification.json/seal.json`。manifest 冻结源任务 hash、明确选择的评价、导出器版本/hash、split 和文件 hash。qualification 列出每个任务可用性/排除理由；零合格样本也会输出可检查的空数据集。

| `--export` | 数据与当前资格 |
| --- | --- |
| `trajectory` | 完整封存的任务与事件，包含真实失败，用于复盘/标注，不要求奖励或正确示范 |
| `decision` | 按 decision ID 关联模型输入/原始输出、观测、动作、反馈和终止；需配对调用和决策身份，缺状态明确标 availability，仍需算法状态/奖励适配 |
| `sft` | 实际 messages/tools 和原始 assistant；需可信规格、有效评价，意图与执行成功均为 true；按调用排除被拒提案、非法协议及缺合法工具请求／成功结果的回合，只对真实生成的 assistant 内容计损失 |
| `configuration` | 组件/代码配置候选、版本/hash、结果/成本及可选评价，供提示词/工具/记忆/知识/框架候选比较；不运行优化器 |
| `task_pool` | 可信任务、地图/初态和可复现起点；需已验证 reset_spec 或可恢复 checkpoint，默认观测快照不足 |
| `preference` | 同输入/工具、地图/起点/规格/评价器及明确相同预算的原始输出对，需明确且不同奖励；当前仅支持每个任务一次真实模型调用，多轮不自动拼成偏好对 |

```powershell
.\.venv\Scripts\python.exe .\scripts\task_records.py --export trajectory --output .\results\datasets\trajectory-v1
.\.venv\Scripts\python.exe .\scripts\task_records.py --task $editTaskId --export sft --evaluation "evaluations/map-review-v1/$editTaskId.json" --output .\results\datasets\sft-v1
```

`--evaluation` 可重复，每个任务只选一份评价，不猜最新版本。CLI 支持结合任务/角色/日期筛选；Python `export_dataset(store, output_dir, kind=..., task_ids=..., evaluation_refs=..., split_ratios=...)` 可指定划分比例。默认 train/validation/test 为 0.8/0.1/0.1，按 run/session、任务族、用例、相同起点和记忆关联组成的连通组划分，同源不逐条随机拆开。

`run_agent(..., experiment={...})` 可提供 candidate_id、optimization_target、rollout_group_id、branch_id；配置导出保留这些控制变量，rollout/branch 关联纳入 split。顶层 metadata 与 experiment 同字段冲突时拒绝导出。实际 runtime／预算正文保存在行为 bundle；冻结测试与检索的隔离仍由实验组装方使用独立 namespace／知识库保证。

每种导出额外报告 `rl_update.eligible=false`，当前没有选定算法适配器和训练器。行为 logprobs、completion token IDs、tokenizer/行为策略版本、可信奖励、真实采样及算法状态/重置不足逐项列原因。当前可用于审计、符合资格的 SFT、配置比较和严格偏好对，不能直接宣称已适合 PPO/GRPO 更新。`stub/rules` 不是模型策略采样，执行器规范化计划不能替代原始生成 token。

## 旧版数据边界

本次按用户授权清理不适配的 v1 Agent 执行记录和重复详情，具体对象/hash 见[执行计划](../task-records/20261007-181011-unified-training-records/plan.md)和对应分析。旧 `evaluation/task_records.py` 已删除，没有 v1 兼容模块；`scripts/task_records.py` 保留为 v2 查询/评价/导出入口。

把旧文件改 schema 或搬进目录不会得到 v2。缺失的真实 messages、工具结果、执行边界、token 概率、可信规格和奖励不能补造。M1/M2 独立诊断、配置、机器人资产和任务执行文档不属于本次 Agent v1 删除范围。
