# 模块与运行产物约定

第 0 步（A0）将已有实现整理在同一个 `src/embodied_agent` 包中。src 顶层的旧兼容文件已删除，代码只通过下表中的实际模块导入。`scripts/*.py` 作为启动脚本调用新应用模块。训练、对抗 Agent 和完整 Hook/隔离预演按路线图后续阶段实施。

| 模块 | 位置 | 职责与接口 |
| --- | --- | --- |
| 应用组装 | `apps/demo/` | 自由/批量 CLI、`DemoSession`、地图和 Agent 组装；scripts 为薄启动脚本 |
| 共用契约 | `contracts.py`、`models/contracts.py`、`execution/interfaces.py` | 实测观察/物理异常、模型响应/规划接口、`LiveRobotEpisode`；不在契约导入时加载仿真器 |
| 地图 | `maps/schema.py`、`home_schema.py`、`unified_store.py`、各场景适配 | 同一地图目录按schema选择桌面/家居校验、版本冲突/原子保存和场景生成；文件仍在 `configs/maps/` |
| Agent与世界/目标 | `agents/instruction.py`、`goals.py`、`observation.py`、`environment.py`、`map_environment.py` | 单一高层模型先解释意图，再按当前机器人能力规划；通用世界和独立GoalSpec；组件由构造器注入 |
| 提示词 | `prompts/catalog.py`、`prompts/resources/` | `PromptCatalog` 加载四份版本化资源及 text/path 覆盖；实际文本与 SHA-256 可记录，Agent 不内嵌提示词 |
| 查询工具与能力 | `tools/query.py`、`registry.py`、`capabilities.py` | `QueryToolCatalog` 同时提供 schema 与分发，`QueryTools` 读取捕获快照；动态能力与动作契约供规划和执行共同使用 |
| 记忆 | `memory/store.py` | 有限容量、线程安全、namespace 隔离的进程内记忆；默认禁用，不自动持久化 |
| 知识库 | `knowledge/base.py` | 版本化只读文档与词面检索；默认禁用，不依赖向量或模型服务 |
| 辅助上下文 | `context.py` | `AgentContext.build()` 检索辅助数据，`remember()` 记录真实最终结果；Session 共用 store，按 run_id 和角色隔离记忆 |
| 模型服务与预算 | `models/deepseek.py`、`dialogue.py`、`budget.py`、`tracing.py` | 原生工具对话、提案修正、实际SDK请求/响应及绑定ID、反馈交付、全任务预算与取消；SDK懒加载 |
| 执行 | `execution/supervisor.py`、`panda_adapter.py`、`task.py` | 通用决策/动作/验收/恢复循环与Panda实测适配；task保留无LLM的M2物理任务 |
| 几何与程序化验收 | `maps/manipulation.py`、`procedural.py`、`geometry_acceptance.py`、`scripts/geometry_demo.py` | 按当前几何求解停靠/抓放候选，程序化同类地图；默认可视的可信计划物理验收，不代替模型规划评测 |
| 仿真 | `simulation/model.py`、`episode.py`、`live_episode.py`、`robot_episode.py`、`stretch.py`及观察/场景模块 | 统一 live episode 工厂与模型适配；唯一真实模型/数据、物理步进和原显示回调；M1诊断保留 |
| 技能 | `skills/panda.py`、`stretch.py`、`navigation.py` | 按机器人结构实现抓取/移动/携物/释放，通过同一实际步进与相应守卫执行 |
| 安全 | `safety/guards.py`、`path_precheck.py`、`home.py` | 原危险区和家居状态/空间风险守卫；不依赖模型自行宣称安全 |
| 可视化 | `visualization/demo_ui.py`、`world_view.py`、`episode_viewer.py` | Tk 世界显示和可选原生 Viewer；读取执行器正在使用的同一模型/数据 |
| 评估与执行接入 | `evaluation/cases.py`、`batch.py`、`placement.py`、`home.py`、`evidence.py`、`task_writer.py`、`task_query.py` | 用例/独立物理判定，执行事件写入统一记录，报告保留引用；v2查询/评价/导出CLI |
| 统一事实记录 | `recording/store.py`、`journal.py`、`artifacts.py`、`projector.py`、`run.py`、`environment.py` | v2 manifest/追加日志/工件/seal，派生视图与索引重建、严格校验和显式恢复，直接环境preview/apply生命周期 |
| 独立评价与导出 | `recording/assessment.py`、`export.py` | 可信typed规格与版本化不可覆盖评价；trajectory/decision/sft/configuration/task_pool/preference资格检查和同源split，无训练器 |
| 资产与配置 | `assets/`、`configs/` | 保留现有机器人资产、许可证、地图及冻结的技能/预算/用例配置 |

第4、5步升级原有 Demo，家居地图与机器人按类型适配到原会话、窗口和用例链：

| 模块 | 文件 | 家居职责 |
| --- | --- | --- |
| 地图 | `maps/home_schema.py`、`home_store.py`、`home_scene.py` | 独立HomeWorld、状态/空间关系风险、初态原子保存、房间MJCF和明确几何注册 |
| 动作契约 | `agents/instruction.py`、`agents/goals.py`、`execution/home_contracts.py` | 校验结构化目标、方法、来源、禁忌、次序与相关动作段；当前自由入口不要求等于固定完整列表；旧查询技能仅可信兼容 |
| 执行适配 | `simulation/robot_episode.py`、`live_episode.py` | `LiveRobotEpisode` 与统一工厂；移动操作技能、规则/预算/后置条件、当前版本/事件；生命周期与证据由原 DemoSession 管理 |
| 仿真/技能 | `simulation/stretch.py`、`skills/stretch.py`、`skills/navigation.py` | 锁定Stretch资产适配、真实物理步进与逐步几何/持物守卫、受限操作、足迹膨胀A* |
| 安全/评分 | `safety/home.py`、`evaluation/home.py` | 状态风险、整个机器人/持物空间限制、独立连续稳定支撑判定 |
| Agent/统一应用 | `agents/instruction.py`、`map_environment.py`、`maps/unified_store.py`、`apps/demo/session.py` / `visualization/demo_ui.py` / `world_view.py` | 单一指令核心；同一地图目录、原窗口、用例/批量、结果；机器人控制和验收保留各自适配 |

命令、适配限制和产物详见 [HOME_ROBOT.md](HOME_ROBOT.md)。任务工具不能调用地图save；原环境编辑Agent只保存有限初态字段。无界面stub执行不加载Viewer/Tk/训练或LLM后端；llm模式沿用原模型服务。

`components.prompts/tools/memory/knowledge` 的配置、直接 Agent 注入与 DemoSession 共享示例见 [AGENT_COMPONENTS.md](AGENT_COMPONENTS.md)。默认记忆/知识禁用；这次拆分提供组件接口和读写记录，未接入强化学习训练器。旧 `models/prompts.py`、`agents/query_tools.py`、`agents/capabilities.py` 已删除，改用对应独立包。

调用方向为应用组装 → 模型解释并确认GoalSpec → 多轮工具/动作段规划 → 执行监督 → 每动作实时审查 → 技能/每步守卫/仿真 → 独立验收 → 实际反馈/下一规划。查询读取每次决策捕获的快照，环境查询读取存储地图；工具本身不步进或保存。GUI等待期间主线程HOLD继续物理监控。家居/桌面自由模型提案共用goals/actions，允许相关部分段；环境operations仍统一校验后原子保存。

`configs/agent_runtime.json` 的agent段限制每次输出4096 tokens、每个对话8轮/24工具/2次修正，默认interpret_goals_first=true。整个任务共享24请求、131072实际tokens、128动作、4恢复和300秒，恢复不重置额度。执行适配接口为observe/review_next_action/run_action/hold/safe_stop/evaluate_goal_evidence，运行时使用同一真实model/data。请求、工具、原目标、动作前后态、attempt和判定进入统一v2事实日志；取消后的迟到回复只进入run审计，consumed=false，不执行或改写终态。

自由指令保留当前物理状态，只有地图选择/重置与明确的用例起点重置环境。批量运行复制地图到当前输出目录，逐用例恢复私有基线；原始地图不受评测编辑影响。`success_count` 表示实际成功数，`passed_count` 表示满足预期的用例数，预期安全拒绝只计入后者。

## 运行与导入

在仓库根目录运行真实可视自由模式：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode free --planner stub --environment-planner rules
```

M1/M2物理与安全示例保留，指令入口使用instruction_agent.py，详见 [README.md](README.md) 与 [DEMO.md](DEMO.md)。自动化批量回归显式使用无界面模式：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode batch --headless --planner stub --environment-planner rules --output .\results\demo\a0_regression
.\.venv\Scripts\python.exe .\scripts\instruction_agent.py --batch --headless --planner stub --output .\results\m3\a0_regression
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

src 中原 `demo.py`、`demo_ui.py`、`world_view.py`、`worlds.py`、`planner.py`、`language_agent.py`、`environment_agent.py`、`runtime.py` 已删除，不再提供旧模块重导出或别名。调用者直接从实际模块导入，例如：

```python
from embodied_agent.apps.demo.session import DemoSession, create_batch_session, run_batch
from embodied_agent.evaluation.cases import load_cases, check_expected
from embodied_agent.agents.instruction import InstructionAgent
from embodied_agent.prompts import PromptCatalog
from embodied_agent.tools import QueryTools, QueryToolCatalog
from embodied_agent.memory import MemoryStore
from embodied_agent.knowledge import KnowledgeBase
from embodied_agent.context import AgentContext
from embodied_agent.recording import TaskRecordStore
from embodied_agent.recording.assessment import EvaluationStore, assess_task
from embodied_agent.recording.export import export_dataset
from embodied_agent.models.contracts import PlannerResponse, PlannerError
from embodied_agent.maps.store import MapStore
from embodied_agent.execution.supervisor import EmbodiedTaskRunner
from embodied_agent.simulation.episode import Episode
```

2026-10-07删除agents中的旧语言/规划模块、旧执行器及其兼容注册，运行配置改为 `configs/agent_runtime.json`；原场景数据通过当前robot入口执行。

共用契约 `contracts.py`、项目路径 `paths.py` 和包初始化文件保留实际实现。库和测试不依赖 scripts 中的重导出。src 旧入口移除的执行证据见[本次调整记录](../task-records/20261004-190613-remove-src-compat/analysis.md)。

## 产物与清理

- Demo/指令/环境Agent主根默认固定为 `PROJECT_ROOT/records`，含 runs、`tasks/<YYYYMMDD>/<UUID>`、内容地址artifacts和evaluations；`--output` 只选报告目录，`--records-dir` 才选主根。任务manifest/journal/工件是事实源，task.json和索引可重建，seal绑定终态；结果目录只含摘要/引用及批量地图副本。
- 用户授权清理不适配的v1 Agent执行记录和重复详情；旧 `evaluation/task_records.py` 删除，不提供兼容读取。M1/M2独立物理诊断、配置、资产及执行文档保留，删除清单与hash见[本次计划](../task-records/20261007-181011-unified-training-records/plan.md)及对应分析。评价/六种导出和缺失token/logprob/恢复状态的训练边界见 [TASK_RECORDS.md](TASK_RECORDS.md)。
- `results/tmp`、`cache` 用于可再生本地产物，但其中已有文件可能被历史执行文档引用；不整体清空。清理前必须核对用途、引用及唯一失败证据。
- 地图原子写入的临时文件和 `.lock` 属于保存协议，异常遗留先确认无写入者，再按清单处理。
- `.env`、`.venv`、未提交/未跟踪正式源码、配置、资产、许可证及 `task-records` 保留。
- A0 的冻结源码、SHA-256、原 Git 状态、清理清单、经校验的压缩归档及回归证据位于[本次任务记录](../task-records/20261004-000502-robot-agent-a0/plan.md)。删除对象和可恢复位置见同目录 `cleanup-manifest.md` 与归档索引。
