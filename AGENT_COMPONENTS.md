# Agent 组件配置与注入

更新日期：2026-10-07。指令 Agent 和环境修改 Agent 共用独立的 `prompts/`、`tools/`、`memory/`、`knowledge/` 包，由 [AgentContext](src/embodied_agent/context.py) 组装辅助上下文，由 [DemoSession](src/embodied_agent/apps/demo/session.py) 组装和复用组件。提示词和工具定义不再内嵌在 Agent 文件中。

可替换组件已接入统一v2记录：真实模型输入/输出、组件候选、读写和检索进入同一事件源；独立可信规格评价和六种数据导出已实现。奖励由调用方明确选择，训练器、算法适配器和策略参数更新仍待实现，训练资格见 [TASK_RECORDS.md](TASK_RECORDS.md)。

## 包与职责

| 包或模块 | 公开接口 | 职责 |
| --- | --- | --- |
| [prompts/](src/embodied_agent/prompts/__init__.py) | `PromptSpec`、`PromptCatalog` | 四份默认提示词资源、显式版本、精确文本 SHA-256、按配置替换 |
| [tools/](src/embodied_agent/tools/__init__.py) | `QueryTools`、`QueryToolCatalog`、`QueryToolDefinition` | 同一目录提供模型 schema 和调用分发；查询捕获状态、实体和能力 |
| [tools/capabilities.py](src/embodied_agent/tools/capabilities.py) | `build_capabilities`、`validate_actions` | 生成机器人能力与动作契约；供 Agent 和执行器共同使用 |
| [memory/](src/embodied_agent/memory/__init__.py) | `MemoryStore` | 进程内、线程安全、有容量上限的 namespace 隔离记忆 |
| [knowledge/](src/embodied_agent/knowledge/__init__.py) | `KnowledgeBase` | 有版本的只读文档与词面检索；英文词和中文相邻字匹配 |
| [context.py](src/embodied_agent/context.py) | `AgentContext` | 按配置检索辅助内容，通过 `build()` 提供模型上下文，通过 `remember()` 写实际任务结果摘要 |

记忆和知识是辅助数据，不能替代当前实测观察、已确认目标、权限或执行反馈。默认两者禁用，默认提示词文本与拆分前逐字一致，默认四个查询工具及权限保持原行为。

`models/prompts.py`、`agents/query_tools.py` 和 `agents/capabilities.py` 已删除；直接导入表中的新包，不提供旧路径的重导出。

## components 配置

配置位于完整运行字典的 `components` 段，默认文件为 [configs/agent_runtime.json](configs/agent_runtime.json)。下面是可合并进运行配置的组件段：

```json
{
  "components": {
    "prompts": {"overrides": {}},
    "tools": {
      "query_names": ["observe", "query_world", "inspect", "get_capabilities"],
      "descriptions": {}
    },
    "memory": {
      "enabled": false,
      "max_entries": 128,
      "history_limit": 8,
      "max_chars": 4000,
      "record_completed_tasks": true
    },
    "knowledge": {
      "enabled": false,
      "version": "knowledge-v1",
      "documents": [],
      "top_k": 3,
      "max_chars": 4000
    }
  }
}
```

传入的配置在组件构造时分离复制。比较两个候选时构造两个组件实例，避免修改全局定义。`enabled` 显式为 `false` 时，即使注入 store 或 knowledge 对象也保持关闭；要使用注入对象，须把对应配置设为 `true`。

### 提示词

| prompt ID | 默认版本 | 使用位置 |
| --- | --- | --- |
| `instruction.intent` | `instruction-agent-v2` | 模型解释任务意图 |
| `instruction.plan` | `instruction-agent-v2` | 模型规划和重规划 |
| `environment.desktop` | `environment-agent-desktop-v1` | 桌面初始地图编辑 |
| `environment.home` | `environment-agent-home-v1` | 家居初始地图编辑 |

`PromptCatalog(config, project_root=...)` 接收完整运行字典，`get(id)` 返回不可变的 `PromptSpec(id, version, text)`。`to_dict()` 返回 `id/version/sha256/text`，摘要对应实际 UTF-8 文本。

覆盖项必须包含非空 `version`，并且在 `text` 与 `path` 中选择一个。例如：

```json
{
  "components": {
    "prompts": {
      "overrides": {
        "instruction.plan": {"path": "configs/prompts/plan-candidate.txt", "version": "plan-trial-v1"}
      }
    }
  }
}
```

相对路径以传入的 `project_root` 为基准，默认使用项目根；文件必须是 UTF-8。每次 `get()` 读取文件，保留换行和末尾字符，替换文件后不会命中旧文本缓存。修改内联配置时构造新 catalog。路径缺失、未知 ID、缺版本和 `text/path` 同时出现都会明确报错。

覆盖替换整份提示词。环境默认资源已经包含只读查询说明，消费方不会再追加旧 suffix；自定义环境提示词需要自行保留所需输出契约和查询说明。

### 查询工具

`QueryToolCatalog(config)` 读取 `components.tools.query_names` 和 `descriptions`；选择工具的顺序也是发布 schema 的顺序。`query_names` 可以是四个注册工具的有序子集，未知名称和重复名称拒绝加载。默认意图提示词要求先调用 `observe`；删掉该工具的实验需要同步调整提示词。

`QueryTools(world, snapshot, catalog=...)` 从同一个 catalog 取得 `schemas` 并执行 `call(name, arguments)`，避免模型看到的定义与执行分发不一致。查询使用本轮捕获状态，不能步进物理、保存地图或解除权限。技能动作仍经过执行器的真实审查与守卫。

通过 `QueryToolCatalog(config, definitions={...})` 可注入既有名称的 `QueryToolDefinition`：`name/description/parameters/handler/version`。handler 接收 `QueryTools` 上下文和复制后的参数。替换可以调整描述、限制 ID 或换可信的只读实现；必须保留封闭参数结构和原参数类型。catalog 不接受新增权限或任意新工具名，定制实现的只读行为由实现者负责。

### 记忆与知识库

`MemoryStore(max_entries=128)` 的容量是整个 store 的条目总数；满时按最早写入淘汰。`write/read/snapshot/search/clear` 都需要显式 namespace，读写使用复制的 JSON 数据，namespace revision 用于标识快照。`search(..., limit=8, max_chars=4000)` 返回词面相关条目；字符预算计入条目、metadata 和分数。放不下的结构化值整体省略，字符串节选带截断标记。

该 store 不写磁盘，进程退出后不会自动恢复记忆。`record_completed_tasks` 控制是否自动写任务结果摘要，成功和失败均保留其实际状态；它不把模型计划、预测或建议认作完成。

`KnowledgeBase(documents, version=...)` 复制文档，每份文档包含唯一 `id`、非空 `text` 和可选 `metadata`。`retrieve(query, top_k=3, max_chars=4000)` 只返回实际词面匹配，无相关内容时返回空列表。它是本地词面检索，没有 embedding、向量数据库或自动联网。

配置可以使用 `knowledge.documents`，或互斥的 `knowledge.documents_file`。文件路径相对项目根，内容为文档列表，或 `{"version":"...","documents":[...]}`。文档加载后保持该实例的快照，更新内容需要构造新 KnowledgeBase。字符预算包含文档 metadata 和分数，节选带明确截断标记。

默认配置含有 `documents: []`；改用 `documents_file` 时先移除 `documents` 键。

## 直接组装 Agent

从项目根执行下面的代码可以加载组件并组装 Agent，不调用模型或启动仿真。先让 Python 可导入 `src`，例如在项目虚拟环境运行脚本时使用与现有 tests 相同的路径设置：

```python
import json
from pathlib import Path
import sys

root = Path.cwd()
sys.path.insert(0, str(root / "src"))

from embodied_agent.agents.instruction import InstructionAgent
from embodied_agent.context import AgentContext
from embodied_agent.knowledge import KnowledgeBase
from embodied_agent.memory import MemoryStore
from embodied_agent.prompts import PromptCatalog
from embodied_agent.tools import QueryToolCatalog

config = json.loads((root / "configs" / "agent_runtime.json").read_text(encoding="utf-8"))
components = config.setdefault("components", {})
components.setdefault("memory", {})["enabled"] = True
components.setdefault("knowledge", {})["enabled"] = True

memory = MemoryStore(max_entries=128)
knowledge = KnowledgeBase([
    {"id": "units", "text": "坐标单位为米，初始地图编辑需要校验 revision。"}
], version="example-knowledge-v1")
prompts = PromptCatalog(config, project_root=root)
tools = QueryToolCatalog(config)
context = AgentContext(config, role="instruction", namespace="example:instruction",
                       memory_store=memory, knowledge_base=knowledge, project_root=root)
agent = InstructionAgent(config, prompt_catalog=prompts, tool_catalog=tools,
                         context_provider=context)
print(prompts.get("instruction.plan").version)
print(context.build("坐标单位"))
```

`InstructionAgent` 与 `EnvironmentAgent` 构造器都支持 `prompt_catalog`、`tool_catalog`、`tool_factory`、`context_provider`。提示词注入对象需要实现 `get(prompt_id)` 并返回 PromptSpec；无需继承 PromptCatalog。`tool_factory` 构造每轮查询上下文：指令路径调用 `factory(world, snapshot, capabilities=..., catalog=...)`，返回对象需要 `snapshot/schemas/call()/capabilities()`；环境路径调用 `factory(world, edit_mode=True, catalog=...)`，没有 snapshot 位置参数，返回对象提供 `schemas/call()`。

`context_provider` 可以是 AgentContext，或实现 `build(instruction)` 与 `describe()` 的替代对象。辅助内容放在独立 `auxiliary_context` 字段，不能覆盖 instruction、snapshot、world、capabilities、feedback 或 task_context。`InstructionAgent.plan()` 只生成提案；调用方取得真实执行结果后才调用 `AgentContext.remember(instruction, result, task_id=..., map_id=...)`。该接口只接受终态 `SUCCESS/FAILED/ABORTED`，调用方仍须保证来源真实。

`EnvironmentAgent/UnifiedEnvironmentAgent.preview()` 和 `apply()` 默认建立独立v2生命周期；preview不提交地图，apply保存经过校验的有限编辑。构造参数 `records_dir` 选择主根，默认 `PROJECT_ROOT/records`；测试只校验纯Agent逻辑时可明确 `recording_enabled=False`。Session复用外层任务生命周期，不重复生成内层任务。有效提案、保存效果、场景重载和用户意图评价分别记录。

## DemoSession 共享与结果写入

DemoSession 支持 `runtime_config`、`prompt_catalog`、`tool_catalog`、`memory_store`、`knowledge_base` 注入。以下代码可接在前一示例后运行，仅组装和关闭会话：

```python
import tempfile
from embodied_agent.apps.demo.session import DemoSession

with tempfile.TemporaryDirectory() as folder:
    session = DemoSession(root=root, output_dir=Path(folder) / "run",
                          records_dir=Path(folder) / "records",
                          runtime_config=config, prompt_catalog=prompts,
                          tool_catalog=tools, memory_store=memory,
                          knowledge_base=knowledge)
    try:
        print(session.run_id)
        session.finish()
    finally:
        session.close()
```

两个 Agent 共用传入的 store 和知识对象；会话为 instruction/environment 分别建立 `<run_id>:instruction`、`<run_id>:environment` namespace，覆盖配置中的通用 namespace。不同角色、不同运行的经验不会在共享 store 中混读。共享的知识库是明确注入的同一版本，实验/训练数据隔离仍由调用方组装不同实例和 namespace。

记忆开启时，Session 取得实际任务的最终结果后，通过对应角色的 `AgentContext.remember()` 写摘要。摘要保存真实 status、error_code、verified、persisted 等字段，能够区分失败与已持久化编辑；由模型返回一段有效 JSON 不会触发成功经验写入。任务记录上下文启用时采集实际读写和检索事件，最终 transport messages 保存实际显示的辅助内容。

Session的run manifest保存启动组件快照；每项任务保存行为bundle及源码/配置/资产引用，`components.selected` 标识当次实际提示词正文/版本/hash、工具schemas/目录指纹和上下文配置。记忆/知识启用后的 `memory.read/retrieval.finished/memory.write` 保存实际可见内容、revision或文档版本，transport messages保留实际 `auxiliary_context`。派生attempt.events保留旧 `agent_components/memory_read/knowledge_retrieval/memory_write` 名称，标准读取使用v2规范事件名。记忆写入失败记录 `component_errors`，保留真实结果，不重复动作或地图提交。

主根固定为 `PROJECT_ROOT/records`，显式 `output_dir` 不改变它；示例使用临时 `records_dir` 隔离数据。results报告只保存摘要/引用，可信规格和独立评价不覆盖事实源。`configuration` 导出用于比较组件/代码候选，SFT/偏好导出还需独立意图与结果证据。默认无token IDs、完整行为logprobs或可恢复checkpoint，所有导出明确报告RL更新资格不足，详见 [TASK_RECORDS.md](TASK_RECORDS.md)。

需要观察机器人实际运行时，继续使用原可视入口：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode free
```

程序化注入示例不增加 CLI 参数。CLI 使用默认配置；要组合候选组件，用 DemoSession 的 Python 接口，或显式调整运行配置。机器人动作、实时安全检查与独立验收仍按 [自然语言任务执行链路](自然语言任务执行链路.md) 执行。
