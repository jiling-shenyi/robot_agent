# 通用模拟世界测试台

A0 后，应用组装位于 `src/embodied_agent/apps/demo/`，显示位于 `visualization/`，地图、执行、Agent、模型服务和评估拥有独立模块。`scripts/demo.py` 调用应用模块；接入使用[模块说明](MODULES.md)中的公开接口。

P0–P2 已将自由指令接入统一的 [InstructionAgent](src/embodied_agent/agents/instruction.py) 和 [EmbodiedTaskRunner](src/embodied_agent/execution/supervisor.py)：家居 Stretch 与桌面 Panda 使用同一目标、查询、执行反馈和任务预算体系，由各自适配器执行实际技能。在 `robot_agent` 目录运行，默认加载家居地图并显示 MuJoCo 场景：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode free
```

机器人规划和环境编辑默认调用项目 `.env` 配置的 DeepSeek。离线开发可显式使用指令 stub 和环境规则解析器：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode free --planner stub --environment-planner rules
```

`stub` 只替代语言规划请求，抓取、移动、放置仍由真实 MuJoCo 物理仿真和原有安全检查执行。`rules` 是有限语法解析器；这两种模式的结果不能当作真实 LLM 验收结果。

## 界面操作

- 左上角选择地图并点击 **加载地图**。默认 `home_living_room` 为 Stretch 家居机器人；`classic` / `alternate` 为原 Panda 机械臂。切换时重建相应机器人，仍在同一窗口运行。
- 中央使用 MuJoCo 显示正在执行的同一份 `MjModel` / `MjData`，并绘制世界 X/Y/Z 轴及米制参考网格。画面随窗口大小和最大化状态匹配整个显示区域；调整窗口不会重置手动镜头。拖动鼠标旋转、滚轮缩放、双击恢复视角。
- **机器人 Agent** 输入自然语言，如家居的“把遥控器放到地板上”“把遥控器送到沙发”，或桌面的“把方块放到 A 区”。同一 Agent 从当前实体、支撑面和能力中选择目标，经过结构化校验、逐动作审查及物理步守卫执行；是否可达由当前几何、权限和真实状态决定。
- **模拟世界环境修改 Agent** 修改所选地图的初始环境，成功后保存并重置显示。编辑不是机器人动作，不计作抓取或搬运成功。
- **重置环境** 重新读取所选地图最新保存的初始状态，复位机器人、方块、速度、控制和仿真时间。它不会撤销已经保存的地图编辑。
- 右上角选择 **测试用例** 并点击 **执行用例**。每条用例重置到其指定的初始布局后执行，日志保留结果。冻结 M3 用例有自己的方块起点；普通用例使用地图初始坐标。
- 动作结束或被安全检查拒绝后，场景和结果保持可见。执行期间地图、重置和提交按钮暂时禁用，关闭窗口会停止后续动作并保留证据。

自由指令连续作用于当前运行状态。导航、抓取、携带、放置可分多条提交，持物和物品当前位置持续保留；执行下一条不会自动恢复初态。如需重新开始，使用重置按钮或测试用例入口。家居技能与指令示例见 [HOME_ROBOT.md](HOME_ROBOT.md)。

在 `--planner llm` 模式中，原始自然语言交给模型解析；契约校验模型给出的结构化目标和动作，不扫描原话的关键词或目的地别名。家居和桌面共用 `goals` / `actions` 提案，保留用户指定的对象、源位置、方法、顺序和禁止事项。歧义可返回 `CLARIFICATION_REQUIRED`，缺少实际技能可返回 `CAPABILITY_GAP`；例如要求“扔”不能静默改成受控放置。`--planner stub` 保留有限的显式离线语法，不能代表模型的自然语言理解能力。

机器人和环境修改 Agent 均支持原生 function/tool calling：模型可以调用 `observe`、`query_world`、`inspect`、`get_capabilities`，程序执行只读查询，以 `role=tool` 回传结果，进入下一轮模型交互。每次机器人决策的查询读取该次决策开始时捕获的快照；它不会在后台直接访问或推进 MuJoCo。环境查询读取待编辑的存储地图，也没有保存接口。

指令执行形成“决策 → 单步审查/执行 → 实测反馈 → 独立验收”的循环。模型可一次给出一段动作；每个动作调用前重新检查当前状态、地图版本、风险、路径和许可。家居导航每0.1秒复核剩余路线及移动障碍的短时预测，路径改变时先经守卫制动，再在本地重规划；可恢复失败向模型提供真实失败和已完成动作，继续保留原始目标。完成的前缀仅在实测效果仍成立时跳过，停止、等待等显式意图不会被删除。碰撞、损伤、无法可靠停止等错误终止任务；模型不能绕过安全守卫。

等待模型时，GUI 主线程持续执行受监控的 `HOLD`、检查取消与预算并刷新同一仿真画面；语言工作线程处理捕获的数据。停止后再规划仍须维持持物安全，物体的支撑、释放和连续稳定由独立物理验收确认。所有目标满足后结束任务；未满足时可继续下一段。环境编辑仍将最终 `operations` 统一校验后原子保存。

[configs/agent_runtime.json](configs/agent_runtime.json) 的 `agent` 段限制每次模型输出 **4096 tokens**，每个查询对话最多 **8轮模型交互、24次工具调用**；`execution.max_decision_rounds` 限制整项指令最多8次决策。所有决策、查询对话及格式修复共用 [TaskBudget](src/embodied_agent/models/budget.py)：**24次模型请求、131072输入加输出tokens、128次动作尝试、4次失败恢复、300秒墙钟时间**。达到任一适用上限或取消任务后停止后续执行，安全制动仍会完成；实际使用量写入任务记录。LLM错误不会自动回退为规则模式。

指令和环境修改 Agent 共用记录 v2，默认主根固定为项目 `records/`，与 `--output` 无关；`--records-dir` 可指定另一主根。每项任务写入 `records/tasks/<上海日期YYYYMMDD>/<任务UUID>/`，以 `manifest.json`、追加式 `events.jsonl` 和引用工件保存地图、真实前后观测、模型与工具调用、提案、关联 attempts、动作和反馈；`task.json` 是可重建视图，收尾生成 `seal.json`。连续任务证据互不改写，`records/runs/` 保存会话审计。报告目录只写 manifest、摘要和任务引用，批量另保留地图副本。记录、独立评价与导出已实现，训练器尚未接入。字段与查询见 [TASK_RECORDS.md](TASK_RECORDS.md)。

危险区是禁入体积，不是可放置目标。模型提案选择危险区会由独立结构化校验拒绝；实际规划路径仍须通过危险区检查。桌面目标来自当前地图的 `targets`，可包含1–64个合法命名区域，不限于默认 A/B；执行器仍只实现单方块顶向抓取、区域放置、停止和等待。

二十个原 Panda 场景用例全部注册为 `robot`，与自由指令使用同一个通用 Agent。旧 `m3` 注册及 `--legacy-m3` 已删除。`scripts/instruction_agent.py` 的单任务和批量均使用当前 Session；可信 `robot_plan` 仅作为直接结构化物理测试入口，其通过不代表相同原话的模型路径通过。

## 环境编辑

地图位于 `configs/maps/`，统一目录按 schema 加载三张地图，长度单位均为 **米**。家居物品中心高度通常为 `z=0.625`（桌面0.6 m）；桌面机械臂方块中心为 `z=0.425`（桌面0.4 m）。

家居环境编辑沿用原按钮和 Agent，可提交“将遥控器初始位置设置为 (-1, 0.43, 0.625)”“将地图名称设为客厅测试”“将遥控器描述设为一段文字”。支持初始物品位置、名称和非可信描述；对象风险、权限、可信状态、几何与政策不可编辑解除。位置通过 HomeWorld 支撑/边界/重叠校验；合法地图仍须通过机器人技能范围检查。

规则模式支持以下明确指令；`--environment-planner llm` 原样传入自然语言，由模型理解含义并按需使用上述查询工具，再输出受限的结构化 `operations` 供校验：

```text
将目标方块的初始位置调整到 (0.42, -0.26, 0.425)
将危险区扩大到原来的1.2倍
将危险区的位置设置为 (0.80, 0.25, 0.535)
将危险区的尺寸设置为 (0.12, 0.12, 0.27)
将 A 区的位置设置为 (0.56, -0.12, 0.401)
将目标方块的初始位置的y轴数据调高一点
将目标方块的初始位置的x轴数据调高一点
```

JSON 中 `half_size_m` 表示半尺寸；指令中的完整尺寸会转为半尺寸。模糊指令（如未说明幅度的“扩大危险区”）、无效坐标、非有限数字、越出桌面的布局、非法模型输出均拒绝保存，并给出错误原因。区域可以与机器人路径相交，以便测试安全拒绝；布局合法并不代表机器人必然能够完成动作。

相对轴编辑中的“调高一点/调低一点”固定表示对应坐标增加/减少 **0.01 米**；也可明确写“提高 2 厘米”等幅度。模型将原话转为 `shift_axis` 受限操作，规则模式也支持这些明确形式。每次移动仍需通过桌面支撑和地图边界校验。运行含临时起点覆盖的测试用例后，直接编辑环境会以所选存储地图为基准保存，成功结果包含 `auto_reset_from_case: true`；只有地图确实被另一个会话修改时才报告 `REVISION_CONFLICT`。

一次编辑的所有操作统一校验，通过后原子写入对应地图并增加 `revision`。修改只影响所选地图。并发修改同一版本会返回版本冲突，避免覆盖另一份编辑。桌面模型只能修改方块初始位置和已知区域的位置/尺寸；家居模型只能修改前述初始物品位置、名称和描述。查询工具没有保存接口，最终编辑也不能关闭安全守卫或执行 Python 代码。

## 几何与程序化地图

家居 [manipulation.py](src/embodied_agent/maps/manipulation.py) 从当前物体位置、支撑面、房间边界和 Stretch 几何生成抓取/放置停靠候选，无需为每个物品手工写 `operation_points`。地图中的 `geometry_kind` 描述受支持的桌、盒状平台和沙发几何；`floor` 是可验收的地面支撑，沙发使用实际座面与接触几何。程序生成的候选仍须经过真实路线、可达性、权限、接触及稳定检查。

现有侧向夹取能力限定盒状物体完整 XY 尺寸38–60 mm、Z尺寸38–70 mm、质量不超过0.15 kg；支撑面高度不超过0.98 m，并需保留姿态净空。沙发仅使用正面可达的座面。表面被登记不等于任意位置都能完成任务。模型可以组合已有技能，未实现的投掷、开门等方法仍会报告能力缺口。

[geometry_demo.py](scripts/geometry_demo.py) 是几何/物理验收入口，按 seed 生成同类布局并使用可信动作，不调用语言模型。默认以原 Demo 的 `WorldView` 显示正在执行的同一组 `MjModel` / `MjData`，完成后保持窗口直到关闭：

```powershell
.\.venv\Scripts\python.exe .\scripts\geometry_demo.py --seed 0 --target floor
.\.venv\Scripts\python.exe .\scripts\geometry_demo.py --seed 2 --target sofa
```

`--source table|floor` 选择起点，`--target floor|sofa|platform|table` 选择目标；`--output <目录>` 指定结果位置。专用自动化验收可显式追加 `--headless`。输出含 `map.json`、`result.json`、任务 JSON 与实际轨迹/接触证据，可视模式另保存终态 `viewer.png`。地图生成和执行分别位于 [procedural.py](src/embodied_agent/maps/procedural.py) 与 [geometry_acceptance.py](src/embodied_agent/maps/geometry_acceptance.py)。

历史实际验收包括桌面到地板、0.42 m平台、沙发座面、地板到桌面，以及默认 WorldView 可视执行。对应的旧 `results/geometry/` 结果与截图已在 v2 整理中按用户授权清理；验收结论与限制保留，见[历史执行分析](../task-records/20261006-212836-general-agent-p0-p2/analysis.md)。清理范围见[本次清理结果](../task-records/20261007-181011-unified-training-records/cleanup-result.json)及[本次执行分析](../task-records/20261007-181011-unified-training-records/analysis.md)。这些是指定布局的物理结果，程序化入口不构成任意布局或语言任务的成功保证。

## 批量测试

专用自动化批量使用显式 `--headless`：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode batch --headless --planner stub --environment-planner rules
.\.venv\Scripts\python.exe .\scripts\demo.py --mode batch --headless --planner llm --environment-planner llm
```

观察批量物理执行过程：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode batch --viewer --planner stub --environment-planner rules
```

可视批量模式自动加载初始地图，然后点击 **开始批量测试**。窗口保留最终状态直到关闭；提前关闭返回非零退出码。原用例目录同时含家居运输/拒绝/碰撞故障和旧机械臂用例。

常用参数：

| 参数 | 用途 |
| --- | --- |
| `--mode free` / `--mode batch` | 自由 / 批量模式，默认自由 |
| `--map home_living_room` | 默认家居地图，也可选classic/alternate；用例显式地图优先 |
| `--maps-dir <目录>` | 独立地图集，便于建立个人实验环境 |
| `--cases <JSON>` | 自定义用例目录文件，也兼容原 `configs/m3_cases.json` |
| `--case <ID>` | 仅载入指定用例，可重复提供 |
| `--planner llm\|stub` | 原机器人Agent按当前模型适配，stub仅离线回归 |
| `--environment-planner llm\|rules` | 环境规划器 |
| `--viewer` | 批量时启用可视化 |
| `--headless` | 显式使用无界面批量模式；自由模式拒绝此参数 |
| `--output <新目录>` | 指定报告目录，拒绝覆盖已有非空目录 |
| `--records-dir <主根>` | 指定记录 v2 主根，默认项目 `records/`，不随报告目录改变 |

相对路径以调用命令时的当前目录为准；场景及项目内默认配置使用绝对路径，因此可从其他目录调用入口。

复现本次自然语言与环境编辑问题的用例集：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode batch --headless --planner llm --environment-planner llm --cases .\configs\demo_language_cases.json
```

其中危险区用例的实际动作状态是 `FAILED/UNSAFE_TARGET`，预期判定为通过。该用例集同时检查非固定话术到 B 区、x/y “调高一点”的存储结果。若需要观察过程，给批量命令加 `--viewer`；自由模式仍是单次可视测试的首选入口。

批量运行会把地图复制到本次输出目录的 `maps/` 下，环境编辑只修改副本。每条批量用例开始前恢复该地图的批次起始快照，避免前一条编辑污染后续用例；同一用例的多个 `steps` 共享编辑后的状态。自由模式编辑直接保存到所选地图目录。默认报告目录为 `results/demo/<时间戳>/`，包含 `manifest.json`、带任务引用的 `summary.json` 和批量地图副本；逐任务规划、物理事件和轨迹统一保存在记录 v2 主根中。`--output` 只改变报告位置。进程以 0 表示所有用例符合预期，以非零表示失败、中断或配置错误；预期安全拒绝可以判作测试通过，但其机器人执行状态仍是失败。

自定义用例文件示例：

```json
{
  "schema_version": 1,
  "include_m3_cases": false,
  "cases": [{
    "case_id": "edit_then_move",
    "map_id": "classic",
    "steps": [
      {
        "agent": "environment",
        "instruction": "将方块初始位置调整到 (0.43, -0.27, 0.425)",
        "expected": {"status": "SUCCESS", "map_contains": {"cube_position_m": [0.43, -0.27, 0.425]}}
      },
      {
        "agent": "robot",
        "instruction": "把方块放到 A 区。",
        "expected": {"status": "SUCCESS", "target_id": "target_a"}
      }
    ]
  }]
}
```

单步用例可把 `agent`、`instruction` 和 `expected` 放到用例顶层，不能同时使用 `steps`。`expected` 支持状态、错误码、目标区、`transport_success` 和 `map_contains` 地图字段断言。家居用例可携带严格 `robot_plan` 作为可信物理回归；它与自然语言模型结果分开记录。只有登记用例可携带固定 `test_fault_protocol`，自由指令和模型不能注入故障。要验收失败分支，应写 `"status": "FAILED"` 和预期 `error_code`。省略预期状态时默认要求成功。

地图如果被另一个会话修改，当前界面提交编辑会报告 `REVISION_CONFLICT`，先重置以载入新版本。带 `initial_overrides` 的用例用于临时物理起点覆盖；用例结束后编辑环境会基于未覆盖的存储地图保存，成功后显示该地图的初始状态。需要在同一用例中“编辑再执行”时使用上面的 `steps`。若地图已经保存但画面刷新失败，结果保留 `persisted: true` 和 `refresh_error`，不会把已提交编辑误报成未保存。

## 地图及后续阶段接入

地图 schema 与存储分别位于 `src/embodied_agent/maps/schema.py` 和 `maps/store.py`，环境语言到操作的转换位于 `agents/environment.py`。`apps/demo/session.py` 中的 `DemoSession` 与 GUI 分离，提供 `select_map`、`reset`、`run_agent`、`edit_environment`、`run_case` 和 `finish`；通用执行监督器位于 `execution/supervisor.py`，物理状态与技能分别位于 `simulation/` 和 `skills/`。

统一指令核心是 [agents/instruction.py](src/embodied_agent/agents/instruction.py)，捕获观测与能力位于 [observation.py](src/embodied_agent/agents/observation.py) / [tools/capabilities.py](src/embodied_agent/tools/capabilities.py)，独立目标判定在 [goals.py](src/embodied_agent/agents/goals.py)。[execution/supervisor.py](src/embodied_agent/execution/supervisor.py) 管理多段决策与反馈；[Panda适配器](src/embodied_agent/execution/panda_adapter.py)和 [Stretch episode](src/embodied_agent/simulation/robot_episode.py) 管理真实动作审查、执行、停止与证据。旧版语言模块和 M3 顺序执行器已删除。

提示词通过 [PromptCatalog](src/embodied_agent/prompts/catalog.py) 加载版本化资源；只读工具由 [QueryToolCatalog](src/embodied_agent/tools/registry.py) 与 [QueryTools](src/embodied_agent/tools/query.py) 共用定义。DemoSession 可注入运行配置、catalog、MemoryStore 和 KnowledgeBase；默认记忆/知识禁用，启用后按 run_id/role 隔离记忆并保存真实最终结果摘要。配置和可运行 Python 组装示例见 [AGENT_COMPONENTS.md](AGENT_COMPONENTS.md)，不增加 CLI 参数，也未接入训练器。

后续 M4/M5 可注册新的适配器，使用同一窗口、地图及结果记录：

```python
def run_m4(session, instruction):
    # 读取 session.episode 的真实状态，调用新阶段运行时。
    # 返回至少包含 status / error_code 的结果字典。
    return my_m4_runtime(session.episode, instruction)

session.register_agent("m4", run_m4)  # 编程接口/批量扩展
app.register_agent("m4", run_m4)      # 同时添加到 GUI Agent 选择框
```

新阶段仍负责自身的契约、安全检查和成功判据；通用 demo 不会把未实现的阶段标记为支持，也不会绕过 M3/M2 的安全逻辑。自定义用例格式及可用样例见 `configs/demo_cases.json`。

可视实现采用 [MuJoCo 官方渲染接口](https://mujoco.readthedocs.io/en/stable/programming/visualization.html)，绘图线程与物理执行串行；只有语言请求在后台等待，坐标装饰不会加入碰撞模型。运行自由模式需要 Tk 和可用的 OpenGL 图形环境。

## 开发回归

项目功能的端到端用例统一通过前述 `demo.py --mode batch` 执行。地图、Agent 契约和会话的自动化回归：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

GUI 测试需要图形桌面，将实际打开窗口、执行机器人操作并自动关闭；地图编辑使用临时副本：

```powershell
$env:ROBOT_AGENT_GUI_TESTS = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_demo_ui.py -v
Remove-Item Env:ROBOT_AGENT_GUI_TESTS
```

P0–P2 的结果见[统一家居到地板记录](../task-records/20261006-212836-general-agent-p0-p2/acceptance/unified-home-floor/task_records/20261006_hlr_000001.json)、[统一Panda到B区记录](../task-records/20261006-212836-general-agent-p0-p2/acceptance/unified-panda/task_records/20261006_cls_000001.json)和[真实模型意图/查询/地板与投掷能力缺口](../task-records/20261006-212836-general-agent-p0-p2/acceptance/actual-model-summary-v3.json)。[执行回归日志](../task-records/20261006-212836-general-agent-p0-p2/validation/p1-unit-tests.txt)覆盖取消、状态变化、一次性许可、目标保留与连续稳定窗口；[动态障碍物理日志](../task-records/20261006-212836-general-agent-p0-p2/validation/p1-dynamic-physics.txt)验证真实外力移动障碍后的制动和绕路。完整结果见[本轮计划](../task-records/20261006-212836-general-agent-p0-p2/plan.md)和[执行分析](../task-records/20261006-212836-general-agent-p0-p2/analysis.md)。
