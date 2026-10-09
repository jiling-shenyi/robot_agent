# Robot Agent

下一阶段 LangChain 实践的接入架构、分阶段改造、依赖验证、机器人验收和回退安排见 [LangChain 技术方案](LANGCHAIN_TECHNICAL_PLAN.md)；该文档是实施设计，新增后端与参数尚未实现。

已按路线图 A0 整理现有模块，并删除 src 中的旧兼容文件；代码直接使用拆分后的模块路径。脚本启动命令、模块职责、依赖方向及运行产物规则见[模块说明](MODULES.md)。

P0、P1、P2 改造后的实际架构见[自然语言任务执行链路与指令 Agent 原理](自然语言任务执行链路.md)。Panda/Stretch 使用同一个 InstructionAgent：模型先确认目标和方法，再规划动作段；只读查询通过原生工具回传，真实执行的反馈可触发继续规划。等待模型时保持物理监督，每个动作实时审查并独立验收。已支持几何生成操作姿态、地板/沙发座面/新平台、动态区域和受阻绕行；实际能力仍受机器人几何、权限与守卫约束。

提示词、查询工具、记忆和知识库已拆为独立的 `prompts/`、`tools/`、`memory/`、`knowledge/` 包，可通过 `components` 配置和 Agent/DemoSession 构造参数替换。默认记忆/知识禁用；启用后按角色隔离保存真实结果摘要，知识检索使用本地词面匹配。两个 Agent 共用 v2 事件/工件记录、独立评价和六种数据导出；奖励须明确选择，强化学习训练器尚未接入。配置和注入见 [Agent 组件说明](AGENT_COMPONENTS.md)，记录与训练资格见 [TASK_RECORDS.md](TASK_RECORDS.md)。

## 家居地图与移动机器人（第4、5步 / A2、A3）

原 Demo 已升级为家居机器人：默认 `home_living_room` 选择 Hello Robot Stretch 2，`classic` / `alternate` 选择原 Panda 适配器。地图、自由指令、环境编辑、重置、用例、批量和结果共用原 `DemoSession` / `DemoApp`，独立家居入口、会话和 Viewer 已移除。首版限定0.6 m桌面、50 mm盒状物品和验证过的侧向抓取姿态。

启动原可视 Demo，默认加载家居地图：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode free
```

输入“把遥控器送到餐桌”“把书搬到茶几”或“把杯子送到餐桌”，也可在原用例下拉框运行家居用例。WorldView 同步真实物理执行，完成或失败后保留终态直到关闭。连续指令、规则拒绝、真实控制碰撞故障及能力边界见 [HOME_ROBOT.md](HOME_ROBOT.md)。完整 Hook、预演和训练仍在后续阶段。

## 通用模拟世界 Demo（M3 及后续阶段）

[`scripts/demo.py`](scripts/demo.py) 是机器人及原 M3 机械臂共用的自由测试与批量测试入口。默认可视化，专用自动化批量使用显式 `--headless`。Demo 使用 MuJoCo 实际仿真状态，提供统一地图选择、坐标网格、机器人 Agent、环境修改、重置和测试用例执行。

在项目目录启动可视自由测试：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode free
```

窗口默认加载家居地图；左上角可选择地图并点击“加载地图”。家居使用移动操作技能，`classic` 和 `alternate` 使用原机械臂技能；自然语言均通过现有模型接口、计划契约和逐步守卫执行。右上角从同一用例目录选择测试并观察过程。

环境修改 Agent 可调整桌面方块/区域，或家居物品初始位置、地图名称及非可信描述；有效修改保存到所选地图。家居风险状态和操作权限不能由描述解除。重置重新读取最新保存的初态；自由指令持续作用于当前仿真，需要重新开始时使用重置或测试用例入口。

默认使用 `.env` 中配置的 DeepSeek 模型。没有 API 配置时，可用离线模式启动自由测试：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode free --planner stub --environment-planner rules
```

`stub` 和 `rules` 是有限的离线开发解析器，不代表真实模型能力。桌面地图的危险区目标返回 `UNSAFE_TARGET`；家居按当前物品权限、状态和有限停靠点检查目标。相对坐标指令中的“x/y 轴调高一点”默认增加0.01米。

专用自动化批量评测显式关闭窗口：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode batch --headless --planner llm --environment-planner llm
```

可选 `--viewer` 在批量运行时显示场景；例如离线可视化批量测试：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode batch --viewer --planner stub --environment-planner rules
```

常用参数包括 `--map <地图ID>`、`--cases <JSON文件>`、可重复的 `--case <用例ID>`、`--output <新报告目录>` 和 `--records-dir <主记录根>`。运行报告默认在 `results/demo/<时间戳>/`，真实任务事件与工件默认在 `PROJECT_ROOT/records`，指定 `--output` 不改变主记录根。批量环境编辑使用本次运行的地图副本。自由模式要求 Tk/OpenGL，不能使用 `--headless`。

完整 UI 操作、环境编辑指令、用例格式、批量隔离及后续阶段接入方法见[通用 Demo 使用说明](DEMO.md)。复现自然语言和环境编辑用例可运行 `configs/demo_language_cases.json`。

MuJoCo 机器人操作 Agent 项目。问题定义、选型理由与工作原理见 [技术方案文档](TECHNICAL_DESIGN.md)，当前模块安排见 [MODULES.md](MODULES.md)，M0/M1/M2/M3 的历史执行结论及限制见 [NOTES.md](NOTES.md)。

## MuJoCo / M1

在仓库根目录使用项目虚拟环境运行 M1 smoke：

```powershell
.\.venv\Scripts\python.exe .\scripts\m1_track_mocap.py
```

脚本默认打开 MuJoCo viewer，只显示实时任务场景，左右调试面板默认隐藏。viewer 与 smoke 共用同一仿真实例；鼠标左键拖动旋转视角，右键拖动平移，滚轮缩放。关闭 viewer 会停止测试；任务完成后窗口自动关闭。脚本从 `home_scene` 初始化，先对齐 mocap 目标和真实 `ee_site`，再做三个位姿的短距离跟踪、返回 home、空载夹爪三次开合和方块静置检查。结果保存在 `results/m1/`：`tracking.csv`、`model_info.json` 和 `screenshots/panda_task.png`。无人值守运行时可加 `--headless` 跳过 viewer 并尽快完成仿真：

```powershell
.\.venv\Scripts\python.exe .\scripts\m1_track_mocap.py --headless
```

需要调试面板时可加 `--show-ui`。部分 Windows AMD OpenGL 驱动会将原生面板绘制成大块黑色遮挡和错位控件（[MuJoCo 官方问题 #639](https://github.com/google-deepmind/mujoco/issues/639)）；已在本机复现。默认隐藏面板可正常显示场景；此类环境中请保持默认模式，并避免用 Tab / Shift+Tab 重新打开面板。

Panda 上游文件及 Apache-2.0 许可证位于 `assets/third_party/franka_emika_panda/`；Menagerie 提交、模型 OID 和项目派生 XML 的修改记录见 [`assets/third_party/README.md`](assets/third_party/README.md)。M1 仅验证仿真末端跟踪、空载夹爪和场景静置，不代表完成抓取、搬运或放置。

共享场景中，绿色和蓝色区域是放置目标；桌面后侧红色半透明柱体是危险区可视标记。标记不产生物理碰撞，M2 通过软件距离门槛约束 Panda 碰撞几何和方块。

## MuJoCo / M2 确定性 Pick/Place

`scripts/m2_pick_place.py` 在 M1 场景里执行单方块顶向抓取与放置，使用仿真真值和 MuJoCo 实际指尖碰撞，不调用 LLM，也不直接改写方块位姿。单次任务默认打开实时 viewer；画面使用正在执行物理步进的同一组 `MjModel`/`MjData`，隐藏调试侧栏并按仿真时间播放。关闭 viewer 会中止本次 episode，且不写入最终验收证据。单次运行示例：

```powershell
.\.venv\Scripts\python.exe .\scripts\m2_pick_place.py --target a --seed 0 --output .\results\m2\demo_a_seed0
```

单次任务可用 `--headless` 关闭 viewer。冻结的 10 个验收场景默认 headless；若要逐条打开 viewer，添加 `--viewer`（每个 episode 依次显示一个窗口）：

```powershell
\.venv\Scripts\python.exe .\scripts\m2_pick_place.py --batch --headless --output .\results\m2\my_acceptance_run
```

需要可视化批量 episode 时，把 `--headless` 换成 `--viewer`。

种子清单和阈值在 `configs/m2_scenarios.json`、`configs/m2_thresholds.json`。脚本不会覆盖已有证据；指定新输出目录，或明确添加 `--overwrite`。每个运行目录包含 episode JSONL、50 ms 采样轨迹 CSV 和由原始记录生成的汇总 JSON。

M2 逐个物理步计算 Panda 所有碰撞几何及方块到危险区几何的最近距离；任何对象进入 20 mm 安全边距都会以 `DANGER_ZONE_VIOLATION` 失败，并记录部件、阶段、步数和距离。新增危险区门槛的正式清单与完整复跑均为 10/10，分别见 `results/m2/acceptance_danger_zone_body_pair/` 和 `results/m2/acceptance_danger_zone_body_pair_repeat/`。已知限制是低位抓取时 Panda `link4` 与桌面在仿真中最深接触 `5.986 mm`；仅此接触对使用 `6.5 mm` 诊断上限，其余碰撞对仍限制在 `5 mm`。这项结果不代表真实 Panda 的安全间隙或硬件能力。阈值依据和历史失败记录见 [NOTES.md](NOTES.md)。

## 通用指令 Agent 闭环

所有机器人自然语言入口统一使用 `InstructionAgent`。LLM 先解释原始指令的目标、方法和来源，再查询实际观测与当前能力，提出动作段；执行监督器审核并执行真实动作、独立验收，必要时将实际失败与进展交给同一个 Agent 重新规划。桌面 Panda 与家居 Stretch 使用不同物理适配器。单次任务默认显示执行中的同一组 `MjModel`/`MjData`：

```powershell
.\.venv\Scripts\python.exe .\scripts\instruction_agent.py --instruction "Move the cube to the green target area."
.\.venv\Scripts\python.exe .\scripts\demo.py --mode free --planner llm --environment-planner llm
```

不再要求固定两步模板。动作仍受实际机器人能力、当前前置条件、路径和每步安全检查约束，最终由独立判定器检查接触、释放、支撑和连续稳定时间。

原二十个 Panda 语言/场景用例只保留输入数据，全部通过通用 Agent 执行；批量也默认可视化，自动化评测显式添加 `--headless`。输出目录必须全新：

```powershell
.\.venv\Scripts\python.exe .\scripts\instruction_agent.py --batch --planner llm --output .\results\instruction\my_llm_run
.\.venv\Scripts\python.exe .\scripts\instruction_agent.py --batch --headless --planner stub --output .\results\instruction\my_stub_run
```

每个任务保存到 `records/tasks/<上海开始日期>/<task_uuid>/`，以 manifest、追加事件和内容地址工件为事实源，终态 seal 校验封存；`task.json` 与索引可重建。默认根固定为 `PROJECT_ROOT/records`，只有 `--records-dir` 改变主根；`results` 只保存报告和引用。模型原始输出、实际动作和用户可信目标分别留证，`SUCCESS` 不自动成为奖励或正确示范。旧 v1 Agent 记录及 `evaluation/task_records.py` 已移除，无兼容模块。查询、独立评价和六种导出见 [TASK_RECORDS.md](TASK_RECORDS.md)。

配置为 [configs/agent_runtime.json](configs/agent_runtime.json)：每次输出4096 tokens、每个工具对话8轮/24次调用、最多2次提案修正；整个任务共用24次模型请求、131072 tokens、128次动作、4次恢复和300秒。`stub`/`rules` 仅用于离线回归，LLM失败不会回退到规则模式。旧 Agent、旧 M3 执行器、`m3` 注册、`--legacy-m3` 和旧启动脚本已删除。

### 危险区违规分支示例

M1 的 headless smoke 会检查共享场景中的危险区名称、世界坐标、尺寸、颜色、桌面位置关系和纯视觉碰撞掩码：

```powershell
.\.venv\Scripts\python.exe .\scripts\m1_track_mocap.py --output .\results\m1\danger_zone_config_check
```

M1 smoke 默认显示 Viewer；自动化或无图形环境可显式追加 `--headless`。

下面的故障注入示例会从 home 姿态通过 M2 的 mocap 控制链路驱动 Panda 机械臂朝危险区运动。逐物理步守卫检测到机械臂碰到 20 mm 安全边距后会立即停止并报告 `DANGER_ZONE_VIOLATION`；窗口显示实际机械臂停下的位置、触发部件和间距，并保持打开直到你关闭。方块留在原位，不用于伪造违规：

```powershell
.\.venv\Scripts\python.exe .\scripts\m2_danger_zone_fault_injection.py --output .\results\m2\danger_zone_robot_motion
```

该故障注入是独立诊断，不属于正常 pick/place 验收场景。预期结果为 M1 危险区配置检查通过、Panda 碰撞几何触发 20 mm 安全边距违规、M2 违规分支触发；机械臂会在进入危险体积前被安全守卫拦停。证据文件 `summary.json` 和 `episodes.jsonl` 会在 Viewer 保持打开时写入。无图形环境可显式追加 `--headless`。
无界面运行时可为该命令追加 `--headless`；重复使用同一结果目录时，`--overwrite` 会替换该目录中的诊断证据。

## DeepSeek API 配置

本地 API key 保存在仓库根目录的 `.env`，该文件已由 `.gitignore` 排除，不要将它加入 Git。`.env.example` 只包含变量名和非敏感默认配置。

新环境首次配置时，仅在 `.env` 尚不存在的情况下复制模板，再把 key 填入本机 `.env`：

```powershell
if (-not (Test-Path .\.env)) { Copy-Item .\.env.example .\.env }
notepad .\.env
```

当前 `.venv` 已安装配置和烟测所需依赖。复建环境时可运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-lock.txt
```

配置项如下：

| 变量 | 当前值/用途 |
| --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥，保存在本地 `.env` |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | `deepseek-flash` |
| `DEEPSEEK_REASONING_EFFORT` | `high` |
| `DEEPSEEK_THINKING` | `enabled` |

验证本地配置和 API 连通性：

```powershell
.\.venv\Scripts\python.exe .\scripts\deepseek_smoke.py
```

脚本从 `.env` 读取 key，只发送一条简短请求；不会打印 key。收到 `DeepSeek API connection OK` 和 `READY` 表示配置有效。后续规划器应复用相同的环境变量，并在执行任何模型计划前通过项目闸门校验。

## 当前进度

- 路线图第4、5步（A2/A3）已完成：独立家居地图、状态/关系风险与Stretch真实导航/抓取/携物/释放；3搬运+4拒绝+1碰撞故障验收8/8符合预期。完整测试94项，92通过、2项旧交互测试条件跳过；原生Viewer房间、搬运与故障另已实测。[使用说明](HOME_ROBOT.md)、[执行分析](../task-records/20261004-194446-home-mobile-robot/analysis.md)。

- DeepSeek Chat Completions API 已完成单次连通性验证。
- M0 基线已建立，本地提交为 `9991c06`。
- M1 G0/G1/G1b 已通过：三处目标的末端跟踪最大停留误差为 0.19 mm；夹爪空载开合、桌面方块静置和项目场景截图均已记录。
- M2 确定性抓取与放置已完成：冻结清单 10/10 通过，完整复跑 10/10，结果逐条一致；已知 link4/桌面接触例外见上文和 `NOTES.md`。
- M3 受约束 Agent 闭环已完成：stub 20/20、真实 DeepSeek 预登记指令 20/20；默认 Viewer 的 A/B 单次任务均通过独立终态判定。Viewer 进程生命周期已验证，但当时的锁屏环境未能独立核对窗口画面；历史细节见 [NOTES.md](NOTES.md)。
- M3 使用有限指令和结构化仿真真值，不代表任意语言理解、视觉感知、自动恢复、真实机器人或 Sim2Real 能力。M4 失败检测标定和 M5 恢复尚未验证。
