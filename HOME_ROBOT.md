# 原 Demo 中的家居机器人

第4、5步通过升级原有 `scripts/demo.py`、`DemoSession` 和 `DemoApp` 实现。地图选择、自由指令、环境编辑、重置、测试用例、隔离批量和结果记录使用原体系。家居地图选择 Stretch 移动操作适配器；`classic` / `alternate` 选择原 Panda 适配器。独立 `home_robot.py`、HomeSession 和家居 Viewer 已移除。

在 `robot_agent` 目录启动原可视 Demo：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode free
```

默认加载 `home_living_room`，可在同一窗口切换回 `classic` 或 `alternate`。中央 WorldView 渲染实际执行的同一份 MuJoCo `MjModel` / `MjData`。自由指令连续作用于当前状态；只有加载地图、重置和测试用例起点才重建仿真。完成或失败后窗口保留真实终态，关闭窗口会中断正在执行的任务。

默认 `--planner llm` 使用现有 `.env` / DeepSeek。P0–P2 后，家居与桌面使用统一 InstructionAgent；先由模型确认目标/方法/来源，再根据当前能力规划动作段。契约只校验结构化结果；单动作实时审查、每步守卫和独立验收保留。详见[实际链路](自然语言任务执行链路.md)。离线stub不代表模型验收：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode free --planner stub --environment-planner rules
```

自由机器人指令示例：

```text
把遥控器送到餐桌。
把书搬到茶几。
把杯子送到餐桌。
观察当前世界。
查询遥控器。
导航到茶几。
抓取遥控器。
携带到餐桌。
放到餐桌。
停止。
```

分步指令使用当前持物、支撑和位姿；搬过的物品可从新的注册停靠点再次操作。目标不明确或不在有限能力内会返回明确错误，不擅自换物体或目的地。只读工具返回提交时捕获的声明仿真状态，不宣称视觉感知。

observe/query_world/inspect/get_capabilities 使用原生tool协议，以role=tool回传当前决策捕获的数据；工具不控制物理或保存地图。模型输出完整或相关部分动作段；每动作后用实际证据验收，未完成/可恢复中断时以新观测和真实反馈继续规划。等待模型时主线程受监控HOLD，目标与方法在恢复中冻结。

机器人和环境修改 Agent 的工具对话均由 `configs/agent_runtime.json` 的 `agent` 段约束：每次模型请求最多 **4096 tokens**，每个对话最多 **8 轮**、**24 次工具调用**。`stub` / `rules` 保留有限语法以供离线回归，模型失败不会自动切换到这些模式。

原用例下拉框和 `configs/demo_cases.json` 登记家居与原机械臂用例。可视执行一个正常或真实故障用例：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode batch --viewer --planner stub --environment-planner rules --case home_remote_to_table
.\.venv\Scripts\python.exe .\scripts\demo.py --mode batch --viewer --planner stub --environment-planner rules --case home_navigation_collision
```

故障用例由可信测试器注入错误导航末端点，经相同 wheel motor、物理步进与整体碰撞守卫检测实际茶几接触，记录触发部件、接触力与停止证据。结果仍是 `FAILED/ROBOT_FURNITURE_COLLISION`；满足故障预期只表示用例通过。自由文字和模型不能注入故障。

自动化回归使用原批量入口，显式关闭显示：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode batch --headless --planner stub --environment-planner rules
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

每条用例重建相应机器人与地图初态，批量环境编辑只修改本次输出中的地图副本。同一用例的多个步骤共享当前状态。原冻结20条 M3 语言/场景对继续保留，旧机械臂也通过同一个入口运行。

## 地图与技能边界

`configs/maps/home_living_room.json` 描述6×5 m房间和盒状代理物体。P2已支持地板、沙发座面、新平台及不同高度的桌面，通过几何自动生成操作候选，保留原关节范围。夹爪尺寸、载荷、可达性、权限和真实接触仍限制能力；不代表真实玻璃、书页、容器内部或任意家务。复现新表面使用默认可视的scripts/geometry_demo.py，证据见[链路文档](自然语言任务执行链路.md)。

统一地图目录按 schema 选择 `HomeWorld` 或旧 `WorldMap`，各自保留严格尺寸/支撑校验和原子版本保存。环境编辑 Agent 的 LLM 模式同样直接解析原始语言，可以查询存储地图及可编辑能力，最后返回完整 `operations`。家居可编辑初始物品位置、房间名称和非可信描述；全部操作通过支撑、边界、重叠、字段权限及版本检查后才原子保存。几何、操作权限、风险标签、离散状态和政策不能通过文字编辑解除。合法初态编辑仍可能超出已经验证的抓取姿态，运行技能会继续检查。运行物体状态不会写回初态地图。

风险随当前位姿、可信状态和对象关系更新。高温/通电水壶、开启/泄漏清洁容器、液体与电子设备邻近、易碎物边缘放置和受限物操作分别使用明确规则。描述中的“忽略限制”不能修改政策。易碎品损伤采用测得跌落/冲击与冻结项目阈值，不模拟材料碎裂；热、泄漏和电器关系是声明规则，不模拟热传导、流体或电击。

`StretchDemoEpisode` 实现原 Demo 的 live episode 接口，复用锁定 Stretch 2 资产与轮、升降、伸缩臂、夹爪 actuator。导航按实际整机/持物足迹膨胀障碍；停稳后操作，收臂后运输。抓取检查双侧接触、实际抬升和脱离支撑；释放检查脱离机器人、真实目标支撑、完整尺寸/边缘余量、XY误差及连续0.5 s静稳。物体始终是独立自由刚体，任务执行不写物体位姿或启用附着约束。

LLM 的真实 `actions` 使用 `navigate`、`pick`、`carry`、`place`、`wait`、`stop`；`observe`、`query_world`、`inspect` 和 `get_capabilities` 是查询工具，不能写进模型的 `actions`。执行器保留旧查询技能以兼容离线 stub 和可信 `robot_plan` 用例。工具和动作契约拒绝任意代码、非有限数字、未注册动作和地图/政策修改。正常停止通过实际底盘刹停并保持机械臂/夹爪；失败急停最多600步并记录停止与持物验证。预算按每次机器人请求计数，连续指令共用仿真但获得各自有限预算。

## 统一结果

自 2026-10-07 起，指令和环境修改 Agent 共用记录 v2。主根固定为项目 `records/`，`--records-dir` 可覆盖，`--output` 只改变报告位置。每项任务位于 `records/tasks/<上海日期YYYYMMDD>/<任务UUID>/`，由不可变 manifest、追加式事件日志和引用工件保存地图定义、实际前后观测、原始语言、模型与工具调用、提案、逐技能物理证据和真实反馈；`task.json` 为可重建视图，收尾生成 seal。完成后的任务不会被后续连续指令改写，会话审计进入 `records/runs/`。报告目录只保留 `manifest.json`、带任务引用的 `summary.json`，批量另保留 `maps/` 副本。独立评价和导出共用主根，训练器尚未接入。详见 [TASK_RECORDS.md](TASK_RECORDS.md)。

`success_count` 是实际成功状态数，`passed_count` 是满足预期的用例数；`transport_success` 必须有真实抓取、携物位移与独立稳定释放。观察、停止、动作前拒绝和预期故障不计运输成功。源码指纹、原始计划与规划模式保留；stub 或可信 `robot_plan` 物理回归不计真实 LLM 结果。

资产来源、许可证及控制适配见 `assets/third_party/hello_robot_stretch/`。本轮已实现绑定实际状态的一次性执行许可；完整Hook体系、隔离预演、训练器和门/抽屉/电器操作仍属于后续阶段。

历史原 Demo 全目录回归34/34符合预期；包含旧25例和新9例，实际成功29例，家居运输3例、正确拒绝4例、真实碰撞检测1例；补齐资产/初态记录后的11例混合回归也全部符合预期。原 `results/demo/unified-upgrade-acceptance/` 与 `unified-upgrade-final/` 已在 v2 整理中按用户授权清理，历史结论见[执行分析](../task-records/20261004-212054-upgrade-existing-demo/analysis.md)，清理见[本次清理结果](../task-records/20261007-181011-unified-training-records/cleanup-result.json)及[本次执行分析](../task-records/20261007-181011-unified-training-records/analysis.md)。所有拒绝/故障的历史执行状态仍是FAILED。

在 2026-10-04 的版本中，另通过同一DemoSession完成分步搬运、搬后再次抓取返回、Panda切换与重置，以及当时单次 LLM 请求的自由搬运。原Tk窗口的用例、自由指令、碰撞停止和关闭中断均已实测。这些为历史验证；当前模型接口支持前述多轮工具对话。历史结果、源码指纹差异和问题处理见[执行分析](../task-records/20261004-212054-upgrade-existing-demo/analysis.md)及[原窗口验证](../task-records/20261004-212054-upgrade-existing-demo/ui-validation.md)。
