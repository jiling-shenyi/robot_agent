# Robot Agent

MuJoCo 机器人操作 Agent 项目。问题定义、选型理由与工作原理见 [技术方案文档](TECHNICAL_DESIGN.md)，整体阶段安排见 [PROJECT_PLAN.md](PROJECT_PLAN.md)，M0/M1 执行规划见 [M0_M1_实施规划.md](M0_M1_实施规划.md)，M3 Agent 闭环实施计划见 [M3_实施计划.md](M3_实施计划.md)（已完成，限制见阶段记录）。

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

M2 逐个物理步计算 Panda 所有碰撞几何及方块到危险区几何的最近距离；任何对象进入 20 mm 安全边距都会以 `DANGER_ZONE_VIOLATION` 失败，并记录部件、阶段、步数和距离。新增危险区门槛的正式清单与完整复跑均为 10/10，分别见 `results/m2/acceptance_danger_zone_body_pair/` 和 `results/m2/acceptance_danger_zone_body_pair_repeat/`。已知限制是低位抓取时 Panda `link4` 与桌面在仿真中最深接触 `5.986 mm`；仅此接触对使用 `6.5 mm` 诊断上限，其余碰撞对仍限制在 `5 mm`。这项结果不代表真实 Panda 的安全间隙或硬件能力。分步计划、阈值依据和失败记录见 [`M2_实施计划.md`](M2_实施计划.md) 与 [`NOTES.md`](NOTES.md)。

## M3 Agent 闭环

M3 入口把有限范围内的自然语言目标解析为方块和 A/B 目标，再请求 DeepSeek 生成严格 JSON 技能计划。默认运行单条任务时打开实时 MuJoCo Viewer，窗口绑定正在执行的同一组 `MjModel`/`MjData`；单次演示命令：

```powershell
.\.venv\Scripts\python.exe .\scripts\m3_agent.py --instruction "Move the cube to the green target area."
.\.venv\Scripts\python.exe .\scripts\m3_agent.py --instruction "Place the cube in the blue target area."
```

只允许一次 `pick(cube, top)` 和一次 `place(target)`；模型输出在动作前经过 schema、目标、前置条件和扩张危险区路径检查。M2 按物理步运行的危险区、接触穿透、末端误差及仿真状态检查继续生效。每次技能后重读真实状态，最终由独立判定器确认目标区、释放、桌面支撑和稳定时间。

冻结的 20 条语言验收案例可批量运行；批量模式默认无界面，运行目录必须全新且不会覆盖既有证据：

```powershell
.\.venv\Scripts\python.exe .\scripts\m3_agent.py --batch --planner llm --output .\results\m3\my_llm_run
.\.venv\Scripts\python.exe .\scripts\m3_agent.py --batch --planner stub --output .\results\m3\my_stub_run
```

每次运行写 `manifest.json`、追加式 `events.jsonl` 和 `episodes.jsonl`、按 50 ms 采样的 `trajectory.csv` 及 `summary.json`。`--planner stub` 用于开发/回归，不能作为实际 LLM 结果。M3 自身冻结了 low reasoning、disabled thinking 与 256 输出 token 上限；它独立于 DeepSeek smoke 脚本读取的 `.env` 推理设置。其余预算、提示词版本和 20 条输入见 `configs/m3_runtime.json` 与 `configs/m3_cases.json`；详细边界及验收门槛见 [`M3_实施计划.md`](M3_实施计划.md)。

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

- DeepSeek Chat Completions API 已完成单次连通性验证。
- M0 基线已建立，本地提交为 `9991c06`。
- M1 G0/G1/G1b 已通过：三处目标的末端跟踪最大停留误差为 0.19 mm；夹爪空载开合、桌面方块静置和项目场景截图均已记录。
- M2 确定性抓取与放置已完成：冻结清单 10/10 通过，完整复跑 10/10，结果逐条一致；已知 link4/桌面接触例外见上文和 `NOTES.md`。
- M3 受约束 Agent 闭环已完成：stub 20/20、真实 DeepSeek 预登记指令 20/20；默认 Viewer 的 A/B 单次任务均通过独立终态判定。Viewer 进程生命周期已验证，但本次锁屏环境未能独立核对窗口画面；细节见 [M3 实施计划](M3_实施计划.md)。
- M3 使用有限指令和结构化仿真真值，不代表任意语言理解、视觉感知、自动恢复、真实机器人或 Sim2Real 能力。M4 失败检测标定和 M5 恢复尚未验证。
