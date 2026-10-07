# 项目决策与实验记录

本文件记录项目范围、可复现基线、关键决策和实验结果。计划、环境查询与仿真结果分开记录；没有运行或观察到的内容不标记为通过。后续实验按“日期与阶段、环境/模型、目标与配置、观察结果、验收门槛、失败信息、下一步、关联文件/提交”追加。

**2026-10-07 历史产物清理说明：** 下方历史章节的测试数字、失败原因和物理/可视验收结论保持原意；旧路径与“保留”描述对应当时版本。记录 v2 整理已按用户授权删除旧 Agent 主记录及 `results/demo`、`results/home`、`results/geometry`、`results/m3` 下的旧执行详情、摘要和截图，删除清单共676件；M1/M2留存文件的哈希核对未改变。失效旧产物链接已移除，历史执行分析和验证日志继续留存。清理范围及结果见[清理结果](../task-records/20261007-181011-unified-training-records/cleanup-result.json)和[本次执行分析](../task-records/20261007-181011-unified-training-records/analysis.md)。当前唯一记录主根、摘要引用与查询规则以 [TASK_RECORDS.md](TASK_RECORDS.md) 为准。

**同步至 2026-10-04：** M0–M3 历史保留；通用 Demo、两幅桌面地图、环境编辑、自然语言修复与 A0 整理见下方追加记录。早期记录中的 `planner.py`、`runtime.py` 等为当时路径，当前源码入口以 [MODULES.md](MODULES.md) 为准。本次同步核对现有源码、配置和留存证据，没有重新运行仿真、GUI 或模型请求。

## 决策记录

### D-001：锁定第一版任务边界与仿真验收基准

- **日期 / 阶段：** 2026-09-27 / M0
- **状态：** 已采纳；模型几何和数值参数待 M1 加载模型后补录。
- **背景：** 项目需要先判断仿真控制与执行状态能否可靠复现，再开发更高层的技能和 Agent。当前没有项目场景或已验证的 Panda 运动结果。
- **决策：**
  1. M0/M1 只验证 MuJoCo 中的操作执行与底层控制；不据此声称真实机器人可用、具备视觉感知能力或实现 Sim2Real。M1 不把 LLM 接入场景控制链路。DeepSeek API 连通性只属于独立配置烟测，不代表规划器或仿真已集成。
  2. 第一版任务场景为一台 Franka Panda、一个可移动方块和两个互不重叠的桌面目标区。每次任务由用户指令选择一个目标区；初始方块和目标区只在模型验证后的可达工作区内变化。后续成功判定独立于 Agent 返回值：夹爪张开，方块水平投影完整落入目标区并留有标定边界余量，方块由支撑面承托且速度低于标定阈值并稳定约 0.5 秒，且不再与夹爪持续接触。边界余量和速度阈值需在开发场景中标定并在评测前冻结。
  3. 场景采用实际加载 MJCF 的 MuJoCo world frame；项目坐标约定为 Z 轴向上、长度用米、角度用弧度。Panda 初始关节状态采用所锁定模型的 `home` keyframe。真实末端位置以附着在 Panda `hand` 部位、经模型核实的末端 site 为准；若上游模型没有合适 site，则在项目模型中增加并记录其相对 `hand` 的位姿。M1 记录实际模型的世界原点、`home` 关节值、site 名称及定义和坐标变换；这些数值在模型尚未加载前不猜定。
  4. M1 先评估该末端 site 的**实际位置**与目标位姿之误差，不用 mocap 目标位姿代替实测末端位姿。2 cm 暂作初始工程验收候选值；须结合方块尺寸、目标区尺寸和实际跟踪数据确认后再冻结，不能称为 MuJoCo 或硬件精度保证。
- **理由：** 把任务限制在可观察的仿真真值和单物体操作，便于区分模型目标与实际末端/物体状态，也让 M1 的结果可解释；模型几何参数则必须从实际采用的 XML 核实。
- **环境与模型基线（2026-09-27，只读查询）：** Windows 项目 `.venv` 为 Python 3.13.9；`mujoco 3.14.0`、`mujoco-menagerie 2026.9.2`、`numpy 2.5.3`、`pydantic 2.13.5`、`openai 3.19.2`、`python-dotenv 1.2.3` 已安装，版本与 `requirements-lock.txt` 一致。Menagerie 元数据查询返回仓库提交 `c96a32d28fb5da84da38c1da4d749e7a13212855`、Panda 条目 OID `3d2262eeb81ecec19abfa31dd509e35abbb33e67`、许可证标识 `Apache-2.0`。本条记录建立时只查询了包元数据，尚未下载或加载 Panda XML/mesh，也未检查模型随附的许可证文件；M1 须据实际获取的文件复核并保留许可证。
- **观察与验收结果：** `requirements-lock.txt` 存在且包含上述版本；初始 `.gitignore` 排除 `.venv/`、`.env`、Python 缓存，并全量忽略 `results/`。仓库在 `main` 上尚无提交，Git 显示 `origin/main [gone]`；工作区已有基线文件暂存。环境版本和 Menagerie 元数据查询成功。此记录写明了任务边界及坐标约定。**此条记录建立时 M0 尚未全部验收：** 首个 Git 提交、模型文件来源记录和仿真验证仍待后续完成。未运行仿真，因此没有模型加载、末端跟踪、夹爪或物体稳定性通过结果。
- **失败 / 错误：** 本次只读查询未观察到错误；仿真及控制尚未执行，不能据此报告无仿真故障。
- **下一步及理由：** 完成并审核 M0 基线提交；建立 `assets/third_party/README.md` 来源记录；进入 M1 时获取并检查上述 Menagerie Panda 版本的模型与许可证，记录 `home` 和末端 site 的真实定义，再建立场景并采集实际末端跟踪数据。
- **关联文件 / Git：** `PROJECT_PLAN.md`、`M0_M1_实施规划.md`、`requirements-lock.txt`、`.gitignore`；本记录对应的 Git commit 尚不存在。

### D-002：冻结 M2 确定性抓取与放置控制及验收阈值

- **日期 / 阶段：** 2026-09-27 / M2
- **状态：** 已采纳；配置已冻结，10 个正式场景及完整复跑均通过。
- **背景：** M1 只通过末端跟踪和空载夹爪开合，没有验证真实碰撞抓取。M2 需要确认方块确实由 Panda 手指夹起、搬运、释放并稳定放入目标区。
- **决策：** 采用固定 home 末端姿态、顶向接近和 mocap 分段插值；每物理步目标位移上限 `0.08 mm`，到点停留 150 步。抓取须由左右 finger 与 cube 的 MuJoCo 接触及抬升结果确认；不移动方块 qpos、不使用持物 weld。成功判定使用方块 8 个角点的水平投影，目标边界内缩 `10 mm`；桌面支撑高度误差上限 `3 mm`，线速度上限 `0.01 m/s`、角速度上限 `0.1 rad/s`，连续稳定 250 步（`0.5 s`），且夹爪张开并与方块分离。
- **穿透门槛：** 一般接触全局限制为 `5 mm`。开发 sweep 在低位抓取时反复观察到 `link4:geom28 <-> table:table_top` 接触，最深 `5.986 mm`；只对该 pair 明确配置 `6.5 mm` 上限并逐 episode 记录，其余 pair 仍受 `5 mm` 限制。此例外是已知仿真场景连杆/桌面接触，不表示机器人安全间隙或硬件通过。
- **理由：** `0.2 mm/step` 初版轨迹在长距离搬运中有较深的连杆/桌面接触，也曾在放置触桌时被过早判作运输滑落。减至 `0.08 mm/step` 后，正式场景均保持末端误差低于 `3.1 mm`；最多出现 1 个物理步的单侧接触间隙，动作段终点仍需恢复双侧接触。放置下降阶段若双侧接触结束，只有方块已经由桌面支撑、位于目标区且高度合格才继续释放判定。
- **冻结场景：** `configs/m2_scenarios.json` 内 5 个可达初始位置各运行到 A、B 一次，共 10 条；目标 A/B 各 5 条。该清单、`configs/m2_thresholds.json`、场景 XML 与依赖版本作为复现实验配置保存。
- **验收结果：** 正式运行 `10/10 SUCCESS`；同一冻结清单完整复跑也为 `10/10 SUCCESS`。两轮逐条状态、仿真步数和最终方块位置完全一致。最大实际末端位置误差 `0.003025 m`；最深接触 `-0.005986 m`，且只触发上述明确 pair 例外。所有成功 episode 稳定窗口均为 250 步；无 MuJoCo warning 或非有限状态。
- **边界：** 结果只覆盖当前 MuJoCo 3.14.0、固定 Panda 派生模型、单方块、固定姿态与确定性控制；不包含 LLM、视觉、扰动、失败恢复、真实 Panda 或 Sim2Real。
- **关联文件：** `M2_实施计划.md`、`scripts/m2_pick_place.py`、`configs/m2_scenarios.json`、`configs/m2_thresholds.json`、`results/m2/acceptance/`、`results/m2/acceptance_repeat/`。

## 阶段执行记录

### M0：项目与环境基线

- **日期与阶段：** 2026-09-27 / M0 完成
- **环境版本 / 模型标识：** Python 3.13.9、MuJoCo 3.14.0、Menagerie 2026.9.2、NumPy 2.5.3、Pydantic 2.13.5。`pip freeze` 与 `requirements-lock.txt` 比对一致。Menagerie Panda 提交 `c96a32d28fb5da84da38c1da4d749e7a13212855`，条目 OID `3d2262eeb81ecec19abfa31dd509e35abbb33e67`，许可证标识 Apache-2.0。
- **目标 / 操作与配置：** 固定第一版任务边界、坐标基准和 M1 末端误差判据；补齐忽略本机环境/密钥及临时运行文件的 `.gitignore`；建立第三方来源登记；逐项检查 staged files。
- **观察到的结果 / 验收门槛及结果：** Python 与核心依赖可从仓库 `.venv` 查询；锁文件一致；NOTES 首条决策已建立；`.gitignore` 保留 `results/m1/` 验收证据，同时忽略 `.env`、`.venv/`、缓存和临时日志。首个本地提交 `9991c06a21e8c253cdd10a730be3afc9b64e42d0` 已创建；提交清单不含 `.env` 或 `.venv/`。远端跟踪引用仍显示 `origin/main [gone]`，不影响本地基线。
- **失败现象 / 错误信息：** 无。模型加载和仿真不属于 M0 的完成证据。
- **下一步决定与理由：** 进入 M1，从已锁定的 Menagerie 条目复制 Panda 文件与许可证；保持上游 XML 原样，以独立派生文件实施控制所需修改。
- **关联文件或 Git commit：** `9991c06a21e8c253cdd10a730be3afc9b64e42d0`；`requirements-lock.txt`、`.gitignore`、`assets/third_party/README.md`。

### M1：MuJoCo 与 Panda 跑通

- **日期与阶段：** 2026-09-27 / M1 完成
- **环境版本 / 模型标识：** Python 3.13.9、MuJoCo 3.14.0；Menagerie 2026.9.2，仓库提交 `c96a32d28fb5da84da38c1da4d749e7a13212855`，Panda OID `3d2262eeb81ecec19abfa31dd509e35abbb33e67`，Apache-2.0。仓库内保留上游 `panda.xml`、许可证和它引用的全部 67 个 mesh；逐文件与本机 Menagerie 缓存比对 SHA-256 均一致。上游 XML SHA-256 为 `96ad67da03710f17f798c9478fd9e9efdf24a3bf8359f05e456dd9fb158ea273`。
- **目标：** 验证 G0 模型加载、G1 mocap 到真实末端 site 的短距离跟踪、G1b 空载夹爪独立开合和场景静置稳定性。
- **操作与配置：** `assets/scene/panda_task.xml` 编译为 15 bodies、86 geoms、8 actuators、2 equalities。步长 0.002 s；机械臂 actuator 1–7 在禁用组 0，夹爪 actuator 8 在启用组 1，控制范围 0–255。上游 `home` 机械臂关节值为 `(0, 0, 0, -1.57079, 0, 1.57079, -0.7853)`，夹爪关节为 `(0.04, 0.04)`。`ee_site` 位于 `hand` 局部 `(0, 0, 0.1034) m`，home 世界位姿为 `(0.5544995, 0, 0.5211024) m`（四元数按 MuJoCo WXYZ 记）。以该实测位姿初始化 mocap，再插值移动到三个相距 2.5 cm 的安全目标并返回 home；每段 300 步运动、150 步停留。夹爪以 0/255 命令做 3 次闭合/张开循环。场景桌面顶面 `z=0.4 m`，自由方块半边长 2.5 cm，两个无碰撞标记区域中心为 `(0.58, -0.12, 0.401)` 和 `(0.58, 0.12, 0.401) m`，每个区域半尺寸为 `(0.075, 0.075, 0.001) m`。
- **观察到的结果：** G0 通过，项目 MJCF 可编译，名称和 actuator 范围可查询，并生成 1200×900 场景截图。G1 通过；三个目标停留阶段最大位置误差分别为 `0.0001337 m`、`0.0001309 m`、`0.0001859 m`，返回 home 的最大误差为 `0.0001883 m`。结合目标区域与方块几何的名义边界余量 `0.05 m`，将 `0.02 m` 确认为 M1 跟踪 smoke 的项目工程阈值；这不代表任务放置成功率或硬件精度。G1b 通过；夹爪关/开时两指平均关节位置分别为 `0.00135 m` 与 `0.03866 m`，site 最大漂移为 `0.000189 m`。空闲结束时方块底面 `z=0.3999205 m`，桌面顶面 `z=0.4 m`，线速度 `1.54e-10 m/s`；唯一记录到的接触是方块与桌面的预期支撑接触，最小接触距离 `-0.0000795 m`，低于 5 mm 穿透判据。无 warning、NaN 或未解释求解异常。三个门槛均通过。
- **失败现象 / 错误信息：** 首次 smoke 用 Panda 原始 `home` keyframe 重置包含自由方块的场景时，新增方块 freejoint 被置于世界原点，测试中被机械臂碰撞并离开桌面。修正为场景专用 `home_scene` keyframe，显式记录机器人与方块完整 qpos，并将方块放在跟踪路线之外；随后静置稳定。Windows 下 MuJoCo 对含中文的绝对 XML 路径解析失败，脚本改为切到仓库根目录并使用 ASCII 相对路径。初次离屏渲染请求 1200 px 宽度超过默认 640 px framebuffer，已在场景 visual 配置中将 offscreen 缓冲区设为 1200×900。
- **验收门槛及结果：** G0 PASS；G1 PASS（20 mm 误差限）；G1b PASS。脚本记录 435 个 CSV 样本；model_info、误差轨迹和场景截图均生成。验证对象是仿真跟踪、空载夹爪及静置场景；没有验证抓取、抬升、搬运、释放或完整放置。
- **下一步决定与理由：** 进入 M2 时在本场景基础上实现确定性 pick/place，再单独验证真实碰撞接触与放置后置条件。保留 M1 原始轨迹和模型摘要作为后续阶段基线。
- **关联文件或 Git commit：** `assets/third_party/franka_emika_panda/`、`assets/scene/panda_task.xml`、`scripts/m1_track_mocap.py`、`results/m1/tracking.csv`、`results/m1/model_info.json`、`results/m1/screenshots/panda_task.png`；M1 改动与本记录一起提交，精确提交可由 Git 历史查询。

### M1 后续：Smoke 实时 Viewer

- **日期与阶段：** 2026-09-27 / M1 可视化补充
- **环境 / 配置：** MuJoCo 3.14.0；默认运行 `python scripts/m1_track_mocap.py` 时以 `mujoco.viewer.launch_passive(model, data)` 打开被动 viewer，viewer 与烟测共用同一 `MjModel`、`MjData`。相机初始视角对准 Panda、桌面和方块；viewer 每 10 个物理步同步一次，并按模型步长节流到实时速度。`--headless` 保留原先快速、无窗口的运行方式。
- **观察与验收结果：** 默认图形模式进程完成整段 smoke，跟踪、夹爪、场景静置和穿透检查均 PASS；末端最大停留误差、夹爪读数和方块静置结果与 M1 基线一致。本次只确认运行流程与物理状态，尚未核对实际窗口画面；后续用户反馈原生 UI 绘制异常，复现与修复记录见下。任务完成后先关闭 viewer，再进行离屏截图渲染。用户提前关闭 viewer 时脚本退出并报告未完成，不写入最终结果。
- **关联文件：** `scripts/m1_track_mocap.py`、`README.md`、`results/m1/tracking.csv`、`results/m1/model_info.json`、`results/m1/screenshots/panda_task.png`。

### M1 后续：实时窗口黑块与控件错位修复

- **日期与阶段：** 2026-09-27 / M1 Viewer 修复
- **环境与复现：** Windows；OpenGL 实际使用 `ATI Technologies Inc. / AMD Radeon(TM) Graphics`，版本 `4.6.0 Compatibility Profile Context 22.20.44.37.230215`。同一模型、数据和相机，打开原生左右面板时，真实窗口出现与用户截图一致的巨大黑块及拉伸控件；隐藏两侧面板后，真实窗口正常显示 Panda、桌面、方块和两处目标区。离屏场景渲染也正常。该现象与 [MuJoCo 官方 AMD UI 问题 #639](https://github.com/google-deepmind/mujoco/issues/639) 一致。
- **修复：** smoke 默认传入 `show_left_ui=False, show_right_ui=False`，显示纯场景；保留鼠标相机交互与同一仿真实例的实时同步。增加 `--show-ui` 供图形驱动兼容时调试，帮助信息注明 AMD Windows 原生面板的已知问题。
- **实际窗口验证：** 对完整 smoke 运行中的末端跟踪和夹爪闭合阶段分别抓取窗口客户区，并检查画面；从 1200×900 调整到 1600×1000 窗口后仍正常显示，无黑块或错位控件。保存的 `live_tracking.png` 和 `live_gripper.png` 来自运行中的可见窗口。M1 各物理验收门槛均 PASS，无 warning；仿真 8.7 s、435 个样本，位置误差等读数与基线一致。
- **使用约定：** 本机保持默认纯场景模式；`--show-ui` 或 Tab / Shift+Tab 重新启用原生面板可能再次触发驱动问题。说明已同步到 README。
- **关联文件：** `scripts/m1_track_mocap.py`、`README.md`、`results/m1/screenshots/live_tracking.png`、`results/m1/screenshots/live_gripper.png`。

### M2：确定性 Pick/Place

- **日期与阶段：** 2026-09-27 / M2 完成
- **环境与模型：** Python 3.13.9、MuJoCo 3.14.0、NumPy 2.5.3；使用 M1 的 `assets/scene/panda_task.xml` 与 `home_scene`，没有修改 Panda XML 或场景物理参数。
- **目标与操作：** 实现 `scripts/m2_pick_place.py`，支持单次目标 A/B 和冻结清单批量运行。每 episode 重置机器人与方块完整状态，先预接近、下降、闭爪、验证双侧接触、抬升，再搬运、下降、释放、撤离并独立检查后置条件。方块始终为 freejoint，仅受真实 MuJoCo 接触和重力影响。
- **实际观察与阈值：** 夹爪闭合后左右 finger 与方块均有接触；抬升阶段实测方块跟随末端。冻结目标余量 `10 mm`、桌面高度容差 `3 mm`、线/角速度阈值 `0.01 m/s` / `0.1 rad/s`、稳定时间 `0.5 s`、一般穿透限制 `5 mm`，以及单 pair 接触例外 `6.5 mm`。十条正式轨迹均无 warning、NaN 或预算耗尽；每条至少连续满足 250 个稳定步。
- **验收门槛及结果：** 10 个固定 `scenario_id + seed + target`（A/B 各 5 条）全部成功；第二次完整复跑同样 `10/10`，状态、步数与最终方块位置逐条完全一致。最大末端误差 `3.0243 mm`；全局最深接触为 Panda `link4` 与桌面 `5.9861 mm`，使用 D-002 记录的冻结例外。
- **开发失败与修复：** 快速轨迹初版暴露出长距离搬运的接触深度和放置触桌时过早判滑落；收慢至 `0.08 mm/step`，加入 50 步单侧接触滞回，并将低位放置判定为“接触丢失后必须已获桌面支撑且落在目标内”后，10 个开发场景全部通过。早期失败轨迹保存在 `results/m2/dev_sweep_10/`；最终阈值标定 sweep 保存在 `results/m2/dev_sweep_6mm_a/` 和 `results/m2/dev_sweep_6mm_b/`。
- **证据文件：** `results/m2/acceptance/episodes.jsonl`、`trajectory.csv`、`summary.json`；同配置复跑记录在 `results/m2/acceptance_repeat/`。每 25 个物理步（50 ms）写一条轨迹样本，逐物理步执行状态、接触、警告、穿透和末端误差检查。
- **能力边界与下一步：** M2 证明固定场景内的确定性单方块模拟搬运，不证明视觉、Agent、自动恢复、真实机器人或 Sim2Real。可进入 M3，但应在模型/碰撞验证中继续关注 `link4` 与桌面的已知 5.986 mm 接触。

### D-003：加入危险区并在 M2 逐步执行约束检查

- **日期 / 阶段：** 2026-09-28 / M1 场景与 M2 安全约束补充
- **背景：** 原场景只有 A/B 两个目标标记，没有禁止区；M3 Agent 没有可遵守的空间禁入边界。先在共享 MuJoCo 场景加入危险区，并让现有确定性 M2 作为硬闸门验证每一步的实际几何状态。
- **配置与实现：** `assets/scene/panda_task.xml` 增加纯视觉红色 `danger_zone` box，中心 `(0.82, 0.28, 0.535) m`、半尺寸 `(0.06, 0.06, 0.135) m`，collision mask 为 0，不更改物理接触。`configs/m2_thresholds.json` 固定 20 mm 边距；M2 每个仿真步用 `mj_geomDistance` 检查所有启用碰撞的 Panda 几何和方块几何。距离小于边距时记录 `DANGER_ZONE_VIOLATION`、部件标签、阶段、步数、距离与最近点；episode 后置条件和 trajectory/summary 均保留危险区状态。
- **实测结果：** M1 headless smoke 的模型加载、末端跟踪、夹爪和场景静置均通过，截图可见红色危险柱。新版固定 M2 清单正式运行及完整复跑均为 `10/10`；两轮状态、步数、最终方块位置及最近危险区间距逐项一致，全轨迹最近间距 `113.255 mm`。将方块置入危险体积的故障注入返回 `DANGER_ZONE_VIOLATION`，最终后置条件 `danger_zone_clear=false`。
- **修正记录：** 插入场景 geom 后，MuJoCo 对 Panda link4 碰撞几何的回退标签从 `geom28` 改为 `geom29`。首次新版批量运行 2/10；检查日志确认其余失败均是 link4/table 例外仍引用旧标签，并非危险区违规。为避免依赖不稳定的 geom 编号，最终阈值键改为稳定刚体对 `link4 <-> table`；最终配置的新版清单和复跑均全过。保留首次失败结果以便追溯。
- **限制 / 下一步：** 这是 MuJoCo 几何与配置边距上的软件约束，不证明真实机器人的安全距离。M3 应把 `danger_zone` 空间范围纳入公开场景观测，并在计划执行前做路径/动作闸门；M2 的逐步监测负责阻断确定性执行偏离。
- **证据：** `configs/m2_scenarios.json`、`configs/m2_thresholds.json`、`scripts/m2_pick_place.py`、`scripts/m2_danger_zone_fault_injection.py`、`results/m1/danger_zone_smoke/`、`results/m2/acceptance_danger_zone_body_pair/`、`results/m2/acceptance_danger_zone_body_pair_repeat/`、`results/m2/dev_danger_zone_exception/`、`results/m2/danger_zone_fault_injection/`。
- **回归检查（2026-09-28）：** M1 smoke 增加 `danger_zone` 的固定世界坐标、尺寸、红色半透明外观、纯视觉掩码、桌面贴合和桌面范围检查；headless smoke 全门槛通过。新故障注入脚本把方块放到危险区中心并运行一个正常 M2 物理步，确认 `cube:cube_geom` 触发 `DANGER_ZONE_VIOLATION`，且记录 `danger_zone_clear=false`。M1 运行截图可见红色危险柱。\n
### D-004：危险区故障注入改为真实机械臂运动
- **背景：** 原可视故障示例把方块直接写入危险区中心并推进一步。Viewer 能显示违规方块，但 Panda 保持在 home 姿态，未展示机器人动作，也没有验证手臂碰撞几何触发守卫。
- **架构修正：** M2 抽取 `Episode.prepare()` 统一重置方块并将 mocap 目标对齐真实 `ee_site`，增加 `Episode.move_ee()` 作为正常任务和诊断场景共用的受守卫保护控制入口。M2 episode 结束后更新 Viewer 结果状态并保持窗口打开，直到用户关闭。
- **示例修正：** `m2_danger_zone_fault_injection.py` 现在从 home 姿态朝共享危险区中心运动，不移动方块；M2 每个物理步仍检查 Panda 全部碰撞几何和原有 20 mm 安全边距。报告校验实际末端位移超过 100 mm、违规部件不是方块、错误码为 `DANGER_ZONE_VIOLATION`。
- **可复现轨迹证据：** 在当前配置下，真实 mocap 控制约 2995 个物理步时由 `link7:geom62` 触发，最小间距约 19.963 mm；末端已移动约 0.24 m。守卫按 20 mm 安全边距拦停，因此可视结果是机械臂进入安全边距并停在危险体积外缘，而不是发生物理接触。场景危险区和冻结阈值均未修改。
- **规则：** `AGENTS.md` 明确要求违规示例走正常机器人控制与安全守卫链路，并展示部件、阈值和结果；Viewer 与结构化证据均保持可检查。

### M3：受约束 Agent 闭环

- **日期 / 阶段：** 2026-09-30 / M3 完成
- **实现：** 从共享 M2 episode 提取 `initialize_task()`、`pick_skill()`、`place_skill()`、`verify_goal()` 和 `build_result()`；M2 原入口仍调用相同技能。新增 `src/embodied_agent/contracts.py`、`planner.py`、`runtime.py` 与 `scripts/m3_agent.py`，提供目标别名解析、真值观测、严格计划 schema、stub/DeepSeek 规划器、运动段危险区预检、技能前置检查、状态机、调用/仿真步预算和追加式运行日志。M2 每物理步检查继续启用。
- **配置：** `configs/m3_runtime.json` 冻结单次 LLM 请求、30 s 超时、256 输出 token、30000 episode 步、2 次技能调用、pick 10000 步、place 12000 步、0 次重试/重规划、20 mm 危险区边距及 50 mm 末端/负载预检包络。M3 指令别名与 `configs/m3_cases.json` 中 20 个验收案例冻结。
- **M2 重构回归：** `results/m3/m2_refactor_run1/`、`m2_refactor_run2/` 均 10/10。逐个场景比较原 `results/m2/acceptance_danger_zone_body_pair/` 的状态、步数、危险区最小距离和最终方块位置，全部精确一致。
- **M3 stub 集成：** `results/m3/stub_batch/` 为 20/20，绿色/A 与蓝色/B 各 10 个指令，覆盖中文编号别名与英文颜色别名；所有成功均完成一次 `pick`、一次 `place` 并满足同一独立后置条件。
- **真实模型评测：** 提示词 v2 的 `results/m3/llm_batch_v2/` 共 20/20 成功，20 次独立 DeepSeek `deepseek-flash` 请求，A/B 各 10 条；总用量 14536 tokens（每条 725–729），单次执行 14753–18331 个仿真步，逐条目标与人工标注相符。可视单次结果另见 `llm_visible_a_v2/` 和 `llm_visible_b/`，A/B 均成功。
- **首次契约失败及修正：** 原 `results/m3/llm_visible_a/` 的第一次真实输出带有额外顶层 `type` 字段；`INVALID_PLAN` 在技能调用前阻断，仿真仅做 1000 步场景静置。当时保存了完整错误、响应、token 和轨迹记录，这些旧产物现已在 v2 整理中清理；失败结论保留。严格 schema 未放宽，提示词升级为 `m3-pick-place-json-v2`，给出精确对象样例并禁止格式元数据。之后 A/B 可视任务和冻结 20 条全通过。
- **负例与回归测试：** `tests/test_m3_runtime.py` 共 9 项通过：20 个目标解析、歧义/不支持指令、额外/乱序/重复键/过期/目标不符计划、穿越危险区的路径拒绝、非法计划零技能执行、LLM 超时/鉴权/限流/空响应/截断响应的有界终止、未持物时阻止放置、技能步数预算强制停止。命令为 `.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v`。
- **Viewer 观察限制：** 两条单次真实模型运行均未传 `--headless`，Windows MuJoCo 主窗口进程确实启动；代码使用正在步进的同一 `Episode.model/data`，完成后保持窗口并在关闭请求后正常退出。但当时桌面处于 Windows 锁屏，系统截图只捕获锁屏背景，未独立核对 MuJoCo 窗口像素。记录的是 Viewer 进程/生命周期通过、画面内容人工观察未确认；解锁后可用 README 命令再次观看。
- **能力限制：** 当前目标解析只覆盖冻结的中文 A/B 与英文 green/blue 短指令；Agent 读取仿真真值，无视觉。无恢复、扰动下检测精度、真实硬件或 Sim2Real 结论；继续由 M4/M5 阶段处理。

## 通用 Demo、地图与后续整理记录

### D-005：地图持久化与通用 Demo 会话

- **日期 / 阶段：** 2026-10-01 / M3 后续测试台；已实现。
- **决策：** 使用独立 JSON 地图保存初始环境，复用已有 Panda 模型、M3 技能和逐步安全守卫；统一自由指令、环境编辑和批量用例入口，不为演示改变物理成功判据。地图不是静态截图，加载后直接应用到实际执行的 `MjModel` / `MjData`。
- **地图基线：** `schema_version=1`，本次核对两图均为 `revision=1`；Z 轴向上、长度用米，位置指物体中心，`half_size_m` 为半尺寸。桌面顶面 `z=0.4`，边长 `0.05` 的方块初始中心 `z=0.425`。

| 地图 | 方块初始中心（m） | 危险区中心（m） | 危险区半尺寸（m） |
| --- | --- | --- | --- |
| [classic：经典桌面](configs/maps/classic.json) | `(0.40, -0.29, 0.425)` | `(0.82, 0.28, 0.535)` | `(0.06, 0.06, 0.135)` |
| [alternate：变体桌面](configs/maps/alternate.json) | `(0.42, -0.25, 0.425)` | `(0.83, 0.28, 0.535)` | `(0.055, 0.06, 0.135)` |

- **共同目标区：** A（绿色）中心 `(0.58, -0.12, 0.401)`，B（蓝色）中心 `(0.58, 0.12, 0.401)`；半尺寸均为 `(0.075, 0.075, 0.001)`。变体改变方块起点、危险区位置与尺寸，保留 A/B 以复用冻结技能和用例。
- **存储约束：** 严格 schema、有限数值与桌面支撑/边界检查；整次编辑统一校验，通过后在写锁内检查预期版本、原子保存并增加 revision。非法多操作或写入失败不部分提交；过期版本返回 `REVISION_CONFLICT`。源地图合法不保证机器人路径可达或安全，动作仍由原守卫独立判断。
- **会话约定：** 自由模式连续指令保留当前物理状态；选择地图或重置才恢复最新保存的初始环境。重置不撤销已保存编辑。批量复制地图到输出目录，逐例恢复批次起始快照；同一用例的多步操作共享状态。
- **当前实现：** `src/embodied_agent/maps/schema.py`、`maps/store.py`、`maps/scene.py`、`apps/demo/session.py`。设计与首次验收见[地图与 Demo 执行分析](../task-records/20261001-185104-world-maps-demo/analysis.md)。

### Demo：统一可视自由测试与批量评测

- **日期 / 阶段：** 2026-10-01 完成，2026-10-02 补充 README；2026-10-04 整理后回归通过。
- **界面与物理执行：** `scripts/demo.py` 默认进入 Tk 单窗口自由模式；先选择并加载地图，再显示真实 MuJoCo Renderer 场景。提供世界 X/Y/Z 轴、米制刻度与桌面网格、旋转/缩放/恢复视角、机器人和环境双输入区、重置，以及右上角用例选择/执行。物理步进与绘制在主线程串行执行，语言请求后台等待；执行期间禁用冲突操作，终态保持可见。
- **当前入口与接口：** CLI 在 `src/embodied_agent/apps/demo/cli.py`；`DemoSession` 提供地图选择、重置、Agent 执行、环境编辑、用例执行和结果收尾。`session.register_agent()` / `app.register_agent()` 可接入后续适配器；当前内置 M3 与环境 Agent。
- **用例与输出：** 当时的 [configs/demo_cases.json](configs/demo_cases.json) 包含原 M3 的 20 条用例及 5 条变体地图/环境用例，共 25 条。支持单步、多步和独立 `expected` 校验；旧结果格式包含 `manifest.json`、`actions.jsonl`、`cases.jsonl`、`summary.json` 及机器人 `episodes.jsonl`、`events.jsonl`、`trajectory.csv`。这些旧产物现已在 v2 整理中清理；当前 `results/demo/<时间戳>/` 只写 manifest、摘要与任务引用，批量地图副本仍保存，详细事实统一进入项目 `records/`。新报告目录仍拒绝覆盖已有非空目录。
- **中断与统计：** 窗口执行中关闭产生 `ABORTED/VIEWER_CLOSED`，余例记 `NOT_RUN`。`passed_count` 表示符合预期的用例数，`success_count` 表示实际结果成功数；预期安全拒绝只计入前者。全部符合预期退出 0，失败、中断或配置错误退出非零。

在 `robot_agent` 目录首选可视自由模式；默认机器人和环境规划均读取项目 `.env` 调用 DeepSeek：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode free
```

离线可视开发与可视批量：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode free --planner stub --environment-planner rules
.\.venv\Scripts\python.exe .\scripts\demo.py --mode batch --viewer --planner stub --environment-planner rules
```

专门的自动化批量评测可使用无界面模式；批量默认无窗口，自由模式拒绝 `--headless`：

```powershell
.\.venv\Scripts\python.exe .\scripts\demo.py --mode batch --headless --planner stub --environment-planner rules
```

- **验收证据：** 2026-10-01 默认批量为 25/25 符合预期，实际成功 25；当时的 `language_fix_default_batch_20261001` 汇总已在 v2 整理中清理。2026-10-04 移除旧兼容入口后的完整批量仍为 25/25，实际成功 25，见[当次 Demo 汇总](../task-records/20261004-190613-remove-src-compat/validation/demo-batch/summary.json)。图形测试验证选图、编辑、重置、真实 M3 动作、窗口关闭中断；当次 2/2 通过，见[GUI 回归日志](../task-records/20261004-190613-remove-src-compat/validation/gui-tests.stderr.txt)。
- **问题与解决：** Windows 中文路径的 XML 加载改为受锁保护的相对路径加载；批量前例污染后例改为私有地图与逐例基线恢复；显示与存储不一致增加版本/快照检查；保存后刷新失败仍保留 `persisted`、before/after 和 `refresh_error`；关闭窗口统一保留中断和未执行证据。对应地图、事务、会话和 GUI 回归已通过。
- **边界与下一步：** stub/rules 仅用于有限离线解析，机器人动作仍运行真实 MuJoCo 物理。测试台扩展接口已完成，后续 Agent 必须实现自己的契约、安全检查和成功判据；不据此宣称 M4/M5 或家居场景已完成。使用说明见 [DEMO.md](DEMO.md) 和 [README.md](README.md)。

### Demo 后续：模型优先的机器人语义与环境相对编辑

- **日期 / 阶段：** 2026-10-01 / 用户报告问题修复完成。
- **机器人链路：** Demo 的 llm 模式先让 DeepSeek 解释原始指令，输出受限对象/动作/目标，再由 M3 规划器生成 pick/place 技能计划；不再在模型请求前用固定 A/B 动词别名过滤。语义响应、请求次数、模型与用量进入执行证据。原 `scripts/m3_agent.py` 冻结解析/评测语义保留。
- **安全语义：** “把方块放到危险区”经语义解释后由独立契约返回 `FAILED/UNSAFE_TARGET`，不启动抓放；不能重定向为 A/B 来声称成功。物理动作仍限于单方块到 A/B。
- **环境编辑：** `agents/environment.py` 将自然语言转为受限 `set_position`、`set_half_size`、`scale`、`shift_axis` 操作；只修改方块初始位置和已知区域位置/尺寸。支持米/厘米/毫米、完整尺寸转半尺寸、明确比例缩放、x/y/z 相对移动；“调高一点/调低一点”固定为对应坐标 ±0.01 米，仍需通过完整地图校验。相对加法使用十进制处理，避免保存冗长的浮点尾数。
- **用例后编辑修正：** 临时起点覆盖保留原存储地图快照，随后编辑以存储地图为基准；成功保存并重置显示时记录 `auto_reset_from_case: true`。外部真实版本更改仍拒绝覆盖。
- **针对性验收：** [demo_language_cases.json](configs/demo_language_cases.json) 共 4 条。当时真实模型批量为 4/4 符合预期，其中实际成功 3（1 条机器人动作、2 条环境编辑），安全拒绝 1；原汇总已在 v2 整理中清理。“请将小方块搬运至天蓝色目的地”实际放到 B 区，2 次模型请求、18331 物理步；危险区指令 1 次语义请求、0 次技能规划、0 物理步。两条相对编辑分别保存 y=-0.28、x≈0.41；早期记录曾含修正前的浮点尾数，历史结果说明未改写。
- **真实可视证据：** 当时的 `gui_smoke.json` 与编辑后截图记录 B 区成功、危险区零步安全拒绝、用例后 y 相对编辑及真实渲染；旧结果与截图已在 v2 整理中清理。详细问题、修复和当时结论见[执行分析](../task-records/20261001-211151-demo-agent-language-fix/analysis.md)。
- **复现：** 自由模式输入上述指令；针对性批量使用 `--mode batch --planner llm --environment-planner llm --cases .\configs\demo_language_cases.json`，观察时加 `--viewer`。非固定话术用例需要真实模型，stub 不能验收同等语义能力。

### Demo 后续：加载地图时的窗口缩放处理

- **日期 / 阶段：** 2026-10-02 / 修复代码与静态检查完成。
- **现象及处理：** 用户反馈加载地图后页面与文字一起缩小；最初恢复 Tk geometry/scaling 未解决。后续依据 GLFW 在 Windows 首次初始化时改变进程 DPI 感知的机制，在创建 `tk.Tk()` 前短暂创建、激活并释放 MuJoCo OpenGL context，使 GLFW 初始化先于 Tk；保留加载后的窗口状态/尺寸/缩放恢复。批量无界面路径不创建该上下文。
- **验证边界：** 当时只完成语法与行尾检查，没有目标桌面视觉确认。2026-10-04 已有整体 Demo 图形回归 2/2 通过，但未单独记录该缩放现象的专项复测或用户确认，因此不把专项视觉问题标为已验收。当前实现位于 `visualization/demo_ui.py`，原调查见[窗口调整分析](../task-records/20261002-145739-demo-window-size/analysis.md)。

### A0：已实现功能的模块整理与旧 src 入口移除

- **日期 / 阶段：** 2026-10-04 / A0 完成，随后按用户要求删除旧 src 兼容文件。
- **当前结构：** 应用位于 `apps/demo/`；地图、Agent、模型服务、执行、仿真、技能、安全、可视化、评估分别位于 `maps/`、`agents/`、`models/`、`execution/`、`simulation/`、`skills/`、`safety/`、`visualization/`、`evaluation/`。脚本保留启动命令并调用实际模块，src 顶层仅保留 `__init__.py`、`contracts.py`、`paths.py`；八个旧重导出文件已删除，源码和测试直接导入新实现。职责及公开接口见 [MODULES.md](MODULES.md)。
- **基线与保护：** 当时整理前后源码、Git 状态和 hash 清单保存到任务目录；当次保留了资产、配置、地图、历史验收和唯一失败证据，仅对清单确认的空会话和重复图像校验归档后清理。旧 Agent 运行产物后来按用户授权在记录 v2 整理中删除，范围见顶部清理说明。manifest/summary 当时增加迁移后 `source_hashes`；相关变更当时在工作区，未提交或推送。
- **最新验证：** 全套 66 项测试中 64 通过、2 项图形默认跳过；另启用真实桌面运行 2/2 通过，未把跳过计为通过。移除兼容文件后 50 个模块独立新进程导入通过、Demo 25/25、M3 20/20；见[最终汇总](../task-records/20261004-190613-remove-src-compat/validation/final-report.json)及[单元回归日志](../task-records/20261004-190613-remove-src-compat/validation/unit-tests.stderr.txt)。A0 还验证 M2 10/10、M1 诊断、原生 Viewer 正常控制链路触发危险区守卫，见 [A0 执行分析](../task-records/20261004-000502-robot-agent-a0/analysis.md)。
- **能力边界与下一步：** A0 和入口移除回归均未重新调用真实模型 API；模型验证采用模拟 SDK/冻结行为对比，2026-10-01 真实模型证据当时单独保存，旧运行产物现已在 v2 整理中清理。当时后续计划按 [CAREER_ROADMAP_2026.md](CAREER_ROADMAP_2026.md) 推进 A1，本地模型部署、家居环境、完整 Hook/隔离预演、训练及对抗 Agent 属于该阶段后续计划。旧入口移除清单和历史结论见[执行分析](../task-records/20261004-190613-remove-src-compat/analysis.md)。

### 任务级 JSON：统一请求记录

- **日期 / 范围：** 2026-10-05，按用户要求完成任务级 JSON；统一 Hook、滚动预测和自动重规划继续搁置。
- **新记录：** Demo、家居机器人和共享 M3 按每请求保存地图定义与哈希、真实前后观测、原始语言、规划输入/响应/指令集、实际技能及物理证据切片、真实反馈和来源/成本。采用 UUID 和上海日期＋地图简写＋记录根内全局日序号；关键边界原子落盘，完成后不随下一任务改写。
- **输出与查询：** 当时默认 `records/tasks/`，显式 `--output` 使用其 `task_records/`；这是已退出使用的 v1 规则，旧主记录已在 v2 整理中清理。当前默认主根固定为项目 `records/`，`--records-dir` 可覆盖，`--output` 只决定 manifest、摘要和任务引用的报告位置，批量地图副本仍保留。无仿真/模型依赖的 `scripts/task_records.py` 查询已迁移到 v2，用法与契约见 [TASK_RECORDS.md](TASK_RECORDS.md)。
- **验证与边界：** 核心并发/原子故障、执行事件、Demo/M3、模型边界、查询及真实桌面回归通过；保留 5 个真实执行任务样本，包含 Panda 14994 步成功放置与危险目标零步拒绝。观测不是完整可恢复 checkpoint；未接入 Hook 的 checks 和未实际送达 Agent 的反馈保持空。具体结果见[执行分析](../task-records/20261005-214709-task-json-records/analysis.md)和[样本验证](../task-records/20261005-214709-task-json-records/validation-results.json)。


## 2026-10-07：清理旧指令Agent（已完成）

删除 `agents/home_language.py`、`language.py`、`planning.py`、旧 `execution/runtime.py`，移除旧TaskGoal/固定两步计划及提示词、Session/UI兼容成员和m3注册。原生工具对话成为唯一LLM协议，旧启动脚本替换为 `scripts/instruction_agent.py`，运行配置整理为 `configs/agent_runtime.json`。二十个场景数据改走当前通用Agent。本文此前M3配置与入口记录为历史，不代表现状。相关有效物理/模型/记录测试迁移至当前接口。
