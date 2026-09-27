# 项目决策与实验记录

本文件记录项目范围、可复现基线、关键决策和实验结果。计划、环境查询与仿真结果分开记录；没有运行或观察到的内容不标记为通过。后续实验按“日期与阶段、环境/模型、目标与配置、观察结果、验收门槛、失败信息、下一步、关联文件/提交”追加。

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
