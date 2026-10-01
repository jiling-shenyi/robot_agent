# 执行分析：M2 危险区机械臂示例与架构修复

## 实际完成

- 在 `scripts/m2_pick_place.py` 中抽取 `Episode.prepare()`，统一 keyframe 重置、末端姿态对齐和共享 M2 场景初始化；增加 `Episode.move_ee()` 作为正常 pick/place 与诊断示例共用的末端运动入口，并要求动作经逐物理步危险区守卫检查。
- M2 正常 episode 使用上述共用入口。动作违规时 Viewer 显示错误码与触发几何；episode 结束后显示最终状态并保持窗口打开，直至用户关闭。
- 将 `scripts/m2_danger_zone_fault_injection.py` 改为从 home 姿态驱动 Panda 朝危险区运动。方块保持在桌面安全位置；报告额外检查机械臂确实发生位移且触发几何不是方块。
- 在 `AGENTS.md` 增加项目规则：机器人违规示例必须通过正常控制链路触发逐步守卫，并在可视窗口展示部件、阈值和动作结果。
- 同步更新 `README.md`、`PROJECT_PLAN.md`、M2 实施计划和 `NOTES.md`，描述真实机械臂动作及安全边距语义。

## 完成判据及证据

- M1 危险区配置检查通过，几何坐标、尺寸、颜色、桌面关系和纯视觉设置均符合当前配置。
- M2 使用真实 Panda 碰撞几何触发预期错误：`DANGER_ZONE_VIOLATION`；故障部件为 `link7:geom62`，物理步数为 2995，最近间距为 19.963 mm，门槛为 20.0 mm。
- 末端从 home 姿态移动 236.1 mm 后被守卫拦停；报告中的 `robot_geom_triggered=true`、`danger_zone_clear=false`、`passed=true`。
- 无界面运行证据：`results/m2/danger_zone_robot_motion_headless/summary.json` 和 `episodes.jsonl`。
- 默认可视运行也完成同一运动并写入 `results/m2/danger_zone_robot_motion_visible/summary.json` 和 `episodes.jsonl`；入口进入等待关闭 Viewer 的阶段，验证窗口不会在结果出现时被核心 M2 逻辑立即关闭。
- `git diff --check` 未发现空白错误；只输出仓库既有的 LF/CRLF 转换提醒。
- 危险区坐标和冻结的 20 mm 阈值未改变。守卫按 20 mm 最小间距提前拦停，因此实际可见状态为进入安全边距后停在危险体积外，不发生物理接触。

## 问题及处理

- 原示例通过直接修改 cube 的 freejoint 位置制造违规，机械臂没有运动。改为通过 `Episode.prepare()` 和 `Episode.move_ee()` 执行动作，由 M2 原有的逐物理步危险区检查触发失败。
- M2 原入口在 `run()` 的清理阶段立即关闭 Viewer。将关闭职责移到显式的 Viewer 生命周期方法，由 CLI 在结果产生后等待用户关闭窗口。
- PowerShell 管道的代码页在首次写入中文规则/文档时把中文替换为问号。改用 ASCII Unicode 转义写入 UTF-8，并回读校验编码内容；规则与记录文件已恢复正确文本。
- 可视运行的标准输出由非交互管道缓冲；确认可视运行已生成 PASS 结果并处于 Viewer 等待阶段后，为结束本次交互进程从终端发出中断。进程的退出码 1 来自这次终端中断，结果文件仍记录 `passed=true`。

## 结束状态

所有本次目标已完成。
