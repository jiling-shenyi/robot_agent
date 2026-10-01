# 执行分析：危险区示例可视运行与项目规则

## 完成情况

- 更新 `scripts/m2_danger_zone_fault_injection.py`：默认创建并同步 MuJoCo Viewer，镜头对准危险区；在画面中显示故障注入和 `DANGER_ZONE_VIOLATION` 状态；判定完成后保持窗口运行，直到用户关闭。增加显式 `--headless` 选项。
- 新增仓库根 `AGENTS.md` 项目规则，要求新建和修改的机器人/场景示例默认可视，故障注入要显示状态并保持窗口可检查；无界面运行需显式指定。
- 更新 README 的示例命令和可视/headless 说明。
- 更新 `PROJECT_PLAN.md` 的 M2 验收条目，加入违规分支故障注入和 M1 危险区配置检查要求。
- 更新 `NOTES.md`，记录 M1 配置回归和 M2 违规故障注入结果。

## 完成判据与证据

1. `python -m py_compile scripts/m2_danger_zone_fault_injection.py` 通过。
2. Headless 故障注入命令退出码为 0；M1 危险区 11 项配置检查全部通过；一个物理步后 M2 记录 `DANGER_ZONE_VIOLATION`、几何 `cube:cube_geom`、距离 `0.000 mm`（低于 `20.0 mm`），`danger_zone_clear=false`。证据位于 `results/m2/danger_zone_fault_injection_headless/`。
3. 默认可视命令（未传 `--headless`）输出 `Live viewer opened`、M1 检查全部通过、M2 故障注入 `PASS`，并提示 Viewer 会保持打开直到用户关闭。证据位于 `results/m2/danger_zone_fault_injection_visible/summary.json` 和 `episodes.jsonl`，摘要 `passed=true`。验证使用交互终端，Viewer 正常启动并等待关闭；之后通过终端中断关闭验证窗口。工具对该中断返回进程码 1，脚本输出了 `Viewer closed from the terminal.`，已写入的故障注入结果仍为通过。
4. M1 smoke 的场景配置 gate 通过，截图显示桌面后方的红色半透明危险柱；证据位于 `results/m1/danger_zone_config_regression/`。
5. `git diff --check` 对本次编辑文件通过（仅有 Git 的行尾转换提示，无空白错误）。

## 遇到的问题与处理

- 首次可视验证使用非交互管道，stdout 缓冲导致 Viewer 已运行时暂时看不到日志。改用 TTY 交互终端复跑，确认 viewer 启动日志和检测结果可见。
- 可视脚本按设计在违规后保持等待，不能像 headless 测试那样立即退出。验证完窗口与终端输出后用终端中断关闭；脚本捕获该中断并关闭 Viewer。正常使用时可直接关闭 MuJoCo 窗口。

未遇到阻碍；任务完成。
