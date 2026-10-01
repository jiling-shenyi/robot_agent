# 执行计划：危险区示例可视运行与项目规则

## 目标与范围

让 `scripts/m2_danger_zone_fault_injection.py` 默认打开 MuJoCo Viewer，使运行者能看到方块位于 M1 红色危险区内，并看到 M2 的 `DANGER_ZONE_VIOLATION` 结果。增加明确的 `--headless` 选项供自动化或无显示环境使用。为 `robot_agent` 建立仓库级 Codex 项目指令，并加入后续示例执行必须提供可见模式的规则。

## 已知情况与假设

- 危险区故障注入脚本已能在一个物理步内触发 M2 违规分支，但没有启动 viewer，也没有等待窗口查看。
- M1/M2 共用 `assets/scene/panda_task.xml`；M1 危险区标记为纯视觉几何，M2 通过软件距离检查实施约束。
- 按 Codex 官方项目指令机制，仓库根目录 `AGENTS.md` 会作为项目级指令被发现；新增后当前会话将手动遵循，后续会话会在启动时加载。
- “可见”指示例默认启动真实的 MuJoCo 可视窗口；无界面验证显式使用 `--headless`。

## 执行步骤

1. 检查当前故障注入脚本与 M2 Viewer API，确认启动、同步、关闭窗口的生命周期。
2. 在本目录实现默认可视的故障注入：让 Viewer 对准危险区，显示注入状态，在记录违规后继续同步窗口直到用户关闭；增加 `--headless` 快速路径。
3. 创建仓库根目录 `AGENTS.md`，明确所有项目运行示例必须提供可见执行能力；README 示例默认展示可见执行命令，并记录显式 headless 用法。
4. 分别验证 headless 违规路径和可视模式窗口启动/渲染；确认退出码、违规记录和可见场景内容。
5. 创建 `analysis.md`，记录实际改动、测试证据、问题和解决情况。

## 预期产物

- 更新 `scripts/m2_danger_zone_fault_injection.py`。
- 新增 `AGENTS.md` 项目规则。
- 更新 `README.md` 的可视/无界面运行说明。
- 运行证据保存在新的 `results/m2/danger_zone_fault_injection_visible/` 目录。
- 本目录中的 `analysis.md`。

## 完成判据

- 不带 `--headless` 运行示例会启动并同步一个展示危险区与违规方块的 MuJoCo Viewer，违规信息可在终端看到，窗口保持打开直到手动关闭。
- 使用 `--headless` 可在无界面环境中完成相同违规判定，并写出可审阅的证据。
- 仓库根 `AGENTS.md` 明确要求后续示例执行提供可视功能，并让 README 的示例命令与规则一致。
- `analysis.md` 完整记录完成结果和验证命令。

## 主要风险与处理

- 被动 Viewer 可能因立即退出而来不及显示违规状态：触发后持续同步 Viewer，直至用户关闭窗口。
- 无图形显示环境无法运行被动 Viewer：提供显式 `--headless` 路径；在当前环境检查可视模式可用性，若受显示环境限制则如实记录并用可视化渲染帧和 headless 违规验证补足。
- 诊断运行目录可能已有证据：选用独立输出目录，且默认禁止覆盖。
