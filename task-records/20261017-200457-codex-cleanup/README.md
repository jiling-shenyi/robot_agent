# Codex 卸载后残留清理

把 `Cleanup-Codex.cmd` 和 `Cleanup-Codex.ps1` 保持在同一目录。先通过 Windows 设置卸载 Codex，关闭相关应用，再双击 `Cleanup-Codex.cmd`。脚本展示五个目录并检查卸载状态，输入大写 `DELETE` 后才执行永久删除，其他输入取消。通常不需要管理员权限。

清理范围固定为当前用户的 `.codex`、本地应用数据下的 `OpenAI/Codex` 与 `Codex`、漫游应用数据下的 `Codex`，以及本地应用数据下的 `Packages/OpenAI.Codex_2p2nqsd0c76g0`。不修改项目、注册表、环境变量或 WindowsApps。

**先备份：** `.codex` 可能包含本地聊天、配置、技能、插件、生成文件和工作副本，并与命令行版、编辑器扩展共用。删除不进入回收站。本工具只清理残留，不会卸载应用或结束进程，也不负责迁移到 D 盘。

在文件所在目录打开 Windows PowerShell，可执行只读预览：

```powershell
.\Cleanup-Codex.cmd -PreviewOnly
```

不存在的目标自动跳过。遇到符号链接/目录联接，脚本拒绝清理对应目标，避免删除其他位置的数据。程序包查询失败、应用尚未卸载或进程仍运行时，正式清理会停止；文件占用或权限错误会逐项报告。可关闭应用或重启后重试，不要强行修改系统目录权限。

清理结果显示在窗口，并尝试写入同目录的 `Cleanup-Codex-result.txt`。退出码：`0` 为预览/取消/清理成功，`1` 为有目标清理失败，`2` 为前置检查或其他错误导致停止。用户取消时窗口有明确提示，退出码 `0` 不单独代表已完成删除。

启动器仅对本次 PowerShell 进程使用 `ExecutionPolicy Bypass`，不会修改系统执行策略。PowerShell 文件须保留 UTF-8 BOM 编码，以便 Windows PowerShell 5.1 正确显示中文。
