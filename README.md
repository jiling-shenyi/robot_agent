# Robot Agent

MuJoCo 机器人操作 Agent 项目。整体阶段安排见 [PROJECT_PLAN.md](PROJECT_PLAN.md)，M0/M1 执行规划见 [M0_M1_实施规划.md](M0_M1_实施规划.md)。

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
- MuJoCo 场景、mocap 控制和 Agent 规划器仍按阶段方案推进；烟测成功不代表机器人仿真或任务执行已经完成。
