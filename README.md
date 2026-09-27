# Robot Agent

MuJoCo 机器人操作 Agent 项目。整体阶段安排见 [PROJECT_PLAN.md](PROJECT_PLAN.md)，M0/M1 执行规划见 [M0_M1_实施规划.md](M0_M1_实施规划.md)。

## MuJoCo / M1

在仓库根目录使用项目虚拟环境运行 M1 smoke：

```powershell
.\.venv\Scripts\python.exe .\scripts\m1_track_mocap.py
```

脚本加载 `assets/scene/panda_task.xml`，从 `home_scene` 初始化，先对齐 mocap 目标和真实 `ee_site`，再做三个位姿的短距离跟踪、返回 home、空载夹爪三次开合和方块静置检查。结果保存在 `results/m1/`：`tracking.csv`、`model_info.json` 和 `screenshots/panda_task.png`。

也可以在图形窗口中打开项目场景：

```powershell
.\.venv\Scripts\python.exe -m mujoco.viewer --mjcf .\assets\scene\panda_task.xml
```

Panda 上游文件及 Apache-2.0 许可证位于 `assets/third_party/franka_emika_panda/`；Menagerie 提交、模型 OID 和项目派生 XML 的修改记录见 [`assets/third_party/README.md`](assets/third_party/README.md)。M1 仅验证仿真末端跟踪、空载夹爪和场景静置，不代表完成抓取、搬运或放置。

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
- 完整抓取/放置、Agent 规划器、视觉和真实机器人尚未验证；API 连通性 smoke 不代表这些功能已经集成。
