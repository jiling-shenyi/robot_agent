# 第三方模型来源

## Hello Robot Stretch 2

- 来源：[锁定MuJoCo Menagerie模型](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/hello_robot_stretch)。包版本 `2026.9.2`，条目 OID `d37c1fc83fc259f0681856608f3c59df66ce7c1d`。
- 原始MJCF/mesh/纹理保持原样，项目副本位于 `hello_robot_stretch/`；许可证为该目录内的 `LICENSE`（BSD-3-Clause-Clear）。
- 版本和哈希见 [MODEL_PROVENANCE.json](hello_robot_stretch/MODEL_PROVENANCE.json)，运行时坐标/控制/碰撞适配见 [PROJECT_ADAPTATION.md](hello_robot_stretch/PROJECT_ADAPTATION.md)。

## Franka Emika Panda

- **来源：** [MuJoCo Menagerie `franka_emika_panda`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda)
- **Python 包：** `mujoco-menagerie==2026.9.2`
- **Menagerie 仓库提交：** `c96a32d28fb5da84da38c1da4d749e7a13212855`
- **模型条目 OID：** `3d2262eeb81ecec19abfa31dd509e35abbb33e67`
- **许可证：** Apache-2.0，原始许可证保存在本目录的 `franka_emika_panda/LICENSE`。
- **上游入口：** `franka_emika_panda/panda.xml`；模型 mesh 位于同目录的 `assets/`。
- **项目副本：** `assets/third_party/franka_emika_panda/`，包含未经修改的 `panda.xml`、上游 `README.md`、`CHANGELOG.md`、`LICENSE` 和该 XML 引用的 67 个 mesh 文件（合计 34,292,713 字节）。
- **上游 XML SHA-256：** `96ad67da03710f17f798c9478fd9e9efdf24a3bf8359f05e456dd9fb158ea273`。
- **许可证文件 SHA-256：** `a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9`。

### 项目派生模型

上游 `panda.xml` 保持原样。项目场景通过 `assets/third_party/franka_emika_panda/panda_mocap.xml` 使用以下 M1 修改：

1. 将 mesh 搜索路径调整为供 `assets/scene/panda_task.xml` include 使用的相对路径。
2. 固定仿真步长为 0.002 s；将机械臂 actuator 1–7 分配到禁用组 0，将夹爪 actuator 8 分配到启用组 1，并禁用组 0。
3. 在 `hand` body 上增加 `ee_site`，局部位置为 `(0, 0, 0.1034) m`、局部姿态为单位四元数；此点定义为本项目的 TCP。
4. 增加 `mocap_weld`，将 `ee_site` 与项目场景中 `mocap_target` body 的 `mocap_site` 约束在一起。

M1 使用项目场景的 `home_scene` keyframe 初始化：其中 Panda 的关节值沿用上游 `home`，并另外明确设置自由方块位姿，避免新增方块重置到世界原点。随后把 mocap site 对齐到实测 `ee_site` 位姿。
