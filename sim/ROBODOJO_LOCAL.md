# RoboDojo 本地环境与复现记录

本次已验证环境最初位于 `OpenETA-robodojo` 工作区（源码 `880522b`）。
实现已合入同仓库 `main`，以下路径均相对当前 clone。安装记录不代表新机器
已通过端到端验证；[接续入口](../docs/robodojo-workspace.md) 说明两种 harness。

## 固定版本

| 组件 | 已验证版本 |
| --- | --- |
| Host 与仿真 Python | 3.11.14，两个独立 venv |
| RoboDojo | `ee67a1468510da7624a089164402359f2afc72c8` |
| Isaac Lab | 2.3.2，`afca7b09d60d8beb9c1cb28b43066499940b969b` |
| cuRobo | `d17b54ce32cba095c0b000c4c58777075d11de0e` |
| XPolicyLab 源码 | `432f82b1758c5b1202e42a3dfe014546dbc50871` |
| Isaac Sim | 5.1.0，含 `all,extscache` |
| PyTorch | 2.7.0，CUDA 12.6 wheel；未使用上游 cu128 安装方式 |
| 资产 revision | `35efbc7dedfdbeeb6e95fb749bd885d73d483e41` |

固定源码仓库和已下载归档的 SHA-256 见
[source-manifest.json](setup/robodojo/source-manifest.json)。
[isaac-constraints.txt](setup/robodojo/isaac-constraints.txt) 是本机安装约束；
[isaac-installed-constraints.txt](setup/robodojo/isaac-installed-constraints.txt)
记录 Isaac 安装阶段的固定依赖；
[installed-packages.txt](setup/robodojo/installed-packages.txt) 是最终包版本记录，
不能直接当成可移植的一键安装文件。Editable 包来自表中的源码，
没有把本机 `pip freeze` 错误推断出的外层仓库地址用作安装来源。

## 环境准备

Host 在 `.venv` 安装本仓库，仿真在 `sim/venvs/robodojo` 单独安装 Isaac、
Isaac Lab、cuRobo 和原生 RoboDojo 依赖。MCP 1.28.1 要求较新的 uvicorn，
与 Isaac 固定的 `uvicorn==0.29.0` 冲突；不要把完整 OpenETA/MCP 依赖装进
Isaac 环境。worker 从 `sim/bench_worker.py` 的源码路径加载项目。

源码布局：

```text
third_party/RoboDojo/
  third_party/IsaacLab/
  third_party/curobo/
  third_party/XPolicyLab/
  Assets/
```

可使用 `scripts/setup_envs.sh --help` 查看仓库现有源码准备入口，再按上述
commit 核对实际 checkout。依赖准备参考固定 RoboDojo 源码的
`scripts/install.sh`，本机没有执行其含 `sudo apt-get` 的系统安装流程。
仅安装到用户环境；不改驱动、系统 CUDA 或 GPU 设置。

本机使用了两处第三方兼容调整：

- Isaac Lab `source/isaaclab/setup.py` 中 `starlette==0.49.1` 改为
  `starlette==0.45.3`，与 RoboDojo 的运行时约束一致。
- 将 `third_party/IsaacLab/apps/isaaclab.python.kit` 复制为
  `isaaclab.local.python.kit`，仅将 URDF 扩展要求从 `2.4.31` 改为
  已安装 wheel 提供的 `2.4.30`；通过 `--experience` 显式使用副本。
  若自己的 Isaac 扩展版本已匹配，应使用对应文件，不照搬本机替换。

cuRobo 若由无 Git 元数据的固定归档安装，使用
`SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.0.0+git.d17b54ce`。
最终仿真环境 `pip check` 已通过。安装后的路径、驱动和硬件仍需在目标机器验证。

## 三任务资产子集

[清单](setup/robodojo/assets-smoke3-manifest.json) 包含 260 个文件、约 685.8 MB，
仅支持 seed 0 / layout 0 的 `pour_balls_into_vase`、`fill_pen_holder` 和
`swap_blocks`。不代表完整的约 41 GB 资产库，也不保证其他 seed 可用。

以下下载需自行确认网络、空间和下载授权：

```bash
sim/venvs/robodojo/bin/python scripts/local_robodojo_assets.py
sim/venvs/robodojo/bin/python scripts/local_robodojo_assets.py --download
(cd third_party/RoboDojo && ../../sim/venvs/robodojo/bin/python utils/update_embodiment_config_path.py)
```

下载器依赖 requests、huggingface_hub/hf_xet，使用固定清单，验证 Git blob SHA-1
或 LFS SHA-256，并复用已校验文件。一分钟吞吐低于 1 MB/s 会停止。
最后一条原生工具生成当前机器的 cuRobo 绝对资产路径，不修改机器人参数。

## 服务与检查

复用已有服务时无需启动新的 worker。需要自行部署时，准备好 GPU 资源，
在当前仓库执行：

```bash
source scripts/robodojo_env.sh
# 阅读并接受 NVIDIA EULA 后设置；不提供自动系统安装。
export OMNI_KIT_ACCEPT_EULA=YES
.venv/bin/python -m scripts.robodojo_single_worker_server \
  --worker-python "$OPENETA_ROBODOJO_PYTHON" \
  --experience "$OPENETA_ROBODOJO_EXPERIENCE" \
  --output outputs/robodojo-server-new --host 127.0.0.1 --port 9889
```

入口沿用本地实验的预启动单 worker 方式，保留调用方的 `CUDA_VISIBLE_DEVICES`，
禁止 manager 另建 worker；无需 Slurm。默认回环地址，跨机器可自行做 SSH 转发。
`server-ready.json` 表示 worker 健康检查通过；不代表任务成功。
本机曾遇到计算节点无法访问登录节点回环代理的问题；只在确有该问题时为仿真
子进程去掉失效代理，不修改系统网络设置。其他环境仍可使用上游
`scripts/start_robodojo_v112_server.sh`；其通用 GPU 选择路径与本机单 worker 部署不同。

两个环境检查脚本也保留在仓库中，必须在已分配 GPU 的环境运行：

```bash
export OPENETA_SMOKE_OUTPUT="$PWD/outputs/isaac-smoke-new"
"$OPENETA_ROBODOJO_PYTHON" scripts/local_isaac_smoke.py \
  --/app/extensions/registryEnabled=false

"$OPENETA_ROBODOJO_PYTHON" scripts/local_robodojo_task_smoke.py \
  --task pour_balls_into_vase --output outputs/task-smoke-new \
  --headless --enable_cameras --experience "$OPENETA_ROBODOJO_EXPERIENCE" \
  --kit_args='--/app/extensions/registryEnabled=false'
```

本机验证：RTX 4090、Ubuntu 22.04、驱动 560.28.03；CUDA 张量、方块下落、
相机渲染、Isaac Lab/原生 RoboDojo 导入、cuRobo GPU FK 已通过。
三任务均完成 reset、三路 640×480 相机、左臂上移 2 cm，右臂未变化；
每个任务 2 次原生动作、20 个物理步。倒球初次检查存在在线材质错误，
后两项处理失效代理后没有 Error 级日志。它们都是环境检查，无模型 API。

后续 Codex 和 OpenETA 倒球试跑均未完成任务，详见
[当前进度](../PROGRESS.md) 和 [实验摘录](../docs/experiments/robodojo-local-20261019.json)。
新整理的可移植入口只做 CPU 验证，本次发布未启动新 GPU 作业或模型请求。
