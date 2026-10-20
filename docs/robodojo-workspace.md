# RoboDojo：从当前 main 接续工作

这个 fork 的 `main` 同时包含 OpenETA 自主 harness、Codex 入口，以及两者共用的
RoboDojo Host、控制器、仿真适配与公共技能。不需要切换分支或另 clone 一个
`OpenETA-robodojo`。原参考工作区的代码来自 `880522b`，OpenETA 迭代截至
`4bf70c2`；两者的历史均保留在 Git 中。

## Clone 和准备

```bash
git clone https://github.com/Ha0X/OpenETA.git
cd OpenETA
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

仿真环境单独安装，版本、兼容修正、资产子集和已验证范围见
[本机安装记录](../sim/ROBODOJO_LOCAL.md)。不要把 Isaac 的依赖和 Host/MCP
安装进同一环境。大型资产、环境目录、登录凭据和完整运行产物没有打包进 Git。
本仓库包含固定版本清单及下载器，但 clone 本身不会安装 Isaac 或下载资产。
Slurm 是本机资源分配方式，不是 harness 依赖；也可连接已有的 RoboDojo MCP 服务。

先检查 15 个 v112 preset 的原始源码绑定（不调用模型或仿真）：

```bash
python - <<'PY'
import json
from tools.harness_config import ConfigRegistry
registry = ConfigRegistry()
for row in json.load(open('configs/harness/presets.json'))['presets']:
    if row['id'].endswith('_repair_v112'):
        registry.resolve(row['id']).require_executable()
        print(row['id'])
PY
```

## 两种运行入口

两条路线都连接相同 Host/MCP 和 Isaac worker，工具检查及官方成功判据保持不变。
`tools/codex_mcp_server.py` 是共享任务 Host 的模块名；OpenETA 运行它不需要
Codex CLI 或 Codex 登录。

以下示例假定已有可用服务，并从原生 reset 获得任务原文。当前倒球任务原文是
`Pour all the balls from the cup into the vase.`，preset 的补充说明另行绑定。

```bash
export SIM_URL=http://127.0.0.1:9889/mcp
printf '%s\n' 'Pour all the balls from the cup into the vase.' > task.txt
```

OpenETA 使用自己的 provider 配置，可通过环境变量或忽略的 `.env` 提供：

```bash
export OPENETA_LLM_PROVIDER=icodeeasy
export OPENETA_LLM_API_BASE='<your OpenAI-compatible API base>'
export OPENETA_LLM_API_KEY='<your API key>'
export OPENETA_LLM_TIMEOUT_S=60
export OPENETA_LLM_MAX_ATTEMPTS=1
python -m scripts.robodojo_openeta_run \
  --sim-url "$SIM_URL" \
  --config-id robodojo_dual_x5_pour_balls_into_vase_repair_v112 \
  --env-id openeta/robodojo_dual_x5_pour_balls_into_vase-v0 \
  --task-file task.txt --seed 0 \
  --model gpt-6-astra --reasoning-effort high --recommend-pick-place \
  --timeout 1200 --max-turns 80 --max-tool-calls 82 \
  --output outputs/robodojo-openeta-new
```

默认 Host 就在当前仓库，解释器为 `.venv/bin/python`；
可用 `--host-python` 指定已准备的 Host 解释器，`--robodojo-root` 保留用于旧版本对照。
示例沿用最近已完成的基线预算。下一次 1800 秒自主试跑仍是待确认实验，
不是已经执行的新基线。

Codex 使用自身 CLI 和登录。准备好自己的文件式认证后：

```bash
python -m scripts.codex_configured_run \
  --sim-url "$SIM_URL" \
  --config-id robodojo_dual_x5_pour_balls_into_vase_repair_v112 \
  --env-id openeta/robodojo_dual_x5_pour_balls_into_vase-v0 \
  --task "$(cat task.txt)" --seed 0 \
  --model gpt-6-astra --effort high --model-transport https \
  --max-requests 80 --timeout 1200 --episode-action-budget profile \
  --output outputs/robodojo-codex-new
```

本地 Codex 基线使用 CLI 0.163.0。两种入口都要求新输出目录；模型可用性与
认证由各自服务决定，不随仓库提供。`--smoke-test` / `--observe-only` 会启动真实
仿真并可能调用模型，不能当成离线测试。

## 当前进度与下一步

| 路线 | 最新已完成条件 | 结果 |
| --- | --- | --- |
| Codex | seed 0、Astra/high、1200 秒、80 Host 请求 | 抓杯并倾倒，但有球落桌；37 次工具调用、358 次原生动作；未成功 |
| OpenETA | seed 0、icodeeasy Astra/high、1200 秒、80 轮／82 Host 请求、推荐读取 pick-place | 30 次模型调用无 API 错误；抓杯并部分搬运；后续端点预检受阻；未成功 |

两条路线的公开逐次请求、TCP、动作回执、源码版本与关键结果收录于
[本地实验摘录](experiments/robodojo-local-20261019.json)。这是操作员复盘数据，
不作为模型提示或技能，也不是可恢复的仿真快照；图像、完整 provider 请求和视频
仍在原工作区的 `outputs/` 中，历史路径记录于 [PROGRESS.md](../PROGRESS.md)。

当前优先诊断 OpenETA 第 23 次调用持杯搬运之后的卡点：后续五次拒绝均为零物理步，
包括三次无合格 IK 和两次自碰撞候选。一次独立诊断回放已在原机器排队，
固定源码为 `8795e7a`，无模型 API 调用；尚无实时诊断结论。操作员工具及边界见
[诊断文档](../sim/ROBODOJO_DIAGNOSTICS.md)。

技能正文跨轮保留、超时收尾和晚到用量统计是已知待办，尚未修复。
接续时读 [PLAN.md](../PLAN.md) 和 [PROGRESS.md](../PROGRESS.md)，先分析诊断证据，
再讨论行为改动。不能将有限 IK 拒绝视为不可达证明，也不应硬编码这次场景的成功轨迹。

上游 [v112 发布说明](robodojo-v112.md) 和 RSI 数据保留其原实验口径；
其中历史成功率不等于这里两次本地运行的结果。

## 验证范围

本次整理通过 366 项 CPU 测试，命令如下（Host 环境）：

```bash
python -m pytest -q \
  tests/test_robodojo_workspace.py tests/test_robodojo_openeta.py \
  tests/test_robodojo_endpoint_diagnostic.py tests/test_backend_tool_schemas.py \
  tests/test_robodojo_protocol_roles.py tests/test_tool_call_budget.py \
  tests/test_success_evidence.py tests/test_robodojo_public_packaging.py \
  tests/test_codex_robodojo_host.py tests/test_harness_robodojo_adapter.py \
  tests/test_harness_robodojo_wire.py tests/test_robodojo_skill_library.py \
  tests/test_provider_fallback.py tests/test_planner_tool_registry.py \
  tests/test_tool_feedback_episode_environment.py \
  tests/test_response_artifacts.py tests/test_mcp_image_artifacts.py
```

全仓库历史测试不保证在基础 Host 环境通过。扩展检查排除 `tests/integration`、
`tests/tools`、`tests/real` 后，仍有 269 failures 和 33 errors；包括缺少仿真依赖、
未公开的历史夹具、刻意保留的旧源码绑定、生成合同文件落后于参考源码，以及
主机 Landlock ABI 不满足 sandbox 要求。没有为了全绿放宽保护或重签旧 preset。
原始检查日志保留在本机 `outputs/publish-cpu-tests.log`。新入口未重新做真实
Isaac／模型端到端试跑，历史任务结果只适用于各自记录的源码和环境。
