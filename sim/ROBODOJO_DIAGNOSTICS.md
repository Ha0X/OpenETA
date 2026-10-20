# RoboDojo 端点卡点诊断

现有环境安装和 Slurm 启动约定见参考工作区的
[`ROBODOJO_LOCAL.md`](../../OpenETA-robodojo/sim/ROBODOJO_LOCAL.md)。
GPU 作业须获得授权，保留 Slurm 的 `CUDA_VISIBLE_DEVICES`。

`scripts/robodojo_endpoint_diagnostic.py` 是操作员诊断工具，不是运行时模型工具。
它只在独立诊断 worker 内包装原 `worker_bridge.operate`，不修改参考工作区、
Host schema、IK、碰撞判据或正式运行入口。探测结果写到操作员目录，不加入
Host 的公开回复，也不用于向模型注入姿态或动作。

## 使用顺序

1. 用已验证的参考解释器和 Isaac experience 启动独立 worker。
   将原 worker 入口替换为以下包装器，后接原 worker 的参数：

   ```bash
   "$reference_root/sim/venvs/robodojo/bin/python" \
     "$main_root/scripts/robodojo_endpoint_diagnostic.py" worker \
     --reference-root "$reference_root" --output "$diagnostic_output" \
     --bench robodojo --port 0 --headless --enable_cameras \
     --experience "$validated_experience"
   ```

2. 通过 OpenETA 的静态回放 backend 和原 Host 重放所选调用前缀；不连接模型
   API。记录各次请求及执行结果，出现关键分歧即停止后续回放。核对检查点的
   TCP、关节及原始图像；机器人状态接近不能代替物体状态核验。
3. 在检查点用原 Host 的 `target_matrix` 函数生成预检计划：

   ```bash
   "$reference_root/.venv/bin/python" \
     "$main_root/scripts/robodojo_endpoint_diagnostic.py" prepare \
     --reference-root "$reference_root" --output "$diagnostic_output" \
     --baseline "$baseline_attempt" --after-call "$checkpoint_call" \
     --packet "$replay_attempt/host/last-observation.json"
   ```

4. 再经原 Host 调用一次 `episode_status`。包装器消费一次 `probe-request.json`，
   在同一冻结场景上调用原 `combined_goals`；生成 `endpoint-probes.json`。
   该文件不经 MCP 返回模型。随后正常关闭 Host、环境和本次 worker。

生成计划包含检查点实测姿态、其后最多五个原失败端点、两次重复求解，以及
围绕首个碰撞失败目标分别作世界坐标轴 ±2 cm 平移和 ±15° 旋转的对照。
目标来自指定日志，不内置任务坐标。仅支持同一机械臂、单端点请求。
最多 20 个预检，总预检预算 600 秒，每次预检 deadline 最多 30 秒。
deadline 由原求解／检查代码执行，不是单个 GPU 内核的强制中断保证；
Slurm 作业还应有独立总时限。

预检禁止调用控制器 `_action`，并逐项比较原生动作计数、物理步计数、
关节、TCP 和观测时钟；任何变化立即停止。开始新的独立诊断应使用新的输出
目录，避免上次 `probe-request.json` 在初始化状态请求时被读取。

## 结果口径

- 回放执行了真实动作，但动作来自历史记录，不是模型自主决策，不能统计为
  任务成功率。
- `endpoint_admissible=true` 仅表示本次找到通过原端点检查的候选。
  尚未检查到该候选的路径，也未验证持物运动、物体扫掠、倾倒或任务成功。
- 相同端点重试差异可能来自有限 IK 搜索；失败不证明不可达。碰撞详情只代表
  被标注的一个拒绝候选，不能推广到所有候选。
- 固定代码快照、preset 哈希、调用前缀、检查点差异、物理计数和原始回执；
  保留失败结果。不得把诊断找到的坐标写成任务策略。

一次完整部署的脚本位于本机
`outputs/robodojo-endpoint-diagnostic-20261020-183157/`：
`job.sbatch`、`server_bootstrap.py`、`client.py`，以及从本地提交导出的
`source/`。它们使用现有环境，不安装依赖或修改系统配置；产物不随 Git 上传。
