# RoboDojo harness — public v112 release

This branch packages the v112 client, the backend overlay used in testing, and
its server-side RoboDojo integration. It does not contain private experiment
logs, deployment addresses, account state, or the internal development history.

## Included

- Codex CLI + native MCP closed-loop execution, isolated per-episode sessions,
  tool budgets, durable action/observation records and explicit cleanup.
- Fixed dual-X5 RoboDojo adapters, calibrated images/TCP, native official reward
  tracking and `check_task_completion` with coarse semantic feedback.
- Original task wording plus separately versioned clarifications; exact text is
  in `tools/task_rule_supplements.py`. Rewards are not rewritten.
- Measured finger-axis collision geometry, FK consistency validation, and
  controller stop/gripper-target diagnostics from v112.
- Fifteen source-locked v112 repair configurations. Older configurations are
  retained as historical definitions and may deliberately reject current source
  hashes. Do not repin them and call the result the same historical experiment.

The release does **not** claim to resolve collision-stop grip-force retention,
all safe-retreat cases, all task failures, or network-interrupted action recovery.
An unknown action outcome must not be blindly replayed.

## Client installation

Use Python 3.11+ for a deployment that also runs RoboDojo; the generic host package
supports Python 3.10+. Create an isolated environment and install the host:

```bash
git clone --branch robodojo-v112 https://github.com/OpenMOSS/OpenETA.git
cd OpenETA
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Install and sign in to Codex CLI using your own eligible account. The launcher
uses saved file-based authentication in an isolated temporary Codex home and
removes its copied credential during cleanup. Keyring-only authentication is
not implemented. Account/model access and usage limits still apply; no account
credentials or private model catalog are distributed here.

Validate the release's source bindings without starting a simulator or model:

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

## Simulator installation

A separate Linux NVIDIA/Isaac runtime is required. Install the official
[RoboDojo](https://github.com/robodojo-benchmark/RoboDojo) assets, **Isaac Sim 5.1 /
Isaac Lab 2.3**, and their dependencies according to upstream instructions.
The tested native revision is `ee67a1468510da7624a089164402359f2afc72c8`.
Assets and third-party simulator binaries are not bundled or relicensed.

`scripts/setup_envs.sh --help` lists the existing environment bootstrap flow.
Its `robodojo` option prepares the pinned source checkout; it does not replace
upstream asset download and Isaac installation. Point `sim/venvs/robodojo` at
the dedicated environment whose Python can import Isaac Lab. In that environment,
also install this checkout and the checked-controller dependencies:

```bash
# Run with the dedicated RoboDojo/Isaac environment's Python.
python -m pip install -e . 'python-fcl==0.7.0.11' scipy pyyaml omegaconf websockets
```

Configure paths for your machine (these are examples, not bundled directories):

```bash
export ROBODOJO_ROOT=/path/to/RoboDojo
export ROBODOJO_ASSET_ROOT=/path/to/RoboDojo/Assets
export ROBODOJO_ENV_CFG=arx_x5
export OPENETA_ROBODOJO_DUAL=1
export OPENETA_WORKER_POOL_MAX=1
```

`ROBODOJO_DUAL_ROBOT_CONFIG` is optional; the native task layout normally selects
its dual-arm robot config. Do not reuse an unrelated single-arm override for a
dual-arm task. If dependencies are installed as a separate validated overlay,
`OPENETA_ROBODOJO_FCL_PATH` selects its import path.

Start the server from this checkout, using the host Python that has OpenETA/MCP
installed; the worker manager launches `sim/venvs/robodojo` separately:

```bash
bash scripts/start_robodojo_v112_server.sh
```

The launcher selects the tested support-actor policy and task-local egg settling
profile, and writes logs/recordings under `outputs/`. It enables registration of
manifest candidates; registration is not a claim of task success. It does not
change native rewards or collision thresholds. Architecture-specific runtime
setup (such as an ARM64 OpenMP preload) must follow your Isaac installation.

Connect to `http://127.0.0.1:9889/mcp`. For another machine, provide your own SSH
forwarding or secured endpoint. Keep the simulator bound to a trusted interface.
The private MCP worker operation is for the Host; do not expose simulator state
or reward internals directly to the model.

## One task attempt

Copy the **official instruction returned by the native reset observation** to
`task.txt`; do not substitute the task ID or invent a new description. The preset
binds the corresponding versioned supplement separately.

For example, after preparing the `pour_balls_into_vase` task:

```bash
python -m scripts.codex_configured_run \
  --sim-url http://127.0.0.1:9889/mcp \
  --config-id robodojo_dual_x5_pour_balls_into_vase_repair_v112 \
  --env-id openeta/robodojo_dual_x5_pour_balls_into_vase-v0 \
  --task "$(cat task.txt)" \
  --seed 0 --model gpt-6-astra --effort high \
  --max-requests 200 --timeout 0 \
  --output outputs/pour-balls-new-attempt
```

Choose a model available to your own Codex account. Model/effort substitutions
change the experiment. Output must be a fresh directory. `--observe-only` can
be used first for a no-motion Host check; it still creates a simulator episode
and may invoke the model, so it is not an offline test.

The reported RSI experiments additionally used reviewed, frozen historical
experience on applicable tasks. Those private bundles are not part of this
release. Running the above without them is a new experiment, not an exact replay
of the reported score. The launcher accepts explicitly supplied, hash-verified
experience via its documented `--help` options.

## Results and validation boundaries

See [interim RSI results and the sanitized attempt ledger](robodojo-rsi-results.md).
The 66.7% figure is cross-version cumulative coverage, **not v112-only accuracy**.
Completion feedback is restricted but still a privileged semantic signal.

Public packaging preserves all 105 files bound by the v112 interface hash.
Deployment integration and local endpoint defaults are documented in
[the release manifest](assets/robodojo-v112-public-manifest.json).
Packaging tests run independently of the ongoing benchmark service. A fresh
end-to-end Isaac/model benchmark on the sanitized checkout is not claimed.

Known historical-test limitation: five tests in `test_harness_experience.py`
select old source-locked profiles and reject the v112 geometry files. These
failures are recorded; historical hashes have not been rewritten to hide them.
