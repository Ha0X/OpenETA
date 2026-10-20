# Source from the checkout. No installation, resource allocation or GPU selection.
OPENETA_CHECKOUT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROBODOJO_ROOT="${ROBODOJO_ROOT:-$OPENETA_CHECKOUT/third_party/RoboDojo}"
export ROBODOJO_ASSET_ROOT="${ROBODOJO_ASSET_ROOT:-$ROBODOJO_ROOT/Assets}"
export ROBODOJO_ENV_CFG=arx_x5
export OPENETA_ROBODOJO_DUAL=1
export OPENETA_WORKER_POOL_MAX=1
export OPENETA_ROBODOJO_SUPPORT_POLICY=native_auxiliary_ownership_v1
export OPENETA_ROBODOJO_REGISTER_UNTESTED=1
export OPENETA_ROBODOJO_INITIALIZATION_PROFILE=egg_precheck_settle_300_v1
export OPENETA_SERVER_PYTHON="${OPENETA_SERVER_PYTHON:-$OPENETA_CHECKOUT/.venv/bin/python}"
export OPENETA_ROBODOJO_PYTHON="${OPENETA_ROBODOJO_PYTHON:-$OPENETA_CHECKOUT/sim/venvs/robodojo/bin/python}"
export OPENETA_ROBODOJO_EXPERIENCE="${OPENETA_ROBODOJO_EXPERIENCE:-$ROBODOJO_ROOT/third_party/IsaacLab/apps/isaaclab.local.python.kit}"
export PYTHONNOUSERSITE=1
# Accept the NVIDIA EULA explicitly in your own environment before starting Isaac.
# Do not source LIBERO's software-rendering environment or overwrite
# CUDA_VISIBLE_DEVICES provided by the resource allocator.
