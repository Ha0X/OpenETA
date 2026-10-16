#!/usr/bin/env bash
# Public, machine-independent equivalent of the tested v112 server configuration.
set -euo pipefail
RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ROBODOJO_ROOT:?Set ROBODOJO_ROOT to your pinned upstream checkout}"
: "${ROBODOJO_ASSET_ROOT:?Set ROBODOJO_ASSET_ROOT to your downloaded assets}"
export PYTHONPATH="$RELEASE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ROBODOJO_ROBOT_CONFIG="$RELEASE_ROOT/sim/envs/robodojo/config/single_x5_left_mount.yml"
export OPENETA_ROBODOJO_DUAL=1
export OPENETA_ROBODOJO_SUPPORT_POLICY=native_auxiliary_ownership_v1
export OPENETA_ROBODOJO_REGISTER_UNTESTED=1
export OPENETA_ROBODOJO_INITIALIZATION_PROFILE=egg_precheck_settle_300_v1
export OPENETA_SIM_SESSION_TTL_S="${OPENETA_SIM_SESSION_TTL_S:-21600}"
export OPENETA_WORKER_POOL_MAX="${OPENETA_WORKER_POOL_MAX:-1}"
export OPENETA_WORKER_LOG_DIR="${OPENETA_WORKER_LOG_DIR:-$RELEASE_ROOT/outputs/robodojo-v112/worker-logs}"
export OPENETA_EPISODE_RECORDING_DIR="${OPENETA_EPISODE_RECORDING_DIR:-$RELEASE_ROOT/outputs/robodojo-v112/recordings}"
unset ROBODOJO_DUAL_ROBOT_CONFIG ROBODOJO_POLICY_PORT ROBODOJO_POLICY_URL
unset OPENETA_ROBODOJO_RUNTIME_PROFILE OPENETA_ROBODOJO_CLOTH_MATERIAL_PROFILE
cd "$RELEASE_ROOT"
exec "${OPENETA_SERVER_PYTHON:-python}" -u -m sim.mcp_server.server \
  --host 127.0.0.1 --port "${OPENETA_SERVER_PORT:-9889}" "$@"
