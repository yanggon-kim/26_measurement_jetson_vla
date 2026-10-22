#!/bin/bash
# Run ONE allow-listed command as host root, through docker (the agent has no sudo on this Thor; the owner chose this
# route on 2026-10-22 instead of a sudoers file, docs/05_setup.md H1). The container joins the host's mount, UTS, IPC,
# network and PID namespaces (nsenter -t 1), so the command sees the host's /sys, /dev and binaries.
# Anything not listed below is refused: never nvpmodel -m, never reboot, never boot / kernel configuration (AGENTS.md).
#
# Usage: env/rootdo.sh jetson_clocks [--fan|--show]
#        env/rootdo.sh nvpmodel -q [--verbose]
#        env/rootdo.sh tegrastats <args>
#        env/rootdo.sh cat /sys/kernel/debug/<path>
#        env/rootdo.sh chown-repo                      # chown -R the repository and scratch back to the agent user
#        env/rootdo.sh ncu <ncu args> <python> <script> <args>
# ncu runs with a clean environment (env -i): only the variables named in PASS below are forwarded, never tokens.
set -euo pipefail

IMAGE="rbonghi/jetson_stats:latest"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ME_USER="$(id -un)"
ME_GROUP="$(id -gn)"
NCU="$HOME/opt/cuda-root/opt/nvidia/nsight-compute/2025.3.0/ncu"
PASS=(HOME PATH CUDA_HOME HF_HOME HF_HUB_OFFLINE TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR
      TRITON_PTXAS_PATH TRITON_PTXAS_BLACKWELL_PATH TRITON_CUOBJDUMP_PATH TRITON_NVDISASM_PATH
      MUJOCO_GL PYTHONHASHSEED CUBLAS_WORKSPACE_CONFIG TORCH_LOGS)

die() { echo "rootdo: refused: $*" >&2; exit 2; }
[ $# -ge 1 ] || die "no command"

run_root() {
  local envs=()
  for v in "${PASS[@]}"; do
    if [ -n "${!v-}" ]; then envs+=("$v=${!v}"); fi
  done
  exec docker run --rm --privileged --pid=host --net=host --entrypoint nsenter "$IMAGE" \
    -t 1 -m -u -i -n -- /usr/bin/env -i -C "$PWD" "${envs[@]}" "$@"
}

cmd="$1"; shift
case "$cmd" in
  jetson_clocks)
    case "${*:-}" in ""|"--fan"|"--show") run_root /usr/bin/jetson_clocks "$@" ;; *) die "jetson_clocks $*" ;; esac ;;
  nvpmodel)
    case "$*" in "-q"|"-q --verbose") run_root /usr/sbin/nvpmodel "$@" ;; *) die "nvpmodel $* (query only)" ;; esac ;;
  tegrastats)
    run_root /usr/bin/tegrastats "$@" ;;
  cat)
    [ $# -eq 1 ] && [[ "$1" == /sys/kernel/debug/* ]] && [[ "$1" != *..* ]] || die "cat $*"
    run_root /usr/bin/cat "$1" ;;
  chown-repo)
    [ $# -eq 0 ] || die "chown-repo takes no arguments"
    run_root /usr/bin/chown -R "$ME_USER:$ME_GROUP" "$REPO" ;;
  ncu)
    run_root "$NCU" "$@" ;;
  *)
    die "$cmd is not on the allow list" ;;
esac
