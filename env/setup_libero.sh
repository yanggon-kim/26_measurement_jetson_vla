#!/usr/bin/env bash
# env/setup_libero.sh — the LIBERO closed-loop simulator stack on the Thor (T001 step 7, docs/05_setup.md A15).
#
#   bash env/setup_libero.sh             # re-runnable; rebuilds ~/venvs/libero
#
# Ported from reference/orin/env/setup_libero.sh. The three version constraints the Orin found by failure are
# KEPT: robosuite==1.4.1 (1.5 deleted single_arm_env, which LIBERO imports), mujoco==2.3.7 (robosuite 1.4.1's
# binding_utils asserts on mujoco 3.x joint types), bddl==1.0.1 (LIBERO's pin). Deviations:
#   * mujoco 2.3.7 has NO cp312 aarch64 wheel (PyPI has cp312 aarch64 wheels only from 3.0.0). Its python
#     bindings are built here from the PyPI sdist (sha256 recorded below) against the official prebuilt
#     libmujoco 2.3.7 linux-aarch64 release tarball (unpacked to ~/opt/mujoco-2.3.7), under `nice -n 19` on
#     4 cores so a concurrent measurement is disturbed as little as possible. The wheel is cached in
#     ~/opt/wheels, so a re-run does not rebuild. The build also compiles the `simulate` viewer, whose GLFW
#     needs X11 extension headers (libxrandr-dev & co.) that are not installed and need apt (no sudo): they are
#     fetched with `apt-get download` and unpacked with `dpkg -x` into ~/opt/x11dev (headers only; the
#     unversioned .so links are pointed at the system runtime libraries), used only at build time.
#   * Python 3.12 (system) instead of 3.10; numpy 1.26.4 (LIBERO's 1.22.4 has no cp312 wheel; the Orin also ran
#     1.26.4). torch 2.10.0+cu130 is installed IN this venv (LIBERO's benchmark module imports torch to load
#     init states) instead of the Orin's .pth bridge into the π0 venv: ~/venvs/lerobot runs numpy 2.x, and a
#     bridge would mix numpy majors. How the policy and the simulator meet for E8 is T011's decision.
#   * python3-venv / ensurepip is not installed: `uv` creates the venv.
#   * EGL: no apt needed. The system already has libEGL.so.1 (glvnd), libEGL_nvidia (tegra-egl) and
#     libOpenGL.so.0; PyOpenGL is installed for mujoco.egl.
#   * LIBERO's tree goes on sys.path via a .pth (libero/ has no __init__.py, so `pip install -e` maps nothing),
#     and ~/.libero/config.yaml is written so the first import does not prompt interactively (as on the Orin).
set -euo pipefail

VENV="${VENV:-$HOME/venvs/libero}"
SRC="${SRC:-$HOME/src/LIBERO}"
SHA="8f1084e3132a39270c3a13ebe37270a43ece2a01"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
TORCH_IDX="https://download.pytorch.org/whl/cu130"
PY="$VENV/bin/python"
SP="$VENV/lib/python3.12/site-packages"
MJ_VER="2.3.7"
MJ_LIB_TGZ_SHA="422d52e98fba8b6328d67e00c87d59ba8108f1c0df8a4c048811fa73be9e8c71"   # mujoco-2.3.7-linux-aarch64.tar.gz
MJ_SDIST_SHA="422041f1ce37c6d151fbced1048df626837e94fe3cd9f813585907046336a7d0"     # PyPI mujoco-2.3.7.tar.gz
MJ_HOME="$HOME/opt/mujoco-$MJ_VER"
WHEELS="$HOME/opt/wheels"
LOG_DIR="$REPO/scratch/t001"; mkdir -p "$LOG_DIR" "$WHEELS"

check_torch() {
  "$PY" - <<'PY'
import torch, numpy
assert torch.cuda.is_available(), "CUDA LOST: a step replaced the CUDA torch wheel"
assert torch.version.cuda.startswith("13"), torch.version.cuda
print(f"  ok: torch {torch.__version__} cuda {torch.version.cuda} numpy {numpy.__version__}")
PY
}

[ -d "$SRC/.git" ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$SRC"
git -C "$SRC" checkout -q "$SHA"
[ "$(git -C "$SRC" rev-parse HEAD)" = "$SHA" ]

uv venv -q --clear --python /usr/bin/python3.12 "$VENV"
CONSTRAINTS="$(mktemp)"
printf 'torch==2.10.0+cu130\nnumpy==1.26.4\nmujoco==2.3.7\nrobosuite==1.4.1\n' > "$CONSTRAINTS"
UVI=(uv pip install -q --python "$PY" -c "$CONSTRAINTS"
     --index-url https://pypi.org/simple --extra-index-url "$TORCH_IDX" --index-strategy unsafe-best-match)

# 1. torch (CUDA) and numpy
uv pip install -q --python "$PY" --index-url "$TORCH_IDX" torch==2.10.0
"${UVI[@]}" "numpy==1.26.4"
echo "after torch:"; check_torch

# 2. mujoco 2.3.7 python bindings, built from the sdist against the prebuilt aarch64 library
MJ_WHL="$(ls "$WHEELS"/mujoco-$MJ_VER-cp312-cp312-linux_aarch64.whl 2>/dev/null || true)"
if [ -z "$MJ_WHL" ]; then
  if [ ! -f "$MJ_HOME/lib/libmujoco.so.$MJ_VER" ]; then
    D="$(mktemp -d)"
    curl -sL -o "$D/mj.tgz" "https://github.com/google-deepmind/mujoco/releases/download/$MJ_VER/mujoco-$MJ_VER-linux-aarch64.tar.gz"
    echo "$MJ_LIB_TGZ_SHA  $D/mj.tgz" | sha256sum -c -
    tar -xzf "$D/mj.tgz" -C "$HOME/opt"; rm -rf "$D"
  fi
  X11DEV="$HOME/opt/x11dev"; XLIB="$X11DEV/usr/lib/aarch64-linux-gnu"
  if [ ! -f "$X11DEV/usr/include/X11/extensions/Xrandr.h" ]; then
    DD="$(mktemp -d)"; (cd "$DD" && apt-get download -q libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
        libxrender-dev libxfixes-dev libxext-dev)
    for d in "$DD"/*.deb; do dpkg -x "$d" "$X11DEV"; done; rm -rf "$DD"
  fi
  for l in "$XLIB"/libX*.so; do  # dev symlinks point at runtime libs that live in the system dir
    t="$(readlink "$l")"; case "$t" in /*) ;; *) ln -sfn "/usr/lib/aarch64-linux-gnu/$t" "$l";; esac
  done
  B="$(mktemp -d)"
  SDIST_URL="$(curl -s https://pypi.org/pypi/mujoco/$MJ_VER/json | "$PY" -c \
      'import json,sys; print([u["url"] for u in json.load(sys.stdin)["urls"] if u["packagetype"]=="sdist"][0])')"
  curl -sL -o "$B/mujoco-$MJ_VER.tar.gz" "$SDIST_URL"
  echo "$MJ_SDIST_SHA  $B/mujoco-$MJ_VER.tar.gz" | sha256sum -c -
  # build in an isolated env with numpy<2 / setuptools; nice + 4 cores to spare concurrent measurements
  MUJOCO_PATH="$MJ_HOME" MUJOCO_PLUGIN_PATH="$MJ_HOME/bin/mujoco_plugin" \
  CMAKE_INCLUDE_PATH="$X11DEV/usr/include" CMAKE_LIBRARY_PATH="$XLIB" CPATH="$X11DEV/usr/include" \
    nice -n 19 taskset -c 10-13 uv build -q --wheel --python /usr/bin/python3.12 \
      --out-dir "$WHEELS" "$B/mujoco-$MJ_VER.tar.gz" > "$LOG_DIR/build_mujoco_$MJ_VER.log" 2>&1
  rm -rf "$B"
  MJ_WHL="$(ls "$WHEELS"/mujoco-$MJ_VER-cp312-cp312-linux_aarch64.whl)"
fi
"${UVI[@]}" "$MJ_WHL"

# 3. the simulator stack at the versions LIBERO works with, and LIBERO's transitive imports
#    (minus transformers / robomimic / wandb: training-only; numpy / torch held by the constraints)
"${UVI[@]}" "robosuite==1.4.1" "bddl==1.0.1" "gym==0.25.2" "hydra-core==1.2.0" "easydict==1.9" \
  "thop" "cloudpickle" "future" "einops" "matplotlib" "opencv-python" "PyOpenGL" "pyyaml" "imageio" "h5py"
echo "after simulator:"; check_torch

# 4. LIBERO on the path, and its config (no interactive prompt on first import)
echo "$SRC" > "$SP/zz_libero_src.pth"
"$PY" - "$SRC" <<'PY'
import os, sys, yaml
root = os.path.join(sys.argv[1], "libero", "libero")
os.makedirs(os.path.expanduser("~/.libero"), exist_ok=True)
cfg = {"benchmark_root": root, "bddl_files": os.path.join(root, "./bddl_files"),
       "init_states": os.path.join(root, "./init_files"), "datasets": os.path.join(root, "../datasets"),
       "assets": os.path.join(root, "./assets")}
p = os.path.expanduser("~/.libero/config.yaml")
yaml.dump(cfg, open(p, "w")); print("  wrote", p)
PY

# 5. checks: imports, then one libero_spatial reset rendered offscreen with EGL
cd "$HOME"   # not from inside a source tree
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$PY" -c "import mujoco, robosuite, libero; print('  imports ok: mujoco', mujoco.__version__, '| robosuite', robosuite.__version__)"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$PY" - <<'PY'
import numpy as np, os, torch
# LIBERO's get_task_init_states() torch.load()s its own bundled .pruned_init files (numpy arrays); torch >= 2.6
# defaults weights_only=True and refuses them. Allow-list the numpy reconstructors (as the Orin's
# bench/libero_eval.py does) rather than turning weights_only off. A harness must do the same.
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype,
    np.dtypes.Float64DType, np.dtypes.Int64DType, np.dtypes.UInt8DType])
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0); env.reset()
obs = env.set_init_state(suite.get_task_init_states(0)[0])
for _ in range(5):
    obs, r, done, info = env.step([0.0] * 7)
img = obs["agentview_image"]; wr = obs["robot0_eye_in_hand_image"]
nz = int((img > 0).sum())
print(f"  libero_spatial task 0 '{task.language}': agentview {img.shape} {img.dtype} nonzero={nz} "
      f"mean={img.mean():.1f}; wrist {wr.shape} mean={wr.mean():.1f}")
assert nz > 0 and img.std() > 1.0, "rendered image is blank"
env.close()
PY

uv pip freeze --python "$PY" > "$REPO/env/requirements_libero.txt"
rm -f "$CONSTRAINTS"
echo "LIBERO environment ready: $VENV  (pins: env/requirements_libero.txt). Run with MUJOCO_GL=egl."
echo "Errors from OpenGL.raw.EGL in __del__ during teardown are harmless (Orin note)."
