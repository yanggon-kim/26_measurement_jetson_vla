#!/usr/bin/env bash
# T006 / E8 — the LIBERO closed-loop simulator stack on Jetson AGX Orin, JetPack 6.2.3.
#
# VERIFIED WORKING NATIVELY on 2026-09-22: MUJOCO_GL=egl offscreen rendering with no X
# display, LiberoEnv construct + reset + step. The measurement plan's Docker fallback
# (docs/01_measurement_plan.md:159) is NOT needed.
#
# Measured on this device with libero_spatial task 0, 256x256, two cameras:
#   env construction      15.3 s per task
#   simulation + render   37.80 ms per control step  (26.5 steps/s)
#   max_episode_steps     280 (libero_spatial)
#
# WHY A SEPARATE VENV. ~/venvs/pi0 pins numpy<2, which is load-bearing for the JetPack
# torch wheel, and LIBERO's own requirements.txt pins numpy==1.22.4. Installing LIBERO
# into ~/venvs/pi0 would fight that pin. Instead ~/venvs/libero holds the simulator and
# reaches into ~/venvs/pi0 for torch and lerobot through a .pth file. Both are numpy
# 1.26.4, so nothing is shadowed.
#
# THE THREE VERSION CONSTRAINTS THAT ACTUALLY MATTER (each found by a failure, not a doc):
#   robosuite==1.4.1  — 1.5.x deleted robosuite.environments.manipulation.single_arm_env,
#                       which LIBERO's bddl_base_domain.py imports at module level.
#   mujoco==2.3.7     — robosuite 1.4.1's binding_utils.get_joint_qpos_addr() asserts on
#                       mujoco 3.x: `assert joint_type in (mjJNT_HINGE, mjJNT_SLIDE)`
#                       fires while setting up the robot's joint references. robosuite
#                       declares only `mujoco>=2.3.0`, so pip happily installs 3.13.
#   bddl==1.0.1       — LIBERO's own pin.
set -uo pipefail
V=$HOME/venvs/libero
PI0=$HOME/venvs/pi0/lib/python3.10/site-packages

python3 -m venv "$V"
"$V/bin/pip" install --upgrade pip setuptools wheel

# simulator, at the versions LIBERO actually works with
"$V/bin/pip" install "mujoco==2.3.7" "robosuite==1.4.1" "bddl==1.0.1"

# LIBERO from source — it is not on PyPI
[ -d "$HOME/LIBERO" ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$HOME/LIBERO"
# pinned commit used for these measurements:
git -C "$HOME/LIBERO" checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01

# LIBERO's transitive imports, minus numpy/torch/transformers (they would fight the pins)
"$V/bin/pip" install "gym==0.25.2" "hydra-core==1.2.0" easydict thop cloudpickle future \
                     einops matplotlib opencv-python

# `pip install -e ~/LIBERO` DOES NOT WORK: libero/ has no __init__.py, so setup.py's
# find_packages() returns nothing and the editable finder maps zero modules. Put the
# source tree and the pi0 site-packages on the path instead.
cat > "$V/lib/python3.10/site-packages/zz_libero_bridge.pth" <<PTH
$HOME/LIBERO
$PI0
PTH

# LIBERO prompts interactively on first import ("Do you want to specify a custom path for
# the dataset folder?"), which deadlocks any non-interactive run. Write its config first.
# `datasets` stays absent on purpose: it holds the demonstration data for training, and
# closed-loop evaluation does not read it.
"$V/bin/python" - <<'PY'
import os, yaml
root = os.path.expanduser("~/LIBERO/libero/libero")
os.makedirs(os.path.expanduser("~/.libero"), exist_ok=True)
yaml.dump({"benchmark_root": root,
           "bddl_files": os.path.join(root, "./bddl_files"),
           "init_states": os.path.join(root, "./init_files"),
           "datasets": os.path.join(root, "../datasets"),
           "assets": os.path.join(root, "./assets")},
          open(os.path.expanduser("~/.libero/config.yaml"), "w"))
PY

# feasibility check — this is the one that decides native vs Docker
MUJOCO_GL=egl "$V/bin/python" -c "
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_string(\"<mujoco><worldbody><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>\")
r = mujoco.Renderer(m, height=128, width=128); d = mujoco.MjData(m)
mujoco.mj_forward(m, d); r.update_scene(d)
print('EGL offscreen render OK', r.render().shape)"
echo "Errors from OpenGL.raw.EGL in __del__ during teardown are harmless and expected."
