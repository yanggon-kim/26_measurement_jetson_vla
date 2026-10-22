# 05 — Setup: requirements, availability checks, and asking the human

T001 executes this file. **Do everything you can alone first.** Ask the human only for what needs a password, a
licence click, a token, a key on GitHub, a power mode or a reboot — and then only with exact copy-paste commands.
Everything marked *hint* is unverified: detect it and record what you found.

## 1. Rules for asking the human
1. Try every alternative in §2 first; record what you tried in `STATUS.md`.
2. Write each item under **"Needs the human"** at the top of **both** `STATUS.md` and `PROGRESS.md`, in this form:
   ```
   - [ ] H<n> <short title> — why: <one line>. Blocks: <tasks or steps>. Check after: `<command that proves it is done>`
     ```bash
     <exact commands, with every placeholder already filled in by you>
     ```
   ```
   Fill in every path, user name and mode id yourself (`id -un`, `command -v ncu`, …); the human pastes, never edits.
3. Continue with everything that does not depend on the item. At the start of every task, re-run each item's
   "check after" command; when it passes, tick the item, move it to the log, and resume the blocked work.
4. **Never** write a password, token or private key into the repository, a logged command line, `meta.json` or a
   commit message. A session-only `SUDO_ASKPASS` helper outside the repository is acceptable if the human offers a
   password in the terminal; it does not survive a reboot.
5. Never reboot, never confirm a reboot prompt, never run `nvpmodel -m`, never change boot configuration, kernel
   modules or kernel parameters yourself, even if the human's item would do so: that is why it is the human's.

## 2. Requirements and availability checks (the T001 matrix)
T001 writes this matrix to `env/README.md` (`item | check | expected | found | status ok/missing/blocked | action
taken | H-item`) and keeps it current.

| # | item | check | expected / alternatives you try yourself |
|---|---|---|---|
| A1 | board, L4T, JetPack | `cat /proc/device-tree/model`, `cat /etc/nv_tegra_release`, `dpkg -l nvidia-jetpack`, `uname -a` | a Jetson AGX Thor; JetPack 7.x / Jetson Linux r38 or later (*hint*). Record what is there; **never reflash**. If JetPack cannot run `torch.compile` with any route of A8, report it and stop at T001 |
| A2 | CUDA, driver | `nvcc --version` (also `/usr/local/cuda*/bin`), `cat /proc/driver/nvidia/version` or `nvidia-smi`, `ls /usr/local/cuda*/lib64` | CUDA 13.x (*hint*). Put the toolkit `bin` on `PATH` in your shell profile if it is not |
| A3 | Python | `python3 --version`, `python3 -m venv --help`, `python3 -m pip --version` | system Python 3.12 (*hint*, Ubuntu 24.04). Install `python3-venv`, `python3-dev`, `python3-pip` with apt if missing; `uv` or conda for a second Python if a model repository needs one |
| A4 | disk | `df -h ~ /` | ≥ 150 GB free for environments, checkpoints and scratch. If less, clean your own caches; then H8 |
| A5 | network | `curl -sI` https://github.com, https://pypi.org/simple/, https://download.pytorch.org/whl/, https://huggingface.co, https://pypi.jetson-ai-lab.io (*hint*), https://nvcr.io | all reachable; if a proxy is needed, H9 |
| A6 | sudo | `sudo -n true`; then per command: `sudo -n nvpmodel -q`, `sudo -n jetson_clocks --show`, `sudo -n tegrastats --interval 1000 & sleep 2; sudo -n tegrastats --stop`, `sudo -n apt-get -s install zstd` | passwordless. If missing: H1 (and use a session `SUDO_ASKPASS` helper only if the human offers a password) |
| A7 | git | `git config user.name`, `git config user.email`, `git ls-remote origin`, `git push --dry-run origin thor` | identity: if empty, set it **repo-local**: `git config user.name "yanggon-kim"`, `git config user.email "yanggon@g.ucla.edu"`. Push: if refused, generate a deploy key yourself (`ssh-keygen -t ed25519 -N "" -f ~/.ssh/thor_deploy -C thor-agent`), configure `~/.ssh/config` for `github.com` with it, set `origin` to the SSH URL, then H3 (the human adds the **public** key) |
| A8 | PyTorch with CUDA for this GPU | in each venv: `python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability(), torch.cuda.get_arch_list())"`; a bf16 GEMM; `torch.compile` of a small MLP in `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`; `triton` import | try in this order and keep the first route that passes every check: (a) PyTorch's own CUDA 13 aarch64 (SBSA) wheels, `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130` (*hint*; some builds omit NVPL / cuDSS dependencies — install them if the import says so); (b) the Jetson AI Lab index for JetPack 7 (*hint*: `https://pypi.jetson-ai-lab.io/jp7/cu130`; check that the path exists) **[corrected T001, 2026-10-22: `jp7/cu130` returns HTTP 404; the index that exists is `https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple/`, used for flash-attn and diffusers in the GR00T envs]**; (c) NVIDIA's NGC PyTorch container `nvcr.io/nvidia/pytorch:<yy.mm>-py3` with the NVIDIA container runtime (*hint*: Thor support from 25.08; needs H6 if `docker` needs sudo; profiling then runs inside the container — record how); (d) a source build (last resort). The architecture must appear in `get_arch_list()` or the wheel must JIT from PTX — a "no kernel image" error fails the route |
| A9 | profilers and binary tools | `command -v nsys ncu cuobjdump nvdisasm c++filt zstd`; `nsys --version`; `ncu --version` | install what is missing (apt: the CUDA toolkit's `cuda-nsight-compute-*`, `cuda-nsight-systems-*` or `nsight-systems-*` packages, `cuda-cuobjdump-*`, `zstd`); if apt has none, NVIDIA's standalone installers into your home directory |
| A10 | ncu permission | the T001 toy run without sudo | runs unprivileged → record and use it. `ERR_NVGPUCTRPERM` → run as root via the `SETENV` sudoers line of H1; the module option of H7 is the alternative the human may choose |
| A11 | power monitor and tegrastats | `ls /sys/bus/i2c/drivers/*/`, `grep -H . /sys/class/hwmon/hwmon*/name /sys/class/hwmon/hwmon*/*_label`, one `tegrastats` line | the rail table of protocol §4 in `env/README.md` |
| A12 | clocks and power mode | `sudo nvpmodel -q --verbose`, `cat /etc/nvpmodel.conf`, `sudo jetson_clocks --show`, the GPU devfreq node (`ls /sys/class/devfreq/`), the EMC rate (`sudo ls /sys/kernel/debug/bpmp/debug/clk/` — the Orin path is a *hint*) | record the maximum-performance mode (protocol §1). If the board is not in it: H5. Never `nvpmodel -m` |
| A13 | Hugging Face | `hf auth whoami` (or `huggingface-cli whoami`); `hf download <repo> --revision <rev> config.json` for every checkpoint of `00_context.md` | public checkpoints download without a token. **π0 / π0.5 need the PaliGemma tokenizer**, which is gated: first try the byte-identical ungated copy the Orin agent used (`reference/orin/env/setup_pi0.sh`, the lines about `paligemma_tokenizer.model`, sha256 `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`) and point the processor at it locally; only if that fails: H2 |
| A14 | model code | `git clone` + `git checkout` of every pin in `00_context.md` (outside the repository, e.g. `~/src/`) | install with `pip install --no-deps -e`, then add dependencies by hand; re-run A8's checks after every install |
| A15 | LIBERO | `MUJOCO_GL=egl python -c "import mujoco, robosuite, libero"` in its own venv | `reference/orin/env/setup_libero.sh`; install EGL / OSMesa packages with apt |
| A16 | one inference per model | the six models on the fixture, eager, default N, finite actions | T001 step 8 |
| A17 | nsys secrets | T001 step 6 | decides whether nsys binaries may be committed |

## 3. "Needs the human" templates (copy, fill in every placeholder, paste into both boxes)
Write only the items that apply. `<USER>` = `id -un`; `<NCU>` = `command -v ncu`; `<NSYS>` = `command -v nsys`;
`<NVPMODEL>`, `<JETSON_CLOCKS>`, `<TEGRASTATS>`, `<CHOWN>`, `<CAT>`, `<APT>` = `command -v` of each, resolved to
real paths (`readlink -f`). Verify every line you write with `visudo -c -f` on a copy in `/tmp` first.

**H1 — passwordless sudo for the measurement commands.** Why: `jetson_clocks` before every run, `tegrastats`,
reading debugfs clocks, apt installs, and `ncu` if it needs root. `nvpmodel` is limited to queries, so the agent
cannot change the power mode. Check after: `sudo -n jetson_clocks --show && sudo -n nvpmodel -q`.
```bash
sudo tee /etc/sudoers.d/90-thor-agent >/dev/null <<'EOF'
Cmnd_Alias THOR_AGENT = <NVPMODEL> -q, <NVPMODEL> -q --verbose, <JETSON_CLOCKS>, <JETSON_CLOCKS> --fan, <JETSON_CLOCKS> --show, <TEGRASTATS> *, <CAT> /sys/kernel/debug/*, <CHOWN> -R <USER>\:<USER> /home/<USER>/*, <APT> update, <APT> install *
<USER> ALL=(root) NOPASSWD: THOR_AGENT
<USER> ALL=(root) NOPASSWD:SETENV: <NCU>, <NSYS>
EOF
sudo chmod 440 /etc/sudoers.d/90-thor-agent && sudo visudo -c
```
(Drop the last line if A10 found that ncu runs without root. `apt-get install` and `ncu` as root are effectively
root access; the human may decline them, and then each apt install becomes its own H-item. The Orin used a
broader rule including `/bin/sh`, `reference/orin/env/README.md` §11.)

**H2 — Hugging Face login and the Gemma licence** (only if the ungated tokenizer route of A13 failed). Why: π0 and
π0.5 load the PaliGemma tokenizer from the gated repo `google/paligemma-3b-pt-224`. Check after:
`hf auth whoami && hf download google/paligemma-3b-pt-224 tokenizer.model --local-dir /tmp/pg_check`.
```text
1. In a browser, signed in to Hugging Face: open https://huggingface.co/google/paligemma-3b-pt-224 and accept the licence.
2. On the Thor, in a terminal (the token is stored in ~/.cache/huggingface/token, outside the repository):
   hf auth login
```

**H3 — push access to GitHub.** Why: results reach the research side only through `git push`. Check after:
`git -C <REPO> push --dry-run origin thor`.
```text
GitHub → yanggon-kim/26_measurement_jetson_vla → Settings → Deploy keys → Add deploy key
Title: thor-agent        Allow write access: yes
Key (public, safe to paste):
<the full one-line contents of ~/.ssh/thor_deploy.pub>
```
(Alternative the human may prefer: `gh auth login` on the Thor.)

**H5 — the maximum-performance power mode.** Why: every measurement runs in it (protocol §1); switching needs a
reboot, which the agent never confirms. Blocks: every measuring task from T002's GPU parts on. Check after:
`sudo -n nvpmodel -q | grep -F '<MODE_NAME>'`.
```bash
sudo nvpmodel -m <MODE_ID>     # answer YES to the reboot prompt; the mode persists across reboots
# after the reboot, start the agent again; it re-applies jetson_clocks itself
```

**H6 — docker without sudo** (only if A8 chose the container route). Check after: `docker info >/dev/null`.
```bash
sudo usermod -aG docker <USER>   # then log out and back in (or reboot)
```

**H7 — unprivileged GPU profiling** (optional alternative to running ncu as root; needs a reboot). Check after:
the T001 toy ncu run succeeds without sudo.
```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u && sudo reboot
```
(Write this item only after checking that the GPU driver on this device is the `nvidia` kernel module —
`lsmod | grep -E '^nvidia'` — and that the option exists: `modinfo nvidia | grep -i RestrictProfiling`.)

**H8 — disk space.** Why: environments, six checkpoints (≈ 40 GB) and uncompressed traces. Give `df -h` and what
you already cleaned; ask for an NVMe mount or a cleanup the human chooses.

**H9 — network.** Give the failing `curl -sI` lines; ask for proxy settings or a firewall exception.

## 4. What to record in `env/`
`env/README.md` (the matrix of §2, the platform as found, the chosen torch route and why the others failed, the
rail table, the maximum-performance mode and its id, the GPU frequency and EMC paths, the ncu permission finding,
the counter names of protocol §3.3 this device exports), one setup script per environment (`env/setup_<env>.sh`,
re-runnable), `env/requirements_<env>.txt` (`pip freeze`), `env/INSTALL_LOG.md` (every install: command, version,
why), `env/tool_help/` (protocol §3).
