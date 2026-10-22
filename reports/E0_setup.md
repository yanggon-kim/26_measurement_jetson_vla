# E0 setup — T001: stack, tools, access, checkpoints, one inference per model (Jetson AGX Thor, 2026-10-22)

## Answer
The Thor is ready to measure all six models. (GR00T N1.7 needed one human item, H2: the owner accepted the gated backbone's licence and logged in to Hugging Face the same day.)

| requirement | result | evidence |
|---|---|---|
| power mode | **MAXN (id 0)** is current and is the maximum-performance mode (no caps); boot default is 120W (id 1) | `results/20261022_E0_toolcheck/nvpmodel_modes.csv` |
| GPU | Thor (ncu chip GB10B), CC 11.0, **`sm_110`**, **20 SMs**, **L2 32 MiB**, GPU clock 314–1575 MHz, EMC 4266 MHz | `env/README.md` §1 |
| PyTorch | torch **2.10.0+cu130** (PyTorch SBSA wheels), triton 3.6.0; bf16 GEMM and `torch.compile` in all four modes pass **only with Triton pointed at CUDA 13.0's `ptxas`** | `results/20261022_E0_toolcheck/stack_verification.json` |
| Inductor `max_autotune_gemm` | disabled by `is_big_gpu` (threshold 68 SMs > 20), as on the Orin | same |
| nsys | 2025.3.2, runs unprivileged | `results/20261022_E0_toolcheck/nsys/` |
| ncu | 2025.3.0, needs root (`ERR_NVGPUCTRPERM`); runs through `env/rootdo.sh` (docker, owner-approved); `--clock-control none` + `jetson_clocks`; no per-process kernel limit at 1,000; sees CUDA-graph nodes | `results/20261022_E0_toolcheck/raw/ncu_*` |
| DRAM bytes | **no `dram__` counters**; L2 sysmem fill / miss counters match the toy's analytic read bytes to 1.003× (validation on ≥ 4 kernels is T003's). ncu's built-in roofline sections need `dram__bytes` and so cannot draw the DRAM roof here | `ncu/metric_presence.csv`, `ncu/roofline_section_metrics.csv` |
| tensor FLOPs | `sm__ops_path_tensor_*` counts executed (tile-padded) MMA operations; tensor-pipe and TMA/TMEM instruction counters exist | same |
| nsys binaries | **NOT CLEAN**: they embed other processes' environments (canary + harness token). Never committed | `results/20261022_E0_secrets_test/` |
| rails | VIN (ina238, module input [ASSUMPTION]), VDD_GPU, VDD_CPU_SOC_MSS, VIN_SYS_5V0 (ina3221); no separate DRAM rail | `env/README.md` §2 |
| models | all six **pass** (real weights, finite actions, eager and compiled per-step callable); N1.7 after H2 with backbone `nvidia/Cosmos-Reason2-2B` @ 9ce19a19, P 273 / S 41; LIBERO renders | `results/20261022_E0_smoke_models/smoke.csv`, `results/20261022_E0_smoke_groot_n17/` |

Measured: nothing in T001 is a performance result. The smoke and stack-check timings are labelled smoke tests
(GPU shared with other users). Derived: the toy's analytic bytes and FLOPs (shapes × dtype sizes). GR00T P and S were read after the review with forward hooks: N1.6 P 284 / S 51,
N1.5 P 814 / S 49 (`results/20261022_E0_addenda/`).

## Conditions
- MAXN throughout; `jetson_clocks` and `jetson_clocks --fan` applied through `env/rootdo.sh` at 03:26 UTC, after
  the first ncu toy and the stack check (both ran with unlocked clocks).
- The device is shared. Other users have long-running sessions, including GPU work. VDD_GPU swung 6.7–30.8 W (10 s sysfs sample; 36.7 W instantaneous in tegrastats) with
  nothing of ours running. On Thor, `nvidia-smi` lists no processes, so other GPU jobs cannot be seen per process.

## What did not work, and what was tried
1. **No sudo.** The owner chose a docker route: `env/rootdo.sh`, allow-listed commands only, never used for installs.
   A sudoers file (H1) was offered and declined.
2. **No `ensurepip`, no CUDA toolkit, no Nsight tools on the image.** Solved without root:
   - `uv` for the venvs;
   - `apt-get download` + `dpkg -x` of the JetPack r38.2 CUDA 13.0 packages into `~/opt/cuda-root`.
3. **Triton could not compile for `sm_110a`.** Its bundled `ptxas-blackwell` is CUDA 12.9. Fixed by
   `TRITON_PTXAS{,_BLACKWELL}_PATH` → CUDA 13.0 `ptxas` (`env/thor_env.sh`). The failure is kept in
   `raw/stack_verification_route_a_bundled_ptxas_FAILED.json`.
4. **π0's first smoke attempt loaded random weights.** lerobot @ f0d2b37b `PI0Policy.from_pretrained` passes
   `kwargs.get("revision")` (always `None`, since `revision` is a named parameter) to `cached_file`, looks up `main`,
   and on failure prints a warning and returns a randomly initialised model. `assert_real_weights` refused it
   (8/400 tensors equal).
   - Fix: load from the pinned snapshot directory, which also guarantees the pinned bytes.
   - Note: `main` has since moved for `lerobot/pi05_libero_base` (5fcf06fc) and `lerobot/smolvla_base` (5e8d12a6),
     away from the pins (a217bfd3, d9f33c94). π0's `main` is still the pin.
5. **The gated PaliGemma tokenizer.**
   - Used the ungated `tokenizer.model` (sha256 matches) plus a minimal `tokenizer_config.json` (Gemma defaults:
     `<bos>`, no `<eos>`, `<pad>` = 0).
   - Conversion needs `sentencepiece`, which the Orin never needed: the gated repo ships a `tokenizer.json`.
   - Cross-check: the fixture prompt gives 21 valid tokens, which is π0's P = 816 with 533 valid, as on the Orin.
6. **GR00T N1.7's backbone is gated.**
   - The environment was verified with a public Qwen3-VL-2B stand-in config (not the pinned configuration): all
     1,031 GR00T parameters came from the checkpoint, and the actions were finite.
   - Pinned smoke: H2, resolved by the owner the same day; passes (`results/20261022_E0_smoke_groot_n17/`).
7. **GR00T N1.5 cannot use its own torch pin** (2.5.1 / 2.8.0; no CUDA 13 build exists). It runs on torch 2.10.0+cu130
   in its own environment. flash-attn 2.8.4 comes prebuilt from the Jetson AI Lab SBSA index, so nothing was built.
8. **nsys leak.** The leak persisted with `--sample=none --cpuctxsw=none` (information only).

## Anomalies and confidence
- GR00T compiled-vs-eager action differences are 3.1 % (N1.6) and 1.2 % (N1.5), against ≤ 0.5 % for the LeRobot
  models. They are not explained yet; T005 checks equivalence for every compiled configuration. Possibly the
  per-call noise is not fully controlled by `torch.manual_seed(0)`.
- `mcc__dram_throughput_*` read 0 on the toy: they may need a different collection scope. T003 checks them again.
- High confidence in every fact in `env/README.md`, each read from this device with its command.

## Reproduce
```bash
bash env/setup_base.sh; bash env/setup_lerobot.sh; bash env/setup_groot16.sh; bash env/setup_groot15.sh
bash env/setup_groot17.sh; bash env/setup_libero.sh
source env/thor_env.sh
~/venvs/base/bin/python bench/stack_check.py --out <dir>/stack_verification.json
env/rootdo.sh jetson_clocks && env/rootdo.sh jetson_clocks --fan
env/rootdo.sh ncu --clock-control none --set full --nvtx --nvtx-include "toy/" --profile-from-start off \
    -o <dir>/raw/ncu_toy -f ~/venvs/base/bin/python bench/ncu_toy.py --case linear
HF_HUB_OFFLINE=1 ~/venvs/lerobot/bin/python bench/smoke_lerobot.py --model pi0 --out <dir>
(cd /tmp && HF_HUB_OFFLINE=1 ~/venvs/groot16/bin/python <repo>/bench/smoke_groot.py --version groot_n16 --out <dir>)
```
Every command, with its exact flags, is in the `meta.json` of the three result directories.

## Deviations
- `docs/00_context.md`: L2 hint corrected to 32 MiB.
- `docs/05_setup.md` A8 (b): the `jp7/cu130` index path is a 404; the existing index is `sbsa/cu130`.
- Both corrections are marked in place "corrected T001, 2026-10-22".
- Step 3's bf16 GEMM check used an fp64 reference instead of fp32: stricter, same pass criterion.
- Step 4g: `cuobjdump --list-elf` was used first; `--list-text` was added after the review (`results/20261022_E0_addenda/`).
- After the independent review, the missing items were added: `ncu/columns.txt`, `ncu/roofline_section_metrics.csv`, the flag spellings, checkpoint hashes, GR00T P and S, saved device facts, corrected idle rail ranges, the secrets-test `meta.json` device fields and the empty toolkit string in the toolcheck `meta.json`.
- Root access through docker instead of a sudoers file (owner's decision).
- `jetson_clocks` was applied part-way through T001 (step 4e). Earlier toy runs had unlocked clocks; none is a result.

## Self-review (protocol §11)
| check | result |
|---|---|
| `env/README.md` holds the matrix and every listed fact (or why it is missing) | yes |
| every requirement ok, or missing / blocked with an H-item in both boxes | yes: all ok; H2 (N1.7) was raised in both boxes and resolved |
| CUDA PyTorch route with `torch.compile` in four modes | yes (route a, with CUDA 13 `ptxas`) |
| toy nsys and toy ncu ran; `metric_presence.csv` answers the three 4d questions | yes |
| nsys-binary verdict recorded, `.gitignore` matches | NOT CLEAN; `.gitignore` unchanged |
| each model has a smoke result | all six pass (N1.7 after H2) |
| `env/setup_*.sh`, `requirements_*.txt`, `INSTALL_LOG.md`, `tool_help/` committed | yes |
| every number traces to `results/` or a stated command | yes after the review fixes (the addenda directory holds the formerly unsaved facts) |
| `meta.json` complete, no null, hashes | yes (sha256 / bytes of every committed raw file recomputed by the reviewer) |
| raw files committed or `committed: false` with a reason; §6g; secrets scan | yes; no file > 50 MB; nsys binaries uncommitted; every committed raw file scanned clean |
| sizes in `STATUS.md`; nothing outside my partition edited except the two marked factual corrections | yes |
| independent pass (P0) | done by a fresh sub-agent; findings fixed as listed under Deviations |
