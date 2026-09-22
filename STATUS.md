# STATUS — on-device agent

*Maintained by the on-device agent. The research side reads it and never edits it.*

## ⛳ STANDING ITEM — E7 capped power modes need two human reboots (do not lose this)

The owner has committed to two reboots for the capped-mode half of E7 and asked to be
reminded. **These are required, not optional** (they override the research side's
"optional" in `docs/04_research_answers.md` answer 3).

| mode | how | status |
|---|---|---|
| MAXN | current | done / in progress with every task |
| `MODE_50W` (id 3) | same TPC mask as MAXN → try `sudo nvpmodel -m 3` with **no** reboot, as the last item of a MAXN session | agent can do unattended |
| **`MODE_30W` (id 2)** | **needs a human reboot** — `sudo nvpmodel -m 2`, confirm, then `sudo jetson_clocks` | **PENDING — ask the owner** |
| **`MODE_15W` (id 1)** | **needs a human reboot** — same procedure | **PENDING — ask the owner** |

The agent never confirms a reboot prompt and never uses `nvpmodel --force` (`AGENTS.md`).
After each reboot: `sudo jetson_clocks && sudo jetson_clocks --fan`, then re-run the E7
power points. Write "next: resume E7 at mode <id>" here before any switch (protocol §1).

## Headline after T010 (E11): the hybrid model's six [ASSUMPTION] terms, measured

`reports/E11_pi0_hybrid_inputs.md` · nine result directories · 1,564 rows in
`results/20260922b_E11_pi0_inputs/model_inputs.csv`, 33 of the 35 quantities §13.8 names.

| term | assumed | measured | |
|---|---|---|---|
| kernels per host segment | 1 (2 for attention) | **1–10.1**; ~35 launches per layer | 10× too few |
| `T_host_layer` | 0.16 ms | **0.1827 ms** as one CUDA graph per layer | 1.14× — reachable |
| | | 0.4455 ms compiled sequence | 2.78× |
| | | 1.5290 ms eager sequence | 9.56× |
| transfer fixed cost | **0** | **38–40 µs** pinned, either direction | not zero |
| transfer bandwidth | 143 GB/s | **36 GB/s** pinned | 4× optimistic |
| synchronisation | **free** | 0.8 µs query … **68.7 µs** stream hand-off | not free |
| `kv_copied_bytes_per_step` | **0** | **15,040,512 B every step**, 5.5824 ms | not zero |

**Only one implementation reaches the mapping's 0.16 ms: a single CUDA graph over the whole
layer.** Six separate graphs cost 1.6×, a compiled sequence 2.8×, eager 9.6×. Fusion across
segment boundaries is real and worth 2.5×, but it is a graph that closes the gap, not a compiler.

**The GPU is idle 75.6 % of the action-expert region** at both N = 4 and N = 10 — corrected for
nsys inflating the region's wall time ×1.53 while leaving kernel durations alone. The uncorrected
figure is 84 % and charges the profiler's overhead to the model. In CPU terms: the expert loop
pins **one core at 100 %** while eleven of twelve do nothing.

**Overlap belongs in a separate process.** A prefix loop beside the host-segment sequence costs
1.2 % as two processes and **10.4 ×** on the executing part as two CUDA streams in one process.
The contention is for the GIL, not the GPU.

**The prefix is a constant for this checkpoint.** `P_processed = 816` at every point of the sweep
while `P_valid` moves 262 → 807; every region is flat to within 0.7 %. This is a property of the
input format — three always-encoded image slots and a prompt padded to 48 — not of the hardware,
and **it is not expected to transfer to π0.5**, whose `tokenizer_max_length` is 200 and
`empty_cameras` 0.

### ⚠ Three defects in T010 were visible only in the output, never as an error

Recorded because the pattern matters more than the individual bugs. A compiled callable whose
results are discarded is dead-code-eliminated — measured here at 0.0662 ms vs 0.6605 ms for the
same matmul — and the published "compiled sequence = 0.085 ms" measured nothing. The background
hog's duty cycle never applied, so half a table was one condition four times. Fixing that
introduced a closure rebinding that killed every `cpu_spin_control` worker, whose six tracebacks
went to `DEVNULL` with the hog's stdout. Each produced a plausible number and no warning.

Countermeasures now in place: `bench/bw_hog.py --self-test` (every hog kind for 2 s, fails unless
each emits samples with an achieved duty; `mempressure.py` calls it before loading the model), and
`bench/check_report_numbers.py --task=T010`, which re-derives 96 figures from the CSVs and fails if
any is absent from the report.

## Headline after T005 (E1/E2): the expert's GEMMs are already at the speed of memory

**The gap is not weight streaming.** E2 put the action expert's 126 projection GEMMs — 622.85 MB,
97.6 % of the 638 MB that T002's `ae_step` floor was built on — into one CUDA graph, so the
Python interpreter is out of the timed region:

| | ms | × floor |
|---|---|---|
| analytic floor (622.85 MB ÷ **measured** 187.47 GB/s) | 3.322 | 1.00× |
| **the 126 projections, replayed as one CUDA graph** | **4.411** | **1.33×** |
| the same, issued back to back from Python | 8.656 | 2.61× |
| a whole eager denoising step (T002/E3), floor 3.403 ms | 52.56 | 15.4× |

The three MLP projections hold 73 % of the weight bytes and run at **87–90 % of measured peak
bandwidth**. The sensitivity runs the safe way: at the bottom of the measured bandwidth range the
stack multiple falls to 1.13×, so the conclusion strengthens. A memory system that removed *all*
off-chip weight traffic for the projections would save at most 4.4 ms of a 52.6 ms step.

**A denoising step launches 969 GPU kernels, and only 17 % of them are GEMMs** (slope over
N ∈ {2, 10}; the fit predicts N = 4 exactly). 48 % elementwise, 16 % copies, 8 % KV-cache
concatenation, 8 % normalisation. **At S = 51 the marginal cost of a projection is ~62 µs
whatever its shape**, against 8–55 µs of GPU work, so every projection is launch-bound there.
(That column is `max(CPU, GPU)`; above S ≈ 128 it tracks the GPU instead and is no longer a
dispatch cost.) `PiGemmaRMSNorm` costs **275 µs marginal against 17.7 µs of GPU**, 36 times a
step.

**Measured peaks (protocol §8), replacing the spec sheet everywhere:**

| quantity | spec | measured | ratio |
|---|---|---|---|
| memory bandwidth | 204.8 GB/s | **187.47 GB/s** | 0.915 |
| bf16 dense GEMM | 42 TFLOP/s | **33.85 TFLOP/s** | 0.806 |
| ridge point | ≈ 205–208 FLOP/B | **180.6 FLOP/B** | 0.87 |

**E11's analytic host model is 2× pessimistic on compute**: `eff_flops_frac` 0.40 assumed vs
**0.806** measured, `eff_bw_frac` 0.70 vs **0.915**. **`bench/fill_measured_floors.py` filled the
90 `peak_source = measured` rows** every `floor.csv` had left blank since T002.

**Two research-side questions answered.** (1) The L2 sysmem-miss substitute for DRAM traffic is
exact for **reads** (1.000–1.009 on four kernel configurations with analytically known traffic)
but the write path is **identically zero** in every `*_lookup_miss` counter — writes register as
L2 *hits*. Off-chip traffic must be `op_read_lookup_miss.sum × 32 B` **+** `op_write.sum × 32 B`;
using the miss counter for both, as protocol §3.3 has it, reports zero write traffic. Also:
§3.3(3) forbids deriving GB/s from ncu's `Duration`, so those columns are deliberately blank
until an nsys run supplies a duration. (2) The four INA3221 rails do **not** double-count and may
be summed: a load that raises `VDD_GPU_SOC` by 51.8 W raises `VIN_SYS_5V0` by only 8.7 W.

### ⚠ Unexplained: bandwidth varies run to run by up to 1.19×

22 fresh processes running the identical kernel span 159.9–188.6 GB/s, and whole-E1 passes gave
peaks of 161.8, 183.4, 185.6, 187.5 and 189.1 GB/s — with the GPU clock (1300.5 MHz), the EMC
clock (3199 MHz, sampled *during* the timed loop) and the die temperature (47–51 °C) constant
across the range, and identical device pointers. Sustained load does not throttle it either
(12 repeats of an 8192² GEMM: clock stable, no decay, die to 58.4 °C). Five checks are backed by
committed files in `results/20260922_E1_peak/raw/`; six more were run interactively and not
retained, and are reported as reasoning rather than evidence. **Consequence: a bandwidth-derived
floor carries roughly ±8 %, and a measured-vs-floor ratio should use a peak from the same
session.** Re-check after the power-mode reboots.

Report: `reports/E1_E2_microbench.md`.

## Headline after T003 (E4): compilation closes two thirds of the gap, and one line closes half of what is left

**The ladder is complete** (ms per denoising step, π0, MAXN):

| rung | ms/step | × floor |
|---|---|---|
| analytic bandwidth floor | 3.115 | 1.0× |
| research-side "optimistic bound" (2 × floor), *assumed unreachable* | 6.230 | 2.0× |
| measured eager (T002) | 52.56 | 16.9× |
| **measured compiled, best stock mode (`reduce-overhead`)** | **16.30** | **5.2×** |
| *modified: crop instead of cloning the KV cache* | *10.37* | *3.3×* |

The research side's ~90 ms/step estimate is **5.5× too pessimistic** — it applied Jetson
**Thor's** 1.51× compile gain to a derived 136 ms eager step, and both inputs were wrong here.
Measured compile gain is **3.22× per step** (1.39× end-to-end at N = 4, 1.69× at N = 10): a
single end-to-end factor is not transferable across N, only the slope is.

**What each mechanism contributes.** Compilation 2.32×, CUDA graphs a further 1.39×,
autotuning **nothing measurable**. `max-autotune` costs 5× the compile time of `default` for
latency it only gets from CUDA graphs — use `reduce-overhead`.

**Triton GEMM templates win no action-expert shape.** With Inductor's `min_sms = 68` gate
defeated in-process, Triton won **0 of 126** action-expert autotune blocks and 54 of 697
vision/prefix blocks; on four of five expert shapes the best Triton candidate is **~2× slower**
than cuBLAS. The committed `output_code` shows why: the expert graph is **127
`extern_kernels.mm`** — every matmul to cuBLAS — with Triton only for fused RoPE, GeGLU,
RMSNorm and casts. **The SM-count restriction is not why the expert sits above its floor.**

**One line still dominates.** `copy.deepcopy(past_key_values)` at `modeling_pi0.py:919` is
every graph break in the model (33 of 33 break events across all logs). It **cannot be
deleted** — the negative control fails, because the expert appends its suffix K/V regardless
of `use_cache=False` — but `DynamicCache.crop(P)` is an exact substitute: **bitwise identical
actions**, zero graph breaks, 10.37 ms/step. That reaches **1.67× the "optimistic bound"**
that `docs/00_context.md` calls a stand-in for a perfect software stack. **The headroom a PIM
design must beat is much smaller than the original ladder implied.**

Full report: `reports/E4_pi0_compile.md`.

### Errors caught in T003, for the record

Two I caught myself before they reached the report: my own region instrumentation
graph-breaks a compiled model (so E4's primary numbers are uninstrumented, per protocol §2),
and the `forced` variant silently reused cached kernels until both arms were re-run with cold
caches. Three the independent review caught: **all 20 result directories had empty `raw/` and
`raw_files: []`** — the same defect T002's review found, reintroduced verbatim in new code, so
every log-derived number was unverifiable from the repository; CUDA-graph use was *inferred*
from a timing gap rather than measured (now confirmed directly); and several counts were
wrong (autotune blocks double-counted 494→247, 1646→823) including a quoted log line that
existed in no file. Two protocol violations disclosed: a result directory overwritten in
place (§5), and `throttled: false` asserted with no tegrastats log (§1).

## Headline after T002 (E3): the action expert is **not** the dominant phase

**π0, LeRobot implementation, eager, MAXN: one denoising step costs 52.94 ms** (slope fit over
N ∈ {1,2,4,6,8,10}, r² = 0.99999). The research ladder assumes 136 ms — the real device is
**2.6× faster** than assumed, and **17.0× the 3.115 ms bandwidth floor**.

The phase split contradicts the working hypothesis. At the paper's N = 4 the action expert is
the *smallest* of the three phases, and `vision` is the largest:

| phase | paper (Orin, N=4) | measured here, N=4 | measured, N=10 |
|---|---|---|---|
| `vision` | 110 ms (12 %) | **239.4 ms (34.5 %)** | 239.6 ms (23.7 %) |
| `vlm_prefix` | 258 ms (28 %) | 233.1 ms (33.6 %) | 232.9 ms (23.0 %) |
| `ae_total` | 543 ms (59 %) | **215.5 ms (31.1 %)** | 533.0 ms (52.7 %) |
| end-to-end | 920 ms | **694.1 ms** | 1011.6 ms |

The expert *is* memory-bound as predicted (floor 3.115 ms memory vs 0.912 ms compute) — it
just is not the biggest phase. Full report: `reports/E3_pi0_eager.md`.

Three `docs/00_context.md` values are now **confirmed from the loaded model**, not assumed:
**P = 816** (3 × 256 + 48; the doc marked it assumed), **S = 51**, **623.01 MB** of expert
weights per step, and 38.29 GFLOP against the doc's 38.7.

## Three findings

* **80.2 ms per inference encodes a dummy image.** `empty_cameras: 1` makes the policy
  synthesise a third camera filled with −1 and push it through the full SigLIP tower:
  **33.3 % of `vision`, 11.6 % of an N = 4 inference**, and its attention mask is zero so the
  result is discarded. Caching one constant embedding removes it.
* **10.4 % of every denoising step is a Python deep copy of the KV cache.**
  `modeling_pi0.py:919` runs `copy.deepcopy(past_key_values)` inside `denoise_step`, cloning
  all 36 KV tensors (**15.04 MB**) every step: **5.68 ms/step, 56.8 ms and 150 MB per
  inference at N = 10**. At an effective 2.65 GB/s it is Python/allocator cost, not bandwidth
  — and it alone is **1.82× the entire 3.115 ms memory floor of a step**.
  `docs/00_context.md` assumes `kv_copied_bytes_per_step = 0`. The expert branch passes
  `use_cache=False`, so whether the copy is needed at all is one experiment.
* **Denoising step 0 costs +0.83 ms** over the rest — *not* the +3.13 ms first reported. The
  larger figure was ~74 % my own instrumentation: `ae_layer[j]` events land on one step and
  cost it ~2.57 ms. Control: `results/20260921_E3_pi0_eager_probe/step0_control.csv`.

## ⚠ A trap that would have silently produced garbage

PyPI `lerobot==0.4.4` matches the checkpoint's `_v044` name and would have loaded a
**randomly initialised model**: it needs a patched transformers fork, it lacks the
`lm_head → embed_tokens` alias fix this checkpoint requires (the weight exists only as a
safetensors metadata alias, which `load_file` does not reconstruct), and
`PI0Policy.from_pretrained` **catches load failures and returns the model anyway** with only a
printed warning. A random π0 still runs and still yields plausible latencies.
Pinned `lerobot @ f0d2b37beb` instead, and `bench/pi0_common.py:assert_real_weights()` now
refuses to measure unless 400 sampled tensors match the checkpoint file.
`transformers` is pinned to **5.3.0 exactly** — 5.4.0 moved a `× √hidden_size` rescale that
this ref still applies itself, which would scale every prefix embedding by ≈ 45, silently.

## Route change: LeRobot's π0, not openpi's (owner decision)

`tasks/T002_pi0_eager.md` prescribed openpi's PyTorch port. It was researched in full and
rejected on setup cost — 12 GB orbax/OCDBT conversion, a mandatory JAX import, a
`transformers == "4.53.2"` string-equality check with five files copied over site-packages,
and a `requires-python >=3.11` gate — none of which measures anything. LeRobot's port is the
same architecture (the model prints *"a direct port of the OpenPI implementation"*), with a
ready-made PyTorch checkpoint. **The weights are a different fine-tune**, so the paper
comparison is across implementations *and* checkpoints; that is stated in the report.
The openpi route stays open and is the obvious way to test whether the phase-split
disagreement is implementation or measurement. `docs/00_context.md` and the task file were
updated to match.

## Device is at MAXN — measurement can proceed

The human rebooted into **MAXN (id 0)** on 2026-09-21 and it is persisted
(`/var/lib/nvpmodel/status` = `pmode:0000`). `jetson_clocks` + `--fan` are applied and are
now run **before every capture**, per protocol §1. CUDA reports **16 SMs**, `TPC_PG_MASK` 0,
GPU locked at **1 300.5 MHz** (confirmed under load), 12 CPU cores, EMC 3 199 MHz.

The authoritative E0 record is **`results/20260921_E0_platform_maxn/`**.
`results/20260921_E0_platform/` is the earlier MODE_30W run, kept only for comparison; **no
number in it characterises the Orin.**

**Three things do not survive a reboot** and must be re-applied at the start of every session:
```bash
sudo jetson_clocks && sudo jetson_clocks --fan
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'   # ncu counters
```
plus the sudo password itself — **passwordless `sudo` is still not configured**
(`AGENTS.md` lists it as a prerequisite). The user is in group `sudo` but a password is
required; it was supplied interactively and used through a `SUDO_ASKPASS` helper kept
outside the repository, which `/tmp` clearing at reboot removes. A `NOPASSWD` sudoers entry
for `nvpmodel`, `jetson_clocks`, `tegrastats`, `ncu` and reading `/sys/kernel/debug` would
remove this friction.

Note for E7's power-mode sweep: **every mode change that alters TPC gating needs a reboot**,
and declining the prompt does not stage it (`nvpmodel` reports `optMask is 1, no request for
power mode` and leaves the persisted mode untouched). Answer `YES`, or use `--force`.

## Device

| | |
|---|---|
| board | NVIDIA Jetson AGX Orin Developer Kit (`tegra234`), host `cocojet` |
| L4T / JetPack | **R36.5.0** / **6.2.3+b81** — JetPack 6 prerequisite **met**, no reflash needed |
| CUDA / cuDNN / TensorRT | 12.6.68 / 9.3.0.75 / 10.3.0.30 |
| kernel / python | 5.15.185-tegra (`oot`) / 3.10.12 |
| GPU | Orin, cc 8.7 → **`sm_87`** (confirmed by torch *and* `cuobjdump` on the system cuBLAS), 48 max warps/SM |
| **SMs** | **16** at MAXN (`TPC_PG_MASK` 0, 8 of 8 TPCs); was 8 in MODE_30W |
| GPU clock | **1 300.5 MHz** locked and confirmed under load; was 612 MHz in MODE_30W |
| EMC | **3 199 MHz** = LPDDR5-6400, as expected |
| memory / disk | 61 GiB RAM, 8 × 3.8 G zram (no disk swap); NVMe 1.8 T, 1.7 T free |
| idle power (MAXN, clocks locked, 58.6 s @ 10 Hz) | **module power 11 012 mW median** (the four rails of protocol §4, from sysfs); tj 41.9 °C; not throttled. MODE_30W gave 8 522 mW / 39.9 °C. |

## Software stack that works

**torch 2.8.0 + torchvision 0.23.0 + triton 3.4.0**, JetPack CUDA 12.6 wheels (cp310
aarch64) from `https://pypi.jetson-ai-lab.io/jp6/cu126`, in `~/venvs/jetson_base`.
`torch.cuda.is_available()` true, `arch_list = ['sm_87']`, bf16 GEMM correct to 4.2e-3 vs
fp32, int8 `torch._int_mm` works.
**`torch.compile` works in every mode**: inductor `default`, `reduce-overhead`,
`max-autotune`, `max-autotune-no-cudagraphs`, plus `aot_eager` and `eager`
(`results/20260921_E0_platform_maxn/stack_verification.json`).
**But `max-autotune` never uses Triton GEMM templates on an Orin** — see point 5 below.
Trap to remember: `pip install torch==2.8.0` with `--extra-index-url https://pypi.org/simple`
installs PyPI's **CPU-only** `2.8.0+cpu`, because pip prefers a manylinux wheel over the
index's `linux_aarch64` one. Install the three CUDA wheels by full URL with `--no-deps`
(`env/setup.sh`).

Profilers: **`ncu` 2024.3.1.0**, **`nsys` 2024.5.4.34**, `cuobjdump`/`nvdisasm` 12.6.68 —
all shipped with JetPack, nothing to install. `--set` values offered: `basic`, `detailed`,
**`full`**, `nvlink`, `pmsampling`, `roofline` (protocol expected ≥ basic/roofline/full ✓).

## Which ncu metrics this Orin exposes

Full verdicts: `results/20260921_E0_platform_maxn/ncu/metric_presence.csv` — **52 present,
14 present-but-zero, 9 absent**; every exported column: `ncu/columns.txt` (1 117 raw
columns). `present_zero` is the "empty" case T001 asks about, made precise: the column
exists and every value is exactly 0.

**Present with non-zero values:** all protocol §3.3 occupancy metrics (`SM Busy`,
`SM Active Cycles`, `Achieved Occupancy`, `Theoretical Occupancy`,
`Achieved Active Warps Per SM`, `sm__warps_active.avg.pct_of_peak_sustained_active`);
`Duration`, `Compute (SM) Throughput`, `Memory Throughput`, `Mem Busy`, `L2 Hit Rate`,
both `Executed Ipc`; the LSU global/shared counters this kernel exercises; all launch /
occupancy-limit metrics; pipe utilisation for `lsu` (21.6 %), `alu` (13.5 %), `adu`,
`cbu`, `fma`; **15 of the 18 warp-stall reasons**.

**Present but zero on the toy kernel (14):** mostly legitimate for this kernel — the local
and atomic LSU counters, shared-atom wavefronts, the `xu` pipe, `L1/TEX Hit Rate`, and the
`membar` / `sleeping` / `tex_throttle` stalls. **Three are not legitimate: every
`dram__*`.** A zero on one kernel is not evidence a counter works; re-check per kernel.

**Absent (9):** `DRAM Throughput`, `DRAM Utilization`, `dram__bytes{,_read,_write}.sum`,
`sm__inst_executed_pipe_fp16...active`, `sm__inst_executed_pipe_{fma,alu}...elapsed`,
`gpc__cycles_active.avg.per_warp`.

**Five things later tasks must do differently:**
1. **No DRAM counters — established three ways.** `ncu --query-metrics` lists **4 426
   metrics on `Orin (GA10B)` and none is `dram__*`** (`env/tool_help/ncu_query_metrics.txt`);
   an explicit `--metrics dram__bytes.sum,...` returns **`n/a`** (`raw/ncu_dram_probe.log`);
   and the `--set full` placeholder columns all read 0. ncu's `Memory Throughput` here
   equals `lts__throughput` to two decimal places — it is **L2**, not DRAM. Off-chip *read*
   bytes must come from `lts__t_sectors_*_sysmem_*_lookup_miss.sum × 32 B`; on the toy
   kernel at MAXN that gave **8.4931 MB against 8.4931 MB** of analytically compulsory
   reads — **ratio 1.0000** (the MODE_30W run gave 1.0032). The **write** path is still not
   validated: the kernel's 0.42 MB output never appears as a sysmem miss.
2. **Stall reasons are on the `raw` page only**, as
   `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`. The `details` page has
   a `Warp State Statistics` section but no `Stall …` rows.
3. **`ncu` needs root here.** `perf_event_paranoid=1` (protocol §12) is not sufficient;
   use `sudo $(which ncu) ...` (plain `sudo ncu` fails — `secure_path`), then `chown` the
   `.ncu-rep` before exporting.
4. **Check ncu's clock against the real one every time.** `--clock-control base` pins the
   SM clock. At MAXN with `jetson_clocks` applied the two agree (ncu 1.29 GHz vs 1 300.5 MHz
   measured, `Duration` 57.09 µs vs nsys 54.6 µs, +4.5 %). In the MODE_30W run, taken with
   unlocked clocks, they did not (305.70 MHz vs 612 MHz). ncu `Duration` is still never a
   latency (protocol §3.2) — but lock the clocks first and the gap is small.
5. **`max-autotune` cannot use Triton GEMM templates on any Jetson Orin.**
   `torch/_inductor/utils.py` hard-codes `min_sms = 68` (`# 3080`) for CUDA; Orin has 16, so
   the check fails **in every power mode** — this was initially mis-attributed to MODE_30W
   and is disproved by the MAXN run (`raw/verify_stack.stdout.log` line 3). `max-autotune`
   falls back to ATen/cuBLAS; its autotune log compares only `addmm` vs `bias_addmm`.
   **E4 must present "max-autotune" here as "default + CUDA graphs + ATen autotuning".**

Also: **`tegrastats` does not print `VDDQ_VDD2_1V8AO`** on this carrier board — the memory
rail is only in sysfs at `/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon4/in2_*`.
`bench/collect_platform.py` now samples all four rails there at 10 Hz
(`power_idle_sysfs.csv`), so protocol §4 module power is available. That channel reads a
**constant 5 040 mV bus**, not 1.8 V — see question 2. `tegrastats` also **appends** to
`--logfile`; delete the file first. And nsys's `--capture-range=cudaProfilerApi` kills the
target at `cudaProfilerStop` unless `--capture-range-end=stop` is passed.

## Done

* **T005 (E1, E2) — microbenchmarks, substitute validation, rail check.** Result directories
  `results/20260922_E1_peak/`, `results/20260922_E2_ae_kernels/`,
  `results/20260922_E2_kernel_counts{,_N2,_N4,_N10}/`, `results/20260922_E7_rails_check/`
  (the `20260921_*` copies are kept, each with a `SUPERSEDED.md`) ·
  Report: `reports/E1_E2_microbench.md` ·
  Harness: `bench/micro_peak.py`, `micro_ae_kernels.py`, `ncu_validation.py`,
  `run_ncu_validation.sh`, `parse_ncu_validation.py`, `power_rails_check.py`,
  `refresh_raw_files.py`, `sustained_load_probe.py`, `bandwidth_mode_probe.py`,
  `count_step_kernels.py`, `analyze_step_kernels.py`, `fill_measured_floors.py`, plus
  `collect_raw()` / `scan_for_secrets()` / `classify_kind()` in `common.py`.
  `hw_constants.json` is complete (protocol §8). Corrected `docs/01_measurement_plan.md`'s
  E2 table as owner (k and v are two separate 1024→256 projections, not one 1024→512).

* **T002 (E3) — π0 end-to-end latency, eager.** Six result directories
  `results/20260921_E3_pi0_eager_N{1,2,4,6,8,10}/` · Report: `reports/E3_pi0_eager.md` ·
  Environment: `env/setup_pi0.sh`, `env/requirements_pi0.txt`, `env/README.md` §10 ·
  Fixture: `bench/fixtures/` (HuggingFaceVLA/libero ep 0 frame 0) ·
  Harness: `bench/pi0_common.py`, `regions.py`, `run_e3.py`, `analyze_e3.py`,
  `fetch_fixture.py`, `measure_preprocess.py`.

* **T001 (E0) — bring-up and platform record. COMPLETE, all four steps.** Step 1 (MAXN +
  `jetson_clocks`) was satisfied after the human authorised the reboot on 2026-09-21.
  Result: **`results/20260921_E0_platform_maxn/`** (authoritative), with
  `results/20260921_E0_platform/` kept as the MODE_30W comparison ·
  Report: `reports/E0_platform.md` ·
  Environment: `env/README.md`, `env/setup.sh`, `env/requirements_base.txt`,
  `env/tool_help/` (12 files) · Harness: `bench/common.py`, `verify_stack.py`,
  `ncu_toy.py`, `ncu_columns.py`, `hw_constants.py`, `collect_platform.py`,
  `gpu_clock_probe.py`, `finalize_raw.py`, `make_meta_e0.py`.
  An independent sub-agent pass (protocol §11) reviewed the run; it found one fabricated
  process claim in the report and nine smaller defects, all fixed and listed in
  `reports/E0_platform.md` §9.

### Carried into the MAXN re-run (accepted, not yet done)
* Apply `sudo jetson_clocks` **before** every capture, not after, and record
  `devfreq cur_freq` alongside each one (`bench/gpu_clock_probe.py`). In this run the two
  toy profiler captures were taken with `schedutil` free to scale 306–612 MHz, and that
  clock was not recorded.
* ~~Re-validate the L2 sysmem-miss substitute for off-chip **write** traffic, and against the
  measured bandwidth peak from E1.~~ **DONE in T005.** Reads validate to 1.000–1.009 on four
  kernels with analytically known traffic. Writes are **identically zero** in every
  `*_lookup_miss` counter — a write sector is recorded as an L2 *hit* — and are recovered
  exactly by `lts__t_sectors_aperture_sysmem_op_write.sum × 32 B` instead. See
  `results/20260922_E2_ae_kernels/ncu/sysmem_miss_validation.csv`.
* Confirm 16 SMs and re-check `max-autotune` (it is refused at 8 SMs).

## In progress

(none)

## Blocked

* (nothing) — T001's MAXN blocker was resolved by the human's reboot and T002 is done.

Historical, for the record: T002 was blocked until the reboot, because it would have run at half the
  SMs and half the GPU clock, so it should wait for the reboot.

## Proposed task Q1 — which models are action-head bottlenecked? (owner request, 2026-09-22)

**Queued, not started.** Runs after π0's phase characterisation and prefix optimisation.

**Why.** The study assumes the action expert is the memory-bound phase worth moving into
memory. Measured on π0, that holds only before optimisation. The three phases start almost
balanced — **33.9 / 32.9 / 32.3 %** eager fp32 — and optimisation *unbalances* them:

| region | eager fp32 | optimised | speedup | share before | share after |
|---|---:|---:|---:|---:|---:|
| vision | 241.31 | 99.92 | 2.42× | 33.9 % | 27.0 % |
| `vlm_prefix` | 234.06 | 192.78 | **1.21×** | 32.9 % | **52.2 %** |
| `ae_total` | 230.12 | 69.72 | **3.30×** | 32.3 % | **18.9 %** |

Each factor is surgical — it moves one region and leaves the others within ±2 %: TF32 is
**2.32× on vision alone**; compiling `denoise_step` is **2.31× on the expert alone**, and
CUDA graphs a further **1.40×**; compiling `pg.forward` is **1.22× on `vlm_prefix` alone**.
CUDA graphs fail outright on both `embed_image` and `pg.forward` (static-buffer overwrite).

arXiv:2604.24447 reports the expert at 59 % of π0 on an AGX Orin and builds two
optimisations (DP-Cache, V-AEFusion) on that. Their configuration is **uncompiled** —
inferred from their own tables and confirmed against ours phase by phase within 11 % — and
they never re-measure the split after the compilation their own Table 3 applies. So the
premise is a property of the baseline, not of deployed inference.

**The question.** Is there a model for which the premise holds under an *identical*
treatment — identically optimised, or identically un-optimised? The controlling ratio is
`N × cost_per_step / (vision + prefix + N × cost_per_step)`, so the prediction is that
dominance tracks the denoising-step count far more than architecture: Diffusion Policy
(N = 100, no VLM) ≫ π0 / π0.5 / SmolVLA (N = 10) ≫ GR00T (N = 4) ≫ ACT (no diffusion).
If that is what we find, the premise is a statement about iteration count rather than about
VLAs, and the study should either target high-N diffusion policies or target the VLM prefix.

**Method.** The identical two-part iteration π0 received, per model: the paper-configuration
run (`bench/paper_repro.py`), per-phase SM utilisation (`bench/phase_utilisation.py`),
analytic operational intensity against our measured ridge of 180.57 FLOP/B, then the same
one-factor-at-a-time ladder (`bench/opt_ladder.py`). One row per model: action-head share
eager, action-head share optimised, and the per-factor speedup per phase.

**Blocked on T006's model registry.** `bench/` is π0-specific in six ways, and the three
wrap points the ladder needs (`embed_image`, `pg.forward`, `denoise_step`) differ per model
— `regions.py`'s `ae_layer[j]` records nothing at all for SmolVLA, whose layer body is
inlined in `SmolVLMWithExpertModel.forward` (verified). Comparison targets for AGX Orin are
saved from the authors' leaderboard in `scratch/T004/vla_leaderboard_20260922.json`:
ACT 23.93, GR00T 256.22, SmolVLA 777.38, π0 920.58, OpenVLA 951.56, π0.5 2994.02,
Diffusion Policy 4379.08 ms.

**One thing to confirm:** the owner named GR00T **N1.5 / N1.6**. Isaac-GR00T is installed
and verified at **N1.7** (`376ba890`, `env/setup_groot.sh`). Whether N1.5/N1.6 checkpoints
and configs are obtainable at that commit is unchecked; if they are not, that is reported
rather than substituted.

## Questions for the research side

1. **DRAM traffic.** `docs/02_protocol.md` §3.3 and E6 assume ncu's `DRAM Throughput` /
   `DRAM Utilization`. They do not exist on Orin (every `dram__*` counter reads 0). We
   propose `lts__t_sectors_aperture_sysmem_*.sum × 32 B` as the measured off-chip byte count
   (validated to 0.4 % on a kernel with known traffic) and will label it
   *measured (L2 sysmem-miss sectors)*. Please confirm that is acceptable for the roofline
   points of E6, and note that ncu's `Memory Throughput` on this device must be read as
   **L2** throughput wherever it appears.
   **T005 update — the substitute is validated, but not in the form the protocol states it.**
   On four kernels with analytically known traffic, reads recover to **1.000–1.009** via
   `lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum × 32 B`. The **write** path is
   **identically zero** in every `*_lookup_miss` counter, including on a kernel that
   demonstrably writes 256 MiB: a write sector is recorded as an L2 lookup *hit*, because the
   L2 is write-back / write-allocate, so it can never appear in a miss counter. Writes are
   recovered exactly (1.001–1.008) by `lts__t_sectors_aperture_sysmem_op_write.sum × 32 B`.
   **So off-chip traffic = `op_read_lookup_miss.sum × 32 B` + `op_write.sum × 32 B`.** Using
   the miss counter for both, as §3.3 has it, would report every kernel's write traffic as
   zero. Evidence: `results/20260922_E2_ae_kernels/ncu/sysmem_miss_validation.csv`. Note also that protocol §3.3(3) forbids deriving achieved GB/s from ncu's `Duration`; those columns are left blank until an nsys run of the same kernel supplies a duration.
2. **Memory-rail power.** `VDDQ_VDD2_1V8AO` exists only in sysfs, and the bus voltage its
   INA3221 channel reports is ≈ 5 V, not 1.8 V. Before E7 quotes memory power we will have
   to establish the correct scaling; if the research side knows the intended interpretation
   of this rail on the AGX Orin devkit carrier, that would save a calibration step.
   **T005 update.** The rail behaves sensibly whatever its scaling: it rises +5.0 W on a pure
   DRAM-copy load and only +5.5 W on a tensor-core load, i.e. it tracks memory traffic rather
   than compute, which is what the name implies. The four rails were also checked for
   double-counting and are independent — a load that raises `VDD_GPU_SOC` by 49.9 W raises
   `VIN_SYS_5V0` by only 8.7 W — so E7 may sum them. What remains open is only the absolute
   scaling of this one rail. Evidence: `results/20260922_E7_rails_check/power_rails_check.csv`.
3. **Power-mode plan.** Since MAXN needs a reboot on this board, so does every return from a
   capped mode. E7's power-mode sweep therefore cannot be done inside one session: it will
   need one reboot per mode, each authorised by the human. Is that acceptable, or should E7
   be limited to MAXN plus whichever capped modes can be entered without a reboot?

## Repository size (protocol §6f)

After T010 (2026-09-22):

```
du -sh results/   106M
du -sh .git        25M
git count-objects -vH:  count: 1821   size-pack: 62.49 KiB

five largest committed files (protocol §6f):
   6,584,184  results/20260922b_E11f_pi0_pipelining/nsys/eager_N10_cuda_gpu_trace.csv
   6,584,072  results/20260922_E5_pi0_eager_N10/nsys/eager_N10_cuda_gpu_trace.csv
   3,301,941  results/20260922b_E11f_pi0_pipelining/nsys/eager_N4_cuda_gpu_trace.csv
   3,301,929  results/20260922_E5_pi0_eager_N4/nsys/eager_N4_cuda_gpu_trace.csv
   2,466,365  results/20260922_E4_pi0_max-autotune+forced_N4/raw/compile_and_stdout.log
```

`find results -size +50M` is empty; the largest committed file is 6.6 MB, well inside §6's limit.
T010 added ~84 MB: nine result directories plus the superseded passes §5 requires be kept, 216,000
per-iteration segment samples, 61 `torch.profiler` traces (~23× under zstd -19), per-core
`/proc/stat` samples and the nsys CSV exports. 453 raw-file entries were secrets-scanned before
commit with zero matches. At 106 MB of results the ~2 GB ceiling is not close.

After T005 (2026-09-22):

```
du -sh results/   22M
du -sh .git       9.1M
git count-objects -vH:  count: 797   size-pack: 62.49 KiB

five largest committed files (protocol §6f):
   2,466,365  results/20260922_E4_pi0_max-autotune+forced_N4/raw/compile_and_stdout.log
   2,466,365  results/20260922_E4_pi0_max-autotune+forced_N10/raw/compile_and_stdout.log
   1,577,638  results/20260921_E2_ae_kernels/raw/latency_samples.csv
   1,227,335  results/20260921_E2_kernel_counts_N10/raw/step_trace.json.zst
     794,544  env/tool_help/ncu_query_metrics.txt
```

`find results -size +50M` is empty, and no committed file exceeds 2.5 MB. T002's figures were
3.8 M / 3.6 M / 362 objects; T003's were 11 M / 5.0 M / 592.

T005 added ~5 MB: two full E1/E2 passes (the `20260921_*` set is kept with a `SUPERSEDED.md`
because the report cites the E1 runs as evidence of bandwidth variation), five ncu reports
compressed 84×, and four `torch.profiler` traces compressed ~23×. Every `.zst` records its
original's sha256, size and a secrets scan run **before** compression.

### Size budget calibration from the toy captures (T001 step 3c)

| file | capture | uncompressed | `zstd -19 -T0` | ratio |
|---|---|---|---|---|
| `nsys_toy.nsys-rep` | 3 inferences of one `F.linear`, 6 kernels | 108 402 B | 41 708 B | 2.6× |
| `nsys_toy.sqlite` | derived from the above | 536 576 B | 53 989 B | 9.9× |
| `ncu_toy.ncu-rep` | **1 kernel**, `--set full`, 30 replay passes | **7 072 411 B** | **92 396 B** | **76.5×** |
| `ncu_toy.ncu-rep` (MAXN) | same, 16 SMs | 7 107 467 B | 92 232 B | 77.1× |

Only the `.ncu-rep` was committed compressed (protocol §6b: > 1 MB); the other two are
under 1 MB and are committed as-is.

**What this implies for later tasks.** The ncu report is ≈ **7 MB per captured kernel**
uncompressed at `--set full`, and compresses ~76× because it is mostly sparse. A π0
`ae_step[0]/` capture is expected to hold several hundred kernels → **order 1–2 GB
uncompressed, order 20–30 MB as `zstd -19`** — committable, but only just, and the
uncompressed intermediate needs room in `scratch/`. Budget `-c` limits accordingly and
check before committing. `bench/finalize_raw.py` compresses above 1 MB and splits above
50 MB with a verified `REASSEMBLE.txt`.

The nsys side is cheap by comparison (≈ 18 KB per kernel uncompressed here, and dominated
by fixed overhead), so the 1–3 inference capture window of protocol §3.1 is not the
constraint; ncu is.
