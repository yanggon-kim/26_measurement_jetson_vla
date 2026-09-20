# 02 — Measurement protocol (binding)

## 1. Device state — record for every run, fix before every run
- Power mode: `sudo nvpmodel -q`. Default for all experiments: **MAXN** (`sudo nvpmodel -m 0`) followed by
  `sudo jetson_clocks`. E7 repeats selected points at the 50 W, 30 W and 15 W modes.
- Save `sudo jetson_clocks --show`, `cat /etc/nv_tegra_release`, `dpkg -l | grep -E 'nvidia-jetpack|cuda-toolkit|tensorrt|cudnn'`,
  `nvcc --version`, `python -c "import torch;print(torch.__version__, torch.version.cuda)"`, kernel version, and the EMC
  (memory) clock: `sudo cat /sys/kernel/debug/bpmp/debug/clk/emc/rate` (expect 3,199 MHz = LPDDR5-6400).
- Thermal: log temperatures with tegrastats; if any run shows throttling or a clock below its locked value,
  mark it `throttled: true`, cool down, repeat. Use the fan at maximum (`jetson_clocks --fan`).
- Nothing else running: no desktop session if possible, no second benchmark, swap/zram state recorded.

## 2. Timing
- Batch size 1, bf16 weights and activations unless a task says otherwise, fixed seeds, fixed inputs
  (a real LIBERO observation saved once to `bench/fixtures/`, not random tensors, so token counts are real).
- GPU time with CUDA events (`torch.cuda.Event(enable_timing=True)`) around each region, synchronised only at
  region boundaries; also wall-clock end to end. Do not time with profilers attached: profile in separate runs.
- ≥ 20 warm-up iterations (for `torch.compile`: until no recompilation is reported, and log the compile time),
  then ≥ 100 timed iterations. Report **median, mean, p5, p95, min, n**. Keep the per-iteration samples.
- Regions (use the same names as NVTX ranges): `preprocess`, `vision`, `vlm_prefix` (prefill that builds the
  KV cache), `ae_total`, `ae_step[i]` for each denoising step, and inside one representative step
  `ae_layer[j]`; `postprocess`. If the code fuses regions so they cannot be separated, say which.
- Sweep the number of denoising steps by the model's own config knob, not by editing loops.

## 3. Profilers
- **Nsight Systems**: `nsys profile -t cuda,nvtx,osrt,cublas,cudnn --cuda-memory-usage=true -o raw/<run>/<name> --capture-range=cudaProfilerApi python bench/...`
  with `torch.cuda.profiler.start()/stop()` around ~10 steady-state inferences. Then export and commit the CSVs:
  `nsys stats -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum,cuda_kern_exec_sum -f csv -o results/<run>/nsys raw/<run>/<name>.nsys-rep`
  and the full kernel trace for one inference (`-r cuda_gpu_trace`). From the trace compute, per NVTX region:
  number of kernels, summed kernel time, **GPU idle time = region time − union of kernel intervals**, the
  distribution of gaps between consecutive kernels, top-20 kernels by time with their launch grids, memcpy time.
- **Nsight Compute** (`ncu`, the user's name for the Nsight Compute CLI): use it on one action-expert step to get
  per-kernel metrics. On Tegra many hardware counters are unavailable; try, record exactly what is and is not
  exposed, and fall back to **analytic FLOPs and bytes per kernel (from tensor shapes) ÷ measured kernel time**
  to get achieved FLOP/s and GB/s.
- `torch.profiler` with `record_shapes=True, with_flops=True` is a welcome cross-check; export
  `key_averages().table` as CSV.

## 4. Power and energy
- `sudo tegrastats --interval 100 --logfile raw/<run>/tegrastats.log` (10 Hz) or read the INA3221 rails directly
  from sysfs at ≥ 10 Hz if finer. Rails to keep: `VDD_GPU_SOC`, `VDD_CPU_CV`, `VIN_SYS_5V0`, `VDDQ_VDD2_1V8AO`
  (the memory rail — we care about it specifically). Convert to a tidy CSV (`t_s, rail, mW`).
- Measure 60 s of idle baseline in the same power mode before each power run.
- Because one inference is shorter than the sampling period, measure power on a **loop of back-to-back
  inferences ≥ 60 s**, and separately on loops that run only the prefix and only the action expert, so the
  phases can be attributed. Energy per inference = mean power × mean latency; report both total and
  above-idle. State the sampling rate and that rail readings are averaged by the sensor.

## 5. Result directory — one per run
`results/<YYYYMMDD>_<Exx>_<model>_<mode>_<short>/` containing:
- `meta.json`: `experiment, task, date_utc, model, model_repo, model_commit, checkpoint, precision, compile_mode,
  num_denoise_steps, action_horizon, prefix_tokens, suffix_tokens, batch, n_warmup, n_timed, power_mode,
  jetson_clocks, emc_hz, gpu_hz, jetpack, l4t, cuda, torch, torchvision, triton, bench_commit, command,
  throttled, raw_files:[{path, sha256, bytes}], notes`.
- tidy CSVs, one row per observation, long format: `latency_samples.csv (region, iter, ms)`,
  `latency_summary.csv (region, median_ms, mean_ms, p5_ms, p95_ms, min_ms, n)`, plus `nsys/*.csv`, `power.csv`,
  `kernels_per_region.csv` when applicable.
- `summary.md`: five to fifteen lines — what was run, the headline table, anything odd.
Never overwrite a result directory; a re-run is a new directory and the old one is referenced.

## 6. Git hygiene
`git pull --rebase` before every push; small commits with a message that names the experiment and task; never
force-push; never rewrite pushed history. Files over 5 MB do not belong in git — summarise them.
