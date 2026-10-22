# E0 toolcheck (T001 steps 1, 3, 4, 5) — 2026-10-22, MAXN (mode 0), smoke checks, not measurements

| check | result | file |
|---|---|---|
| PyTorch route (a) cu130 SBSA wheels: torch 2.10.0+cu130, torchvision 0.25.0, triton 3.6.0 | **pass** with `env/thor_env.sh` (Triton pointed at CUDA 13.0 `ptxas`); **fails** as installed: Triton's bundled `ptxas-blackwell` (CUDA 12.9) rejects `sm_110a` | `stack_verification.json`, `raw/stack_verification_route_a_bundled_ptxas_FAILED.json` |
| bf16 GEMM 4096³ vs fp64 reference | rel. err 2.9e-3 | `stack_verification.json` |
| `torch.compile` bf16 MLP, 4 modes | all run, max rel. err vs eager 4.4e-3; Inductor `is_big_gpu`: threshold 68 SMs, device 20 → False, "Not enough SMs" logged | `stack_verification.json` |
| nsys 2025.3.2 unprivileged | works; toy kernel `nvjet_sm110_tst_256x56_64x5_4x1_v_bz_TNT` | `nsys/nsys_toy_*.csv` |
| ncu 2025.3.0 unprivileged | `ERR_NVGPUCTRPERM`; works as root via `env/rootdo.sh` (docker) | `raw/ncu_toy.*` |
| DRAM bytes | no `dram__*`; `lts__d_sectors_fill_sysmem.sum` 266,233 and `lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum` 266,139 vs 265,408 analytic read sectors; `..._op_write.sum` 13,453 vs 13,056 analytic; `mcc__dram_throughput_*` present but 0 | `ncu/metric_presence.csv` |
| tensor FLOPs | `sm__ops_path_tensor_src_bf16_dst_fp32.sum` = 469,762,048 = 2·56·4096·1024 (M padded 51 → 56) | `ncu/metric_presence.csv` |
| clock control | clocks locked by `jetson_clocks` at 1575 MHz: ncu SM frequency 1.5731 GHz (`base`), 1.5735 GHz (`none`); unlocked, `base` gave 0.899 GHz → use `--clock-control none` with `jetson_clocks` | `raw/ncu_toy_clock_*` |
| ncu kernel limit | 1,000 of 1,000 kernels profiled in one process (no early stop) | `raw/ncu_limit_1000_tiny.*` |
| CUDA-graph replay | visible: 3 replays → 3 kernel nodes (`--graph-profiling node`, the default) | `raw/ncu_graph_node.*` |
| architecture | compute capability 11.0, `sm_110`; cuBLASLt carries 1,140 `sm_110` cubins | `raw/cublaslt_sm_strings.txt` |
| rails | VDD_GPU, VDD_CPU_SOC_MSS, VIN_SYS_5V0 (ina3221), VIN (ina238); idle sample not clean (shared GPU) | `raw/power_idle*.csv`, `raw/tegrastats_idle.log` |

Odd: the GPU is shared with other users' sessions; VDD_GPU swung 7–29 W with nothing of ours running.
