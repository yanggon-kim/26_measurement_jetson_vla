# E13c — can the GPU be confined to a subset of LPDDR5 channels on this Orin? (T016 step 1c, row 4)

**Answer (row 4): no — not on stock Orin without a boot or hardware change. Whether a boot-configuration
route exists is unknown** (not attempted: the task forbids boot / device-tree changes).

## FACIL (Seo et al., HPCA 2025) — A3 (a)–(c), from the paper
(https://seonghoonseo.github.io/files/pdf/research/2025-hpca-facil.pdf)
* **(a) The per-allocation map is a PROPOSED change, not a stock feature** — confirming the research side's reading.
  §V: "Implementing FACIL requires modifications to the memory controller in addition to the introduction of a
  mapping selector"; the MapID lives in unused bits of a 2 MB huge-page PTE, `mmap()` is extended to take it, and
  the memory controller's PA-to-DA frontend gets "an array of N-to-1 multiplexers ... five multiplexers" (Fig. 12).
* **(b) Orin's physical-to-DRAM mapping is not reverse-engineered in the paper.** For the SoC side they "assume a
  typical mapping of row:rank:column:bank:channel, which we verify achieves near-peak sequential read bandwidth"
  (in simulation). Table II lists Orin as LPDDR5-6400, 256-bit, 204.8 GB/s. What they measured on the real Orin
  64 GB: huge-page allocation overhead on model load (Table I) and GEMM time (TinyChatEngine); PIM itself is
  **simulated** (AiM-style, a DRAMsim-based PIM simulator), and the GEMM layout penalty via GPGPU-Sim.
* **(c) Code:** the paper gives no code / artifact link; none found by search.

## Device probes (read-only; raw/ and stride_probe*.csv)
* `/sys/kernel/debug/tegra_mc_utils`: `num_channel` = **16**, `dram_type` = 8 (LPDDR5). `/proc/iomem` shows the
  memory controller's 16 per-channel register apertures (`2c00000.memory-controller ch0..ch15`) and one
  interleaved System RAM range — no per-channel memory windows.
* Live device tree `reserved-memory`: vpr, pva, fsi, camdbg, ramoops carve-outs and the CMA pool — special-purpose
  regions, none described as channel-restricted.
* nvmap debugfs heaps: iovmm (the CUDA path), fsi, vpr, page pool — no channel or interleave attribute exposed.
* EMC / MC debugfs: rates, `num_channel`, `dram_type`, `bpmp/debug/emc/bct` (the boot configuration table
  holds the DRAM / MC setup — the only place an address map is configured, i.e. a boot change).
* CUDA allocator: no channel / placement option exists in the CUDA or PyTorch allocator APIs.
* **Stride probe** (`bench/e13c_stride_probe.py`, 4 Mi one-line reads per stride over 8 GiB): 138–155 GB/s at
  64–256 B, then a smooth halving per stride doubling from 512 B to 64 KiB, flat ~2 GB/s beyond. No cliff at any
  power-of-two stride — the signature of a plain power-of-two channel interleave (all traffic on one channel at a
  stride = granularity × channels) is absent, which is consistent with hashed channel bits, but the probe cannot
  identify the mapping. `stride_probe_v1.csv` (reads per stride = buffer / stride) became launch-bound from 2 KiB and
  is kept only as the first attempt.

## What a direct measurement would need
Either (i) an MC address map that places a whole physical range on 8 of 16 channels — set, if at all, in the BCT
at boot (a boot change: stop, per T016), or (ii) FACIL-style hardware. Neither is available on this device.
