"""summary.md for each E11 result directory (protocol §5).

Each function reads the CSVs the corresponding bench script has just written, so a
summary can also be regenerated for an existing directory without re-measuring:

    python bench/e11_summaries.py results/<run>

The rule these follow is the reporting style in AGENTS.md: the answer first (a table
with median, p5-p95 and n), then the conditions, then the anomalies, then how to
reproduce. Measured / derived / not measured are never mixed in one row.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import write_summary  # noqa: E402


def rd(p: Path) -> list[dict]:
    return list(csv.DictReader(p.open())) if p.exists() else []


def f(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x or "")


def table(rows: list[dict], cols: list[str], heads: list[str] | None = None,
          nd: dict | None = None, limit: int | None = None) -> str:
    heads = heads or cols
    nd = nd or {}
    out = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(cols)]
    for r in rows[:limit]:
        out.append("| " + " | ".join(f(r.get(c), nd[c]) if c in nd else str(r.get(c, ""))
                                    for c in cols) + " |")
    return "\n".join(out)


def _absent(out: Path) -> list[str]:
    """§13.8 quantities this directory could not measure, from meta.json."""
    m = out / "meta.json"
    if not m.exists():
        return []
    j = json.loads(m.read_text())
    n = j.get("not_measured") or []
    return [x if isinstance(x, str) else json.dumps(x) for x in n]


# ---------------------------------------------------------------------------- E11a


def e11a(out: Path) -> Path:
    seg, hl, defs = (rd(out / "segment_summary.csv"), rd(out / "host_layer_summary.csv"),
                     rd(out / "segments.csv"))
    # §13.1's PRIMARY timing is wall_isolated with no pause between calls.
    prim = [r for r in seg if r["timing"] == "wall_isolated"
            and r["variant"].endswith("gap_ms=0")]

    def layer_of(r):
        return r["layer"]

    styles = ["eager", "compiled", "cudagraph"]
    by = {(r["segment"], r["layer"], r["exec_style"]): r for r in prim}
    bytes_of = {d["segment"]: d for d in defs}

    def build(layers: list[str]) -> list[str]:
        """One table. `layer = -1` holds the per-step extras, which are not per-layer
        segments and must not be listed under a layer they never ran in."""
        names, seen = [], set()
        for r in prim:
            if r["layer"] in layers and r["segment"] not in seen:
                seen.add(r["segment"])
                names.append(r["segment"])
        if not names:
            return []
        t = ["| segment | layer | " + " | ".join(f"{st} ms" for st in styles)
             + " | kernels, eager | bytes read + written | FLOPs |",
             "|---|---:|" + "---:|" * (len(styles) + 3)]
        for lay in layers:
            for n in names:
                if (n, lay, "eager") not in by:
                    continue
                cells = [f(by[(n, lay, st)]["median_ms"], 4) if (n, lay, st) in by
                         else "not measured" for st in styles]
                e = by[(n, lay, "eager")]
                d = bytes_of.get(n, {})
                br, bw = d.get("bytes_read", ""), d.get("bytes_written", "")
                t.append(f"| `{n}` | {lay} | " + " | ".join(cells)
                         + f" | {f(e['n_kernels'], 1) if e['n_kernels'] else ''}"
                         + f" | {(br + ' + ' + bw) if br else ''}"
                         + f" | {d.get('flops', '')} |")
        return t

    layers = sorted({r["layer"] for r in prim if r["layer"] != "-1"},
                    key=lambda x: int(x))
    tbl = build(layers)
    extras_tbl = build(["-1"])

    lay_tbl = ["| key | exec_style | layer | median ms | p5 | p95 | n | vs the analytic "
               "0.16 ms |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(hl, key=lambda r: (r["variant"], float(r["median_ms"]))):
        v = float(r["median_ms"])
        lay_tbl.append(f"| `{r['key']}` | {r['exec_style']} | {r['variant']} | "
                       f"**{v:.4f}** | {f(r['p5_ms'], 4)} | {f(r['p95_ms'], 4)} | {r['n']} | "
                       f"{v/0.16:.2f}x |")

    nm = _absent(out)
    sections = [
        ("Answer — per-segment latency, `wall_isolated`, `gap_ms = 0` (the §13.1 primary)",
         "\n".join(tbl) + "\n\n**Per-step extras** (`order_in_layer = -1`: once per "
         "denoising step, not once per layer)\n\n" + "\n".join(extras_tbl)
         + "\n\n`gap_ms = 2`, `gpu_event` and `wall_launch` for every row, "
         "with achieved GB/s and TFLOP/s where the segment has bytes and FLOPs, are in "
         "`segment_summary.csv`; the byte and FLOP formulas and every input's shape, dtype "
         "and contiguity in `segments.csv`; the 216,000 per-iteration samples in "
         "`raw/segment_samples.csv.zst`; kernel counts, their `key_averages` table and one "
         "Chrome trace per (segment, layer, style) in `raw/profiler/`."),
        ("A whole layer, six ways", "\n".join(lay_tbl) + "\n\n`sum_of_segments` is derived "
         "(the six mapping segments added up); `sequence` and `whole_layer_graph` are "
         "measured directly. The ordering is the finding: **one CUDA graph over the whole "
         "layer is the only implementation that comes near the mapping's 0.16 ms**, and it "
         "beats a compiled sequence by about 2.4x. Compiling the sequence is worth ~2.5x "
         "over compiling the six segments separately — fusion does cross the segment "
         "boundaries the mapping treats as separable — but it lands ~2.8x above 0.16 ms, "
         "not below it."),
        ("Conditions",
         "pi0 @ `45dcc8fc`, bf16, MAXN + jetson_clocks, P = 816 processed (533 valid), "
         "S = 51, N = 4. Layers 0 and 9 of the action expert. >= 20 warm-ups and 1,000 timed "
         "calls per (segment, style, timing) — §13.1 asks for >= 20 and >= 1,000. Inputs are "
         "the model's own tensors, captured with hooks from one real denoising step of the "
         "fixture and replayed through the model's own callables; the prefix K/V are read "
         "from the cache **before** the layer updates it, so the replayed attention runs "
         "over the real T = P + S = 867 keys rather than S = 51."),
        ("The caveat that governs every compiled number",
         "A compiled callable whose outputs are discarded is dead-code-eliminated. Measured "
         "on this device to size the error: a compiled 2048^3 bf16 matmul costs **0.0662 ms** "
         "when its result is discarded and **0.6605 ms** when it is returned. An earlier "
         "version of this experiment returned `None` from the timed sequence and reported it "
         "at 0.085 ms, which measured nothing. Every timed callable here returns its "
         "results."),
        ("Reproduce",
         "```\n# kernel counts and profiler traces need root: CUPTI does, on this device.\n"
         "# They run as a SEPARATE pass because attaching a profiler inflates every\n"
         "# subsequent call by ~30 %.\n"
         f"sudo -n /bin/sh -c 'cd $PWD && HOME=$HOME HF_HUB_OFFLINE=1 \\\n"
         f"    exec python bench/host_segments.py --outdir {out} --mode count'\n"
         f"python bench/host_segments.py --outdir {out} --warmup 20 --timed 1000 \\\n"
         f"    --kernels-csv {out}/kernels_per_call.csv\n"
         f"bash bench/run_nsys.sh {out} host_sequences -- \\\n"
         f"    python bench/host_segments.py --outdir {out} --mode nsys\n```"),
    ]
    return write_summary(out, "E11a - pi0 host segments (protocol §13.1, §13.2)",
                         sections, nm)


# ---------------------------------------------------------------------------- E11b


def e11b(out: Path) -> Path:
    fit, xs, sy = rd(out / "xfer_fit.csv"), rd(out / "xfer_summary.csv"), rd(out / "sync_summary.csv")
    pin = [r for r in fit if r["method"] == "pinned"]
    a_us = [1000 * float(r["fit_a_ms"]) for r in pin]
    sections = [
        ("Answer — the fixed cost of a transfer is not zero",
         (f"A host<->device transfer of B bytes costs `t = a + b*B`. Measured on this "
          f"device, with pinned memory, **a = {min(a_us):.1f}-{max(a_us):.1f} us** and "
          f"b corresponds to {min(float(r['GBps_from_b']) for r in pin):.1f}-"
          f"{max(float(r['GBps_from_b']) for r in pin):.1f} GB/s. The hybrid model assumes "
          f"a = 0 and prices every transfer as bytes / 143 GB/s.\n\n"
          + table(fit, ["method", "direction", "fit_a_ms", "fit_b_ms_per_MB", "GBps_from_b",
                        "fit_r2", "n_points"],
                  ["method", "dir", "a (ms)", "b (ms/MB)", "GB/s from b", "r2", "points"],
                  {"fit_a_ms": 4, "fit_b_ms_per_MB": 4, "GBps_from_b": 2, "fit_r2": 4}))),
        ("Per-size medians (every size, both directions, three methods)",
         table(xs, ["method", "direction", "size_label", "size_bytes", "median_ms", "p5_ms",
                    "p95_ms", "n", "GBps"],
               ["method", "dir", "size", "bytes", "median ms", "p5", "p95", "n", "GB/s"],
               {"median_ms": 4, "p5_ms": 4, "p95_ms": 4, "GBps": 2})),
        ("Synchronisation costs",
         table(sy, ["sync", "median_ms", "p5_ms", "p95_ms", "n", "net_of_control_ms", "note"],
               ["sync", "median ms", "p5", "p95", "n", "net of control", "note"],
               {"median_ms": 4, "p5_ms": 4, "p95_ms": 4, "net_of_control_ms": 4})),
        ("Conditions",
         "model = none (no policy is loaded; the sizes are π0's, taken from E11c). "
         "MAXN + jetson_clocks, GPU 1300.5 MHz. MB = 10^6 B in the fits; KiB/MiB labels "
         "are binary. Zero-copy runs in a separate process because "
         "`torch.cuda.memory.change_current_allocator` is global and irreversible, and its "
         "control — the same reduction over an ordinary device tensor — runs in that same "
         "process so the two are comparable."),
        ("Reproduce",
         "```\nnvcc -O3 -arch=sm_87 --shared -Xcompiler -fPIC \\\n"
         "     -o bench/managed_alloc.so bench/managed_alloc.cu\n"
         f"python bench/xfer_latency.py --outdir {out}\n```"),
    ]
    return write_summary(out, "E11b — transfer, synchronisation and zero-copy costs "
                              "(protocol §13.3)", sections, _absent(out))


# ---------------------------------------------------------------------------- E11c


def e11c(out: Path) -> Path:
    hand, bound, kv = rd(out / "handover.csv"), rd(out / "boundary.csv"), rd(out / "kv_copy.csv")
    tot = [r for r in hand if r["layer"] == "TOTAL"]
    sections = [
        ("Answer — the hand-over object",
         ("The object the VLM prefix hands to the action expert is the prefix KV cache.\n\n"
          + table(tot, ["tensor", "shape", "dtype", "bytes", "valid_tokens", "padded_tokens",
                        "notes"],
                  ["tensor", "shape", "dtype", "bytes", "valid", "padded", "notes"]))),
        ("Per-layer detail", table(hand, ["layer", "tensor", "shape", "dtype", "stride",
                                          "contiguous", "bytes", "valid_tokens",
                                          "padded_tokens"],
                                   ["layer", "tensor", "shape", "dtype", "stride", "contig",
                                    "bytes", "valid", "padded"], limit=8)
         + "\n\n(first rows; the full table is `handover.csv`)"),
        ("Boundary tensors", table(bound, ["tensor", "direction", "shape", "dtype", "bytes",
                                           "per", "notes"],
                                   ["tensor", "dir", "shape", "dtype", "bytes", "per", "notes"])),
        ("Is it copied? — measured, not read off the source",
         ("`copy.deepcopy(past_key_values)` at `modeling_pi0.py:918` runs once per denoising "
          "step. The evidence and its cost are in `kv_reuse.md`; the timing is:\n\n"
          + (table(kv, ["what", "median_ms", "p5_ms", "p95_ms", "n", "bytes", "GBps"],
                   ["what", "median ms", "p5", "p95", "n", "bytes", "GB/s"],
                   {"median_ms": 4, "p5_ms": 4, "p95_ms": 4, "GBps": 2}) if kv
             else "`kv_copy.csv` absent — not measured in this run."))),
        ("Conditions",
         "π0, bf16, MAXN + jetson_clocks, P = 816 processed prefix positions of which 533 "
         "carry a valid mask, S = 51, N = 4. Shapes and strides are read from the live "
         "`past_key_values` of a real inference, not from the config."),
        ("Reproduce", f"```\npython bench/handover.py --outdir {out}\n```"),
    ]
    return write_summary(out, "E11c — the VLM→expert hand-over object (protocol §13.4)",
                         sections, _absent(out))


# ---------------------------------------------------------------------------- E11e


def e11e(out: Path) -> Path:
    mp, cu, pw = rd(out / "mempressure.csv"), rd(out / "cpu_util.csv"), rd(out / "power.csv")
    sections = [
        ("Answer — what a memory-bandwidth competitor does to the prefix",
         table(mp, ["load_label", "duty_pct", "bg_GBps_alone", "bg_GBps_during",
                    "pct_of_measured_peak", "median_ms", "p5_ms", "p95_ms", "n",
                    "throughput_inf_per_s", "slowdown"],
               ["load", "duty %", "bg GB/s alone", "bg GB/s during", "% of 187.47",
                "median ms", "p5", "p95", "n", "inf/s", "slowdown"],
               {"median_ms": 3, "p5_ms": 3, "p95_ms": 3, "throughput_inf_per_s": 3})),
        ("CPU utilisation per phase",
         table([r for r in cu if not (r["scope"].startswith("cpu") and r["scope"] != "cpu")],
               ["phase", "scope", "pct", "median_pct", "p5_pct", "p95_pct", "n", "cores_busy"],
               ["phase", "scope", "pct", "median", "p5", "p95", "n", "cores busy"])
         + "\n\nThree scopes, because they answer different questions: `thread` is what the "
           "Python loop itself burns, `process` adds the CUDA driver's worker threads, and "
           "`system_all_cores` / `busiest_core` come from the kernel's own `/proc/stat` "
           "accounting, which is the only one that sees the whole machine. Per-core samples "
           "are in `raw/cpu_util_samples.csv`."),
        ("Power per phase (four INA3221 rails)",
         table(pw, ["phase", "rail", "mean_mW", "median_mW", "p5_mW", "p95_mW", "n"],
               ["phase", "rail", "mean mW", "median", "p5", "p95", "n"])
         + "\n\n`VDDQ_VDD2_1V8AO` is the memory **regulator input**, not DRAM power; it is "
           "never rescaled to 1.8 V. The four rails do not double-count (T004/E7)."),
        ("Conditions",
         "π0, MAXN + jetson_clocks. The competitor is a separate process on cores 6-11 "
         "(the foreground is pinned to 0-5), 20 ms period, duty ∈ {25, 50, 75, 100} %. Its "
         "achieved rate is measured inside its own duty window and compared with the "
         "**measured** peak of 187.47 GB/s (T005/E1), not the 204.8 GB/s spec. Each "
         "foreground point runs ≥ 60 s with tegrastats attached."),
        ("Limits — what this is and is not (protocol §13.5, repeated verbatim as it asks)",
         "This is a **proxy** for memory-side activity on the same DRAM. A real PIM occupies "
         "DRAM channels through the command bus and blocks ordinary accesses to them; a "
         "software hog instead competes at the memory controller and in the caches, and "
         "additionally takes GPU time slices (`gpu_hog`) or CPU cores (`cpu_hog`). The "
         "controls bound those side effects; **the result is the sensitivity of the GPU work "
         "to lost memory bandwidth, not a measurement of PIM interference.**\n\n"
         "The two controls at d = 100 % are `gpu_spin_control` (a compute-bound GPU load with "
         "a small memory footprint — small GEMMs) and `cpu_spin_control` (the same worker "
         "processes on the same cores doing arithmetic that stays in cache). They separate "
         "\"the GPU or CPU was taken\" from \"the memory bandwidth was taken\"; both are "
         "rows of the table above."),
        ("Anomaly to keep in mind",
         "T005 measured up to 1.19× run-to-run variation in achieved bandwidth with clocks "
         "and temperature held constant and no competing process; the cause is not "
         "established. Every bandwidth-derived figure here carries that ±8 %."),
        ("Reproduce",
         f"```\npython bench/mempressure.py --outdir {out} --seconds 60\n```"),
    ]
    return write_summary(out, "E11e — memory pressure and CPU (protocol §13.5, §13.6)",
                         sections, _absent(out))


# ---------------------------------------------------------------------------- E11f


def e11f(out: Path) -> Path:
    th, gi, co = rd(out / "throughput.csv"), rd(out / "gpu_idle.csv"), rd(out / "concurrency.csv")
    sections = [
        ("Answer — throughput of the three loops",
         table(th, ["label", "phase", "mode", "median_ms", "p5_ms", "p95_ms", "max_ms", "n",
                    "wall_s", "throughput_inf_per_s"],
               ["loop", "phase", "mode", "median ms", "p5", "p95", "max", "n", "wall s",
                "inf/s"], {"median_ms": 3, "p5_ms": 3, "p95_ms": 3, "max_ms": 3})),
        ("GPU idle inside the denoising loop",
         table(gi, ["mode", "N", "n_inferences", "n_ae_step_spans", "ae_total_ms",
                    "ae_total_ms_unprofiled", "profiling_inflation", "kernel_busy_ms",
                    "gpu_idle_frac", "gpu_idle_ms_corrected", "gpu_idle_frac_corrected"],
               ["mode", "N", "inferences", "ae_step spans", "ae_total ms (nsys)",
                "ae_total ms (unprofiled)", "inflation", "kernel busy ms",
                "idle frac (raw)", "idle ms (corrected)", "**idle frac (corrected)**"],
               {"ae_total_ms": 3, "kernel_busy_ms": 3, "gpu_idle_frac": 4,
                "gpu_idle_ms_corrected": 3, "gpu_idle_frac_corrected": 4})
         + "\n\n**Quote the corrected column.** Two corrections are folded in:\n\n"
           "1. The idle is computed **per inference** and averaged, not as one envelope over "
           "the whole capture. An envelope across all NVTX instances counts the gap between "
           "inferences — the next inference's vision and VLM prefix, which saturate the GPU — "
           "as expert idle.\n"
           "2. nsys instruments every CUDA API call, and this workload is launch-bound, so the "
           "wall time it reports for the region is inflated (see the `inflation` column). "
           "Kernel durations are **not** inflated — nsys reads them from the GPU — so "
           "`(profiled wall − busy) / profiled wall` charges nsys's own overhead to the model. "
           "The corrected figure divides the same measured kernel-busy time by the "
           "**unprofiled** wall time of the identical loop, measured in the same process a few "
           "seconds earlier with no profiler attached."),
        ("Concurrency — as two streams and as two processes",
         table(co, ["activity", "condition", "sharing", "mode", "median_ms", "p95_ms",
                    "max_ms", "n", "slowdown"],
               ["activity", "condition", "sharing", "mode", "median ms", "p95", "max", "n",
                "slowdown"], {"median_ms": 3, "p95_ms": 3, "max_ms": 3})),
        ("Conditions",
         "π0, bf16, MAXN + jetson_clocks, N = 4 unless the row says otherwise. Region "
         "instrumentation forces a graph break under `torch.compile`, so the NVTX-derived "
         "idle fraction is an eager measurement; the compiled rows report end-to-end only."),
        ("Reproduce", f"```\npython bench/pipelining.py --outdir {out}\n```"),
    ]
    return write_summary(out, "E11f — pipelining and overlap (protocol §13.7)",
                         sections, _absent(out))


# ---------------------------------------------------------------------------- E11d


def e11d(out: Path) -> Path:
    sw, fit = rd(out / "prefix_sweep.csv"), rd(out / "prefix_fit.csv")
    sections = [
        ("Answer — prefix latency vs prefix length",
         table(sw, ["mode", "n_images", "prompt_label", "prompt_tokens", "image_tokens",
                    "P_processed", "P_valid", "region", "median_ms", "p5_ms", "p95_ms", "n"],
               ["mode", "images", "prompt", "prompt tok", "image tok", "P processed",
                "P valid", "region", "median ms", "p5", "p95", "n"],
               {"median_ms": 3, "p5_ms": 3, "p95_ms": 3})),
        ("Fits", table(fit, ["fit", "region", "x", "fit_a_ms", "fit_b", "unit_b", "fit_r2",
                             "n_points", "note"],
                       ["fit", "region", "x", "a (ms)", "b", "unit b", "r2", "points", "note"],
                       {"fit_a_ms": 4, "fit_r2": 4})),
        ("Conditions",
         "π0, bf16, MAXN + jetson_clocks. **P_processed is constant at 816 for every point**: "
         "the processor always encodes three image slots and pads the prompt to 48 tokens, so "
         "what the sweep varies is how many of those 816 positions carry a valid mask "
         "(P_valid), not how many are computed. That is why prefix latency is flat."),
        ("Reproduce",
         f"```\npython bench/prefix_sweep.py --outdir-prefix {out.parent}/{out.name.rsplit('_', 2)[0]}\n```"),
    ]
    return write_summary(out, "E11d — prefix latency vs its input (measurement plan, part 4)", sections, _absent(out))


DISPATCH = {"E11a": e11a, "E11b": e11b, "E11c": e11c, "E11d": e11d, "E11e": e11e, "E11f": e11f}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rc = 0
    for d in sys.argv[1:]:
        out = Path(d)
        part = next((k for k in DISPATCH if f"_{k}_" in out.name or out.name.endswith(k)), None)
        if part is None:
            print(f"{out}: no summary writer (the consolidated directory writes its own)")
            rc = 1
            continue
        print("wrote", DISPATCH[part](out))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
