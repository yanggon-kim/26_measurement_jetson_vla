"""T011 step 2 / E11c — transfer cost at one model's EXACT boundary sizes (protocol §13.3).

T011: "do not repeat the size sweep. Measure only this model's exact boundary sizes (score matrix fp32 / fp16,
activation tiles, kv_bytes_total) with the three methods" -> xfer_exact.csv + model_inputs.csv (`xfer_latency`).
The three methods and their timing are T010's, imported unchanged from bench/xfer_latency.py (pageable, pinned,
zero_copy in its own managed-allocator process, plus the ordinary-device-tensor control for zero_copy).

The sizes are not typed from memory: each comes from a committed measurement of the model and is checked against it
before anything is timed.
  kv_bytes_total   results/20260924_E11c_<m>_handover/boundary.csv, row prefix_kv_cache (the hand-over object)
  S, D, I          results/20260924_E11a_<m>_hostseg/segments.csv tensor shapes, checked by the committed byte counts
  heads, T         pi05: 8 query heads over T = P + S keys (cast_scores_fp32_fp16 bytes = heads*S*T*(4+2), checked)
                   smolvla: 15 query heads (attention output width 960 = 15 x 64 in segments.csv); self-attention
                   layers attend to T = P + S (smolvlm_with_expert.py concatenates the cached prefix K/V with the
                   suffix's), cross-attention layers to T = P -- both score sizes are measured.

    python bench/xfer_exact.py --model pi05 --outdir results/<date>_E11c_pi05_handover_xfer
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi0_common as P  # noqa: E402
import xfer_latency as X  # noqa: E402
from common import REPO, collect_raw, compress_and_record, sh_out, software_versions, write_json  # noqa: E402

SRC = {"pi05": ("results/20260924_E11c_pi05_handover", "results/20260924_E11a_pi05_hostseg"),
       "smolvla": ("results/20260924_E11c_smolvla_handover", "results/20260924_E11a_smolvla_hostseg")}


def seg_bytes(hostseg: Path, segment: str) -> tuple[int, int]:
    for r in csv.DictReader(open(hostseg / "segments.csv")):
        if r["segment"] == segment:
            return int(r["bytes_read"]), int(r["bytes_written"])
    raise KeyError(segment)


def model_sizes(model: str) -> tuple[dict[str, int], dict]:
    handover, hostseg = (REPO / p for p in SRC[model])
    kv = next(int(r["bytes"]) for r in csv.DictReader(open(handover / "boundary.csv"))
              if r["tensor"] == "prefix_kv_cache")
    hm = json.loads((handover / "meta.json").read_text())
    tc = json.loads((hostseg / "meta.json").read_text())["token_counts"]
    Pt, S = tc["prefix_tokens_P"], tc["suffix_tokens_S"]
    if model == "pi05":
        D, I, H = 1024, 4096, 8
        T = Pt + S
        # checks against the committed E11a byte counts (mapping_only cast of the score matrix, GeGLU, residual)
        assert sum(seg_bytes(hostseg, "cast_scores_fp32_fp16")) == H * S * T * (4 + 2), "pi05 score size"
        assert sum(seg_bytes(hostseg, "gelu_gate_mul_up")) == 3 * S * I * 2, "pi05 I"
        assert sum(seg_bytes(hostseg, "residual")) == 4 * S * D * 2, "pi05 D"
        sizes = {"scores_fp32": H * S * T * 4, "scores_fp16": H * S * T * 2}
    else:
        D, I, H = 720, 2048, 15
        assert seg_bytes(hostseg, "residual_rmsnorm2")[0] == 3 * S * D * 2, "smolvla D"  # x + attn_out + weight row
        assert seg_bytes(hostseg, "attention_whole")[1] == S * H * 64 * 2, "smolvla heads x head_dim"
        assert sum(seg_bytes(hostseg, "gelu_gate_mul_up")) == 3 * S * I * 2, "smolvla I"
        sizes = {"scores_self_fp32": H * S * (Pt + S) * 4, "scores_self_fp16": H * S * (Pt + S) * 2,
                 "scores_cross_fp32": H * S * Pt * 4, "scores_cross_fp16": H * S * Pt * 2}
    sizes.update({"activation_tile": S * D * 2, "ffn_tile": S * I * 2, "kv_bytes_total": kv})
    info = {"P": Pt, "S": S, "D": D, "I": I, "query_heads": H, "kv_bytes_total_source": f"{SRC[model][0]}/boundary.csv",
            "shapes_source": f"{SRC[model][1]}/segments.csv", "model_commit": hm["model_commit"],
            "model_repo": hm.get("model_repo"), "checkpoint": hm.get("checkpoint"),
            "num_denoise_steps": hm.get("num_denoise_steps"), "action_horizon": hm.get("action_horizon")}
    return sizes, info


def zero_copy(sizes: dict[str, int], out: Path, warmup: int, timed: int) -> tuple[list[dict], str]:
    if not X.ZC_SO.exists():
        return [], f"{X.ZC_SO} not built; see env/README.md for the nvcc command"
    cmd = [sys.executable, X.__file__, "--zero-copy-worker", json.dumps(sizes),
           "--warmup", str(warmup), "--timed", str(timed)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    (out / "raw" / "zero_copy_worker.log").write_text(r.stdout + "\n" + r.stderr)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("ZERO_COPY_JSON ")), None)
    if r.returncode != 0 or line is None:
        tail = (r.stderr or r.stdout).strip().splitlines()
        return [], f"worker exit={r.returncode}; {tail[-1][:200] if tail else 'no output'}"
    rows = []
    for e in json.loads(line[len("ZERO_COPY_JSON "):]):
        for part, st in e["parts"].items():
            rows.append({"method": "zero_copy", "direction": part, "size_label": e["size_label"],
                         "size_bytes": e["size_bytes"], "median_ms": round(st["median_ms"], 6),
                         "p5_ms": round(st["p5_ms"], 6), "p95_ms": round(st["p95_ms"], 6), "min_ms": "",
                         "n": st["n"], "GBps": round(e["size_bytes"] / (st["median_ms"] / 1e3) / 1e9, 3)
                         if st["median_ms"] > 0 else "", "samples": st.get("samples", [])})
    return rows, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(SRC))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=200)
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    gate = P.gate()
    print("gate:", json.dumps(gate), flush=True)
    sizes, info = model_sizes(args.model)
    print("sizes:", json.dumps(sizes), flush=True)

    rows = X.measure_transfers(sizes, args.warmup, args.timed)
    zc_rows, zc_err = zero_copy(sizes, out, args.warmup, max(200, args.timed))
    if zc_err:
        print(f"  zero_copy NOT MEASURED: {zc_err}")
    ctrl = X.control_gpu_read(sizes, args.warmup, args.timed)
    all_x = rows + zc_rows + ctrl
    samples = [{"method": r["method"], "direction": r["direction"], "size_bytes": r["size_bytes"],
                "iter": i, "ms": v} for r in all_x for i, v in enumerate(r.get("samples", []))]
    for r in all_x:
        r.pop("samples", None)
        print(f"  {r['method']:15s} {r['direction']:10s} {r['size_label']:18s} {r['size_bytes']:>10d} B "
              f"{r['median_ms']:9.4f} ms", flush=True)
    with (out / "xfer_exact.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model"] + list(all_x[0].keys()))
        w.writeheader()
        w.writerows([{"model": args.model, **r} for r in all_x])
    sample_csv = out / "raw" / "xfer_exact_samples.csv"
    with sample_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "direction", "size_bytes", "iter", "ms"])
        w.writeheader()
        w.writerows(samples)
    entry = compress_and_record(sample_csv)
    write_json(out / "raw" / "ORIGINALS.json", {"files": [entry]})

    from host_segments import ModelInputs
    bench_commit = sh_out(f"git -C {REPO} rev-parse HEAD")
    mi = ModelInputs(args.model, info["model_commit"], bench_commit, "MAXN+jetson_clocks", info["P"], info["S"],
                     info["num_denoise_steps"])
    for r in all_x:
        for stat, col in (("median", "median_ms"), ("p5", "p5_ms"), ("p95", "p95_ms")):
            if r[col] != "":
                mi.add("xfer_latency", r[col], "ms", stat, r["n"], "xfer_exact.csv",
                       key=f"{r['method']}:{r['direction']}",
                       variant=f"size_label={r['size_label']};size_bytes={r['size_bytes']}", timing="wall_isolated")
    n_mi = mi.write(out / "model_inputs.csv")

    sw = software_versions()
    meta = {
        "experiment": "E11c", "task": "T011", "step": "2 (transfers at the exact boundary sizes)",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model, "model_repo": info["model_repo"], "model_commit": info["model_commit"],
        "checkpoint": info["checkpoint"],
        "precision": "uint8 buffers for pageable/pinned; float32 for zero_copy and its device control",
        "compile_mode": "none (eager)", "batch": 1, "num_denoise_steps": info["num_denoise_steps"],
        "action_horizon": info["action_horizon"], "prefix_tokens": info["P"], "suffix_tokens": info["S"],
        "n_warmup": args.warmup, "n_timed": args.timed,
        "power_mode": gate["power_mode"], "sm_count": gate["sm_count"], "gpu_hz": gate["gpu_cur_freq_hz"],
        "gate": gate, "emc_hz": int(sh_out("sudo -n sh -c 'cat /sys/kernel/debug/bpmp/debug/clk/emc/rate'") or 0),
        "jetpack": "6.2.3+b81", "l4t": "R36.5.0", "cuda": sw.get("torch_cuda"), "torch": sw.get("torch"),
        "torchvision": sw.get("torchvision"), "triton": sw.get("triton"), "python": sw.get("python"),
        "venv": sw.get("executable"), "bench_commit": bench_commit,
        "bench_files_sha256": sw.get("bench_files_sha256"),
        "command": f"python bench/xfer_exact.py --model {args.model} --outdir {args.outdir}",
        "sizes": sizes, "size_derivation": {k: v for k, v in info.items() if k not in ("model_commit",)},
        "zero_copy_route": "CUDAPluggableAllocator over cudaMallocManaged (bench/managed_alloc.cu)",
        "zero_copy_error": zc_err or "none",
        "throttled": False, "throttled_evidence": "GPU clock pinned min==max, recorded in `gate`.",
        "model_inputs_rows": n_mi, "raw_files": [],
        "notes": ["Methods and timing are T010's (bench/xfer_latency.py), unchanged; no size sweep (T011).",
                  "zero_copy sizes are rounded down to a multiple of 4 B (float32 elements); size_bytes records it.",
                  "The fixed cost of a transfer is fitted in results/20260922b_E11b_xfer (xfer_fit.csv); this "
                  "directory only adds the model's exact sizes."],
    }
    write_json(out / "meta.json", meta)
    meta["raw_files"] = collect_raw(out, meta["command"])
    write_json(out / "meta.json", meta)
    print(f"\nwrote {out} ({n_mi} model_inputs rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
