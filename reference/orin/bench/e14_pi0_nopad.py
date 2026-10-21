"""T016 step 3, pi0 row: `pi0_bf16_both+nopad` -- the prefix without its 283 padded tokens (a VARIANT).

Implementation: bench/pi0_padtrim.py (harness-level; LeRobot untouched): the empty-camera slot is removed
before the image encoder and the prompt is sliced to its valid tokens; the expert gets the trimmed K/V.
Compilation: static shapes -- torch._dynamo automatic_dynamic_shapes is turned OFF so the new prefix length
compiles its own static graph (the same kernels-selection regime as the rung), recorded with the compile
count and wall time.

--mode time     pad / nopad bracketed (pad, nopad, pad, nopad, pad), n frames each: prefix (prefix_once on
                the trimmed ctx) and full frame (model.sample_actions on the trimmed inputs), CUDA events,
                device-side finite check on every action.
--mode equiv    >= 50 distinct inputs (5 fixtures x seeded +-2/255 image perturbation, per-input noise seed):
                actions nopad vs pad, bitwise count, max|d|, rel L2.
--mode ncu      NVTX ranges pad_prefix / nopad_prefix / pad_full / nopad_full, 5 frames each, for
                `ncu --graph-profiling graph --cache-control none` (bench/run_ncu_e14.sh).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
import corun as CR               # noqa: E402
import e12b_common as C          # noqa: E402
import phase_builders as PB      # noqa: E402
import pi0_padtrim as PT         # noqa: E402

N = 10


def setup(n_frames, perturb=False):
    import torch._dynamo as dyn                                                   # noqa: PLC0415
    dyn.config.automatic_dynamic_shapes = False
    dyn.config.cache_size_limit = max(16, dyn.config.cache_size_limit)
    policy, spec, cfg, note = C.load_rung(N)
    frames, manifest = CR.load_frames(policy, spec, REPO / "bench/fixtures/e12", n_frames, seed0=2000)
    if perturb:
        g = torch.Generator(device="cuda").manual_seed(777)
        for fr in frames:
            c = dict(fr["ctx"])
            c["images"] = [t + (torch.randint(-1, 2, t.shape, generator=g, device=t.device) * (2 / 255.0)
                                ).to(t.dtype) for t in c["images"]]
            fr["ctx"] = c
    return policy, spec, note, frames, manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("time", "equiv", "ncu"), required=True)
    ap.add_argument("--outdir", default="")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--inputs", type=int, default=50)
    a = ap.parse_args()
    g0 = C.gate_full()
    if "MAXN" not in g0["power_mode"]:
        raise SystemExit("MAXN only")
    policy, spec, note, frames, manifest = setup(a.inputs if a.mode == "equiv" else 5, a.mode == "equiv")
    fin = C.Finite()
    tok = {v: PT.prefix_tokens(policy, PT.ctx_for(frames[0]["ctx"], v)) for v in ("pad", "nopad")}
    print("tokens:", json.dumps(tok), flush=True)
    comp = {}

    def prefix(v, k):
        return PB.prefix_once(PT.ctx_for(frames[k % len(frames)]["ctx"], v))

    def full(v, k):
        fr = frames[k % len(frames)]
        out = PT.sample(policy, PT.ctx_for(fr["ctx"], v), fr["noise"])
        fin.add(out)
        return out

    with torch.no_grad():
        for v in ("pad", "nopad"):                    # compile each static length once, timed
            t0 = time.perf_counter()
            for k in range(5):
                prefix(v, k); full(v, k)
            torch.cuda.synchronize()
            comp[v] = round(time.perf_counter() - t0, 2)
        import torch._dynamo.utils as du                                            # noqa: PLC0415
        counters = {k: dict(v) for k, v in du.counters.items() if k in ("stats", "frames")}
        if a.mode == "ncu":
            torch.cuda.cudart().cudaProfilerStart()
            for v in ("pad", "nopad"):
                for st, fn in (("prefix", prefix), ("full", full)):
                    torch.cuda.nvtx.range_push(f"{v}_{st}")
                    for k in range(5):
                        fn(v, k)
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
            print("  ncu ranges done; nonfinite", fin.read(), flush=True)
            return 0
        out = REPO / a.outdir
        out.mkdir(parents=True, exist_ok=True)
        (out / "raw").mkdir(exist_ok=True)
        if a.mode == "equiv":
            rows = []
            for k in range(a.inputs):
                p = full("pad", k).float(); q = full("nopad", k).float()
                rows.append({"input": k, "fixture": k % 5, "noise_seed": 2000 + k,
                             "bitwise": bool(torch.equal(p, q)),
                             "maxabs": float((p - q).abs().max()),
                             "rel": float((p - q).norm() / p.norm())})
            nf = fin.read()
            with (out / "equiv.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            res = {"inputs": len(rows), "bitwise": sum(r["bitwise"] for r in rows),
                   "MAXABS": max(r["maxabs"] for r in rows), "REL": max(r["rel"] for r in rows),
                   "median_maxabs": statistics.median(r["maxabs"] for r in rows),
                   "median_rel": statistics.median(r["rel"] for r in rows), "nonfinite": nf}
        else:
            order = ["pad", "nopad", "pad", "nopad", "pad"]
            rows, blocks = [], []
            for bi, v in enumerate(order):
                for st, fn in (("prefix", prefix), ("full", full)):
                    ms = []
                    for k in range(a.frames):
                        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                        e0.record(); fn(v, k); e1.record(); torch.cuda.synchronize()
                        ms.append(e0.elapsed_time(e1))
                    s = C.summ(ms)
                    blocks.append({"block": bi, "variant": v, "stage": st, **s,
                                   "gpu_hz": C.gate_full()["gpu_cur_freq_hz"]})
                    rows += [{"block": bi, "variant": v, "stage": st, "frame": j, "ms": round(x, 4)}
                             for j, x in enumerate(ms)]
            nf = fin.read()
            with (out / "blocks.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(blocks[0])); w.writeheader(); w.writerows(blocks)
            with (out / "raw" / "latency_samples.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            res = {}
            for st in ("prefix", "full"):
                for v in ("pad", "nopad"):
                    vals = [r["ms"] for r in rows if r["variant"] == v and r["stage"] == st]
                    res[f"{v}_{st}"] = C.summ(vals)
                res[f"ratio_{st}"] = round(res[f"nopad_{st}"]["median_ms"] / res[f"pad_{st}"]["median_ms"], 4)
            res["nonfinite"] = nf
        (out / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
        thr = C.throttle_record(g0)
        C.write_meta(out / "meta.json", {
            "experiment": f"T016 step 3: pi0_bf16_both+nopad ({a.mode})", "bench_commit": C.bench_commit(),
            "model_commit": C.CKPT_COMMIT, "checkpoint": spec.checkpoint, "checkpoint_commit": C.CKPT_COMMIT,
            "rung": "bf16_both", "rung_note": note, "variant": "pi0_bf16_both+nopad vs pi0_bf16_both (pad)",
            "power_mode": g0["power_mode"], "sm_count": g0["sm_count"], "granted_partitions": "none (whole GPU)",
            "gpu_clock_hz": g0["gpu_cur_freq_hz"], "emc_rate_hz": g0["emc_rate_hz"], "gate_start": g0, **thr,
            "tokens": tok, "compile_wall_s": comp, "dynamo_counters": counters,
            "compilation": "automatic_dynamic_shapes=False: each prefix length compiles its own static graph",
            "implementation": "bench/pi0_padtrim.py (harness-level; LeRobot untouched)",
            "frames_manifest": manifest, "result": res, "command": " ".join([sys.executable, *sys.argv]),
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        print(json.dumps(res, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
