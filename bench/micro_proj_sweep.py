"""E5b S sweep for GR00T (T009 step 3, docs/03 §"S sweep (isolated kernels)").

docs/03 asks for S in {1, 16, <the model's own S>, 256} on the model's LARGEST projection and
its q projection, run as isolated kernels under ncu. bench/micro_ae_kernels.py does this for
pi0, but its shapes are pi0's expert (hidden 1024, 18 layers, multi-query) read from the pi0
checkpoint; GR00T's action head is a DiT (hidden 1536 per T009, 16 blocks), so the shapes are
different and must be read from GR00T's own loaded model.

Two modes, because loading GR00T needs the groot venv and ~6 GB, while the sweep itself needs
neither and runs cleanly under ncu with nothing else in the process:

  --emit-shapes shapes.json     (groot venv) load N1.7, walk the first DiT block's nn.Linear
                                modules, record every projection's in/out features and
                                dtype, and pick `largest` (max in*out) and `q` (attn1.to_q).
                                Shapes are READ, never typed in.
  --shapes shapes.json          (any venv with torch) for each projection x S: time the GEMM
                                with CUDA events and write sweep.csv; with --ncu-window P:S
                                run just that one GEMM inside cudaProfilerStart/Stop under an
                                NVTX range, for ncu --profile-from-start off.

The sweep is F.linear on random activations of shape (S, in) against a random weight of the
recorded shape and dtype. That is the GEMM the model launches for that projection; which
cuBLAS kernel serves it depends on S, which is the point of sweeping S.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MEASURED_GBPS, MEASURED_TFLOPS = 187.47, 33.8506


def emit_shapes(args) -> int:
    sys.path.insert(0, str(REPO / "bench"))
    import torch                                                        # noqa: PLC0415
    import groot_phases as GP                                           # noqa: PLC0415
    _, _, model, _, n, _ = GP.load_for_segments(args)
    blocks = model.action_head.model.transformer_blocks
    blk = blocks[0]
    projs = []
    for name, mod in blk.named_modules():
        if isinstance(mod, torch.nn.Linear):
            projs.append({"name": name, "in": mod.in_features, "out": mod.out_features,
                          "bias": mod.bias is not None,
                          "dtype": str(mod.weight.dtype).replace("torch.", "")})
    if not projs:
        raise SystemExit("no nn.Linear in DiT block 0 -- refusing to invent shapes")
    largest = max(projs, key=lambda p: p["in"] * p["out"])
    q = next((p for p in projs if p["name"].endswith("to_q")), None)
    if q is None:
        raise SystemExit(f"no *.to_q projection in DiT block 0; have "
                         f"{[p['name'] for p in projs]}")
    # The model's own S: the action horizon the DiT runs on, read from the action head.
    cfg = model.action_head.config
    own_s = None
    for a in ("action_horizon", "num_action_tokens", "action_dim_horizon"):
        if getattr(cfg, a, None):
            own_s = int(getattr(cfg, a))
            break
    out = {"model": f"groot_{args.version}", "checkpoint": args.checkpoint,
           "n_blocks": len(blocks), "N": n, "own_S_from_config": own_s,
           "block0_projections": projs, "largest": largest, "q": q,
           "note": "S in the DiT is the number of action tokens it attends over; T009 states "
                   "S = 41 for N1.7. own_S_from_config is what the loaded config says, "
                   "recorded so a disagreement is visible rather than assumed away."}
    Path(args.emit_shapes).parent.mkdir(parents=True, exist_ok=True)
    Path(args.emit_shapes).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("n_blocks", "own_S_from_config", "largest", "q")},
                     indent=2))
    return 0


def sweep(args) -> int:
    import torch                                                        # noqa: PLC0415
    import torch.nn.functional as F                                     # noqa: PLC0415
    sh = json.loads(Path(args.shapes).read_text())
    S_list = [int(s) for s in args.S.split(",")]
    targets = {"largest": sh["largest"], "q": sh["q"]}

    def make(p, S):
        dt = getattr(torch, p["dtype"])
        g = torch.Generator(device="cuda").manual_seed(0)
        x = torch.randn(S, p["in"], device="cuda", dtype=dt, generator=g)
        w = torch.randn(p["out"], p["in"], device="cuda", dtype=dt, generator=g)
        b = torch.randn(p["out"], device="cuda", dtype=dt, generator=g) if p["bias"] else None
        return x, w, b

    if args.ncu_window:
        key, S = args.ncu_window.split(":")
        p, S = targets[key], int(S)
        x, w, b = make(p, S)
        for _ in range(10):
            F.linear(x, w, b)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push(f"{key}_S{S}")
        for _ in range(3):
            F.linear(x, w, b)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return 0

    rows = []
    for key, p in targets.items():
        for S in S_list:
            x, w, b = make(p, S)
            for _ in range(20):
                F.linear(x, w, b)
            torch.cuda.synchronize()
            ts = []
            for _ in range(args.reps):
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                F.linear(x, w, b)
                e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1) * 1e3)
            us = statistics.median(ts)
            el = x.element_size()
            flops = 2 * S * p["in"] * p["out"]
            nbytes = (p["in"] * p["out"] + S * p["in"] + S * p["out"]
                      + (p["out"] if p["bias"] else 0)) * el
            fmem = nbytes / (MEASURED_GBPS * 1e9) * 1e6
            fcmp = flops / (MEASURED_TFLOPS * 1e12) * 1e6
            rows.append({"model": sh["model"], "proj": key, "proj_name": p["name"],
                         "in": p["in"], "out": p["out"], "dtype": p["dtype"], "S": S,
                         "median_us": round(us, 3), "p5_us": round(sorted(ts)[len(ts)//20], 3),
                         "p95_us": round(sorted(ts)[min(len(ts)-1, len(ts)*19//20)], 3),
                         "n": len(ts), "flops": flops, "bytes": nbytes,
                         "intensity": round(flops / nbytes, 3),
                         "floor_us": round(max(fmem, fcmp), 3),
                         "x_floor": round(us / max(fmem, fcmp), 3),
                         "achieved_GBps": round(nbytes / (us * 1e-6) / 1e9, 2),
                         "achieved_TFLOPps": round(flops / (us * 1e-6) / 1e12, 4)})
            print(f"  {key:8s} {p['name']:28s} S={S:4d}  {us:9.2f} us  "
                  f"x_floor {rows[-1]['x_floor']:6.2f}  {rows[-1]['achieved_GBps']:7.1f} GB/s")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "s_sweep.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"  wrote {out / 's_sweep.csv'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-shapes", default="")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--S", default="1,16,41,256")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--outdir", default="")
    ap.add_argument("--ncu-window", default="", help="largest:41 etc.; run one GEMM for ncu")
    # loader args, same as groot_phases.load_for_segments expects
    ap.add_argument("--checkpoint", default="nvidia/GR00T-N1.7-3B")
    ap.add_argument("--version", default="n17")
    ap.add_argument("--api", default="new")
    ap.add_argument("--embodiment-tag", dest="embodiment_tag",
                    default="oxe_droid_relative_eef_relative_joint")
    ap.add_argument("--gr00t-root", dest="gr00t_root", default="")
    ap.add_argument("--compat-shims", action="store_true")
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()
    if args.emit_shapes:
        return emit_shapes(args)
    if args.shapes:
        return sweep(args)
    ap.error("need --emit-shapes or --shapes")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
