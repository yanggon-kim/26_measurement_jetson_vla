"""T016 step 1d (EV-02c) H1 / H2 split: standalone sweeps of the prefix GEMMs that carry the excess reads.

--run  (under ncu): for each base shape (M, N, K) given with --shapes "M,N,K;M,N,K", run F.linear
       x[m, K] @ w[N', K]^T for m in M x {0.25, 0.5, 1, 2} (N fixed) and N' in N x {0.25, 0.5, 1, 2}
       (M fixed), each point 3 launches inside NVTX range g<i>_M<m>_N<n>_K<k>.
--fit  (after ncu): per base shape, read the per-launch L2 fills of every point and fit
           fills = a * W * ceil(m / Tm) + b * X * ceil(n / Tn)
       where W = N'*K*el (weight), X = m*K*el (activation), and (Tm, Tn) the kernel's CTA tile, taken
       from its name (e.g. `..._128x64_...`), both orientations tried and the better least-squares fit
       kept. At the base shape: weight re-read excess = W*(a*ceil(M/Tm) - 1), activation excess =
       X*(b*ceil(N/Tn) - 1); H1 share = weight excess / (sum). The model and its fit residual are
       reported with the numbers -- a derived decomposition, not a measurement of which bytes were which.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

SCALES = (0.25, 0.5, 1.0, 2.0)


def points(shapes):
    out = []
    for gi, (M, N, K) in enumerate(shapes):
        for s in SCALES:
            out.append((gi, max(1, int(M * s)), N, K, "M"))
        for s in SCALES:
            if s != 1.0:
                out.append((gi, M, max(8, int(N * s)), K, "N"))
    return out


def run(shapes):
    import torch                                                      # noqa: PLC0415
    import torch.nn.functional as F                                   # noqa: PLC0415
    pts = points(shapes)
    g = torch.Generator(device="cuda").manual_seed(0)
    torch.cuda.synchronize()
    for gi, m, n, k, _ in pts:                                        # warm every shape first
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g)
        for _ in range(3):
            F.linear(x, w)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for gi, m, n, k, _ in pts:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"g{gi}_M{m}_N{n}_K{k}")
        for _ in range(3):
            F.linear(x, w)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()


def tile(name):
    m = re.search(r"_(\d+)x(\d+)_", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def fit(raw_csv, shapes, out_json):
    rows = list(csv.reader(open(raw_csv)))
    hdr, units, data = rows[0], rows[1], rows[2:]
    c = {h: i for i, h in enumerate(hdr)}
    nv = next(i for i, h in enumerate(hdr) if "Push/Pop_Range" in h)
    per = {}
    for r in data:
        mm = re.search(r"g(\d+)_M(\d+)_N(\d+)_K(\d+)", r[nv])
        if not mm:
            continue
        key = tuple(int(x) for x in mm.groups())
        f = (float(r[c["lts__d_sectors_fill_device.sum"]].replace(",", "") or 0) +
             float(r[c["lts__d_sectors_fill_sysmem.sum"]].replace(",", "") or 0)) * 32
        per.setdefault(key, []).append((r[c["Kernel Name"]], f))
    res = []
    for gi, (M, N, K) in enumerate(shapes):
        pts = [(k, v) for k, v in per.items() if k[0] == gi]
        kern = {n for k, v in pts for n, _ in v}
        best = None
        for orient in ("MxN", "NxM"):
            A, B, Y = [], [], []
            for (g_, m, n, k), v in pts:
                main = [f for nm, f in v if "splitKreduce" not in nm]
                name = next((nm for nm, _ in v if "splitKreduce" not in nm), "")
                t = tile(name) or (128, 128)
                Tm, Tn = t if orient == "MxN" else (t[1], t[0])
                W, X = n * k * 2, m * k * 2
                A.append(W * math.ceil(m / Tm)); B.append(X * math.ceil(n / Tn))
                Y.append(sum(f for nm, f in v) / max(1, len(main)))   # per call (+ reduce)
            # least squares y = a A + b B
            saa = sum(a * a for a in A); sbb = sum(b * b for b in B); sab = sum(a * b for a, b in zip(A, B))
            say = sum(a * y for a, y in zip(A, Y)); sby = sum(b * y for b, y in zip(B, Y))
            det = saa * sbb - sab * sab
            if det == 0:
                continue
            a = (say * sbb - sby * sab) / det; b = (sby * saa - say * sab) / det
            rss = sum((y - a * x - b * z) ** 2 for x, z, y in zip(A, B, Y))
            rel = math.sqrt(rss / len(Y)) / (sum(Y) / len(Y))
            if best is None or rel < best["rel_rmse"]:
                best = {"orient": orient, "a": a, "b": b, "rel_rmse": rel}
        base = [(k, v) for k, v in per.items() if k == (gi, M, N, K)]
        name = next((nm for nm, _ in base[0][1] if "splitKreduce" not in nm), "") if base else ""
        t = tile(name) or (128, 128)
        Tm, Tn = t if best and best["orient"] == "MxN" else (t[1], t[0])
        W, X = N * K * 2, M * K * 2
        wex = W * (best["a"] * math.ceil(M / Tm) - 1) if best else float("nan")
        aex = X * (best["b"] * math.ceil(N / Tn) - 1) if best else float("nan")
        res.append({"shape": [M, N, K], "kernel": name, "kernels_seen": sorted(kern), "tile": t,
                    "tile_orientation": best["orient"] if best else "", "fit": best,
                    "weight_B": W, "activation_B": X, "weight_reread_excess_B": wex,
                    "activation_reread_excess_B": aex,
                    "H1_share": wex / (wex + aex) if best and (wex + aex) > 0 else None})
    Path(out_json).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", required=True, help='"M,N,K;M,N,K"')
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--fit", default="", help="ncu raw csv")
    ap.add_argument("--out", default="gemm_sweep_fit.json")
    a = ap.parse_args()
    shapes = [tuple(int(x) for x in s.split(",")) for s in a.shapes.split(";")]
    if a.run:
        run(shapes)
    if a.fit:
        fit(a.fit, shapes, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
