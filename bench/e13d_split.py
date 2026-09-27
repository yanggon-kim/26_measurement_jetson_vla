"""T016 1d (EV-02c) H1 / H2 split of the prefix GEMMs' excess reads, from the standalone sweeps.

Supersedes the first fit in bench/e13d_gemm_sweep.py (fills = a W Mt + b X Nt over ALL sweep points),
which returned negative coefficients (relative RMSE 0.11-0.34, H1 share > 1): points of one sweep ran
DIFFERENT cuBLAS kernels (tile shapes change with M and N), so one tile model cannot describe them.

Model, per base shape, over the sweep points that ran the SAME kernel as the model's own shape:
    excess = fills - W - X = a * W * (Mt - 1)  +  b * X * (Nt - 1)
      W = N K 2 (weight), X = M K 2 (activation), Mt = ceil(M / Tm), Nt = ceil(N / Tn),
      (Tm, Tn) the kernel's CTA tile from its name, both orientations tried;
      a, b >= 0 (non-negative least squares on two regressors) -- a = weight re-reads per extra M tile,
      b = activation re-reads per extra N tile.
At the model's shape: weight excess = a W (Mt - 1), activation excess = b X (Nt - 1); the residual
(measured excess - both) is reported as 'neither'. The fit's relative RMSE and point count are reported;
a fit with < 3 same-kernel points is flagged and its split not used.
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path


def nnls2(A, B, Y):
    best = None
    cands = []
    saa = sum(a * a for a in A); sbb = sum(b * b for b in B); sab = sum(a * b for a, b in zip(A, B))
    say = sum(a * y for a, y in zip(A, Y)); sby = sum(b * y for b, y in zip(B, Y))
    det = saa * sbb - sab * sab
    if det:
        cands.append(((say * sbb - sby * sab) / det, (sby * saa - say * sab) / det))
    if saa:
        cands.append((max(0.0, say / saa), 0.0))
    if sbb:
        cands.append((0.0, max(0.0, sby / sbb)))
    for a, b in cands:
        if a < 0 or b < 0:
            continue
        rss = sum((y - a * x - b * z) ** 2 for x, z, y in zip(A, B, Y))
        if best is None or rss < best[2]:
            best = (a, b, rss)
    return best


def main() -> int:
    d = Path(sys.argv[1])
    raw = list(csv.reader(open(d / "ncu" / "gemm_sweep_raw.csv")))
    h, u, data = raw[0], raw[1], raw[2:]
    c = {x: i for i, x in enumerate(h)}
    nv = next(i for i, x in enumerate(h) if "Push/Pop_Range" in x)
    pts = {}
    for r in data:
        m = re.search(r"g(\d+)_M(\d+)_N(\d+)_K(\d+)", r[nv])
        if not m:
            continue
        key = tuple(int(x) for x in m.groups())
        f = (float(r[c["lts__d_sectors_fill_device.sum"]].replace(",", "") or 0) +
             float(r[c["lts__d_sectors_fill_sysmem.sum"]].replace(",", "") or 0)) * 32
        pts.setdefault(key, []).append((r[c["Kernel Name"]], f))
    excess_by_shape = {(int(r["M"]), int(r["N"]), int(r["K"])): r
                       for r in csv.DictReader(open(d / "gemm_excess_by_shape.csv"))}
    out = []
    for gi in sorted({k[0] for k in pts}):
        P = {k[1:]: v for k, v in pts.items() if k[0] == gi}
        base = max(P, key=lambda s: sum(1 for t in P if t[2] == s[2] and (t[0] == s[0] or t[1] == s[1])))
        # the model's own shape is the one with 1.0x scale on both axes: the point present in both sweeps
        Ms = sorted({s[0] for s in P}); Ns = sorted({s[1] for s in P})
        base = next(s for s in P if [t for t in P if t[1] == s[1]] and len([t for t in P if t[0] == s[0]]) > 1
                    and len([t for t in P if t[1] == s[1]]) > 1)
        M0, N0, K = base
        kname = P[base][0][0]
        tm = re.search(r"_(\d+)x(\d+)_", kname)
        tile = (int(tm.group(1)), int(tm.group(2))) if tm else (128, 128)
        same = {s: sum(f for _, f in v) / len(v) for s, v in P.items() if all(n == kname for n, _ in v)}
        best = None
        for orient in ("MxN", "NxM"):
            Tm, Tn = tile if orient == "MxN" else (tile[1], tile[0])
            A, B, Y = [], [], []
            for (m, n, k), f in same.items():
                W, X = n * k * 2, m * k * 2
                A.append(W * (math.ceil(m / Tm) - 1)); B.append(X * (math.ceil(n / Tn) - 1)); Y.append(f - W - X)
            r = nnls2(A, B, Y)
            if r is None:
                continue
            mean = sum(abs(y) for y in Y) / len(Y) or 1.0
            rel = math.sqrt(r[2] / len(Y)) / mean
            if best is None or rel < best["rel_rmse"]:
                best = {"orient": orient, "Tm": Tm, "Tn": Tn, "a": r[0], "b": r[1], "rel_rmse": rel, "points": len(Y)}
        W0, X0 = N0 * K * 2, M0 * K * 2
        f0 = same.get(base)
        meas = (f0 - W0 - X0) if f0 is not None else None
        wex = best["a"] * W0 * (math.ceil(M0 / best["Tm"]) - 1)
        aex = best["b"] * X0 * (math.ceil(N0 / best["Tn"]) - 1)
        usable = best["points"] >= 3
        inmodel = excess_by_shape.get(base, {})
        out.append({"shape": [M0, N0, K], "kernel": kname, "tile_from_name": tile, "fit": best,
                    "same_kernel_points": sorted(same), "all_points_kernels": {str(s): v[0][0][:60] for s, v in P.items()},
                    "standalone_fills_per_call_B": f0, "standalone_excess_per_call_B": meas,
                    "weight_reread_excess_B": wex, "activation_reread_excess_B": aex,
                    "neither_B": (meas - wex - aex) if meas is not None else None,
                    "H1_share_of_modelled": wex / (wex + aex) if (wex + aex) > 0 else None,
                    "usable": usable,
                    "in_model_launches": int(inmodel.get("launches", 0) or 0),
                    "in_model_excess_B": int(inmodel.get("excess_B", 0) or 0)})
    tot = sum(o["in_model_excess_B"] for o in out if o["usable"] and o["H1_share_of_modelled"] is not None)
    h1 = sum(o["in_model_excess_B"] * o["H1_share_of_modelled"] for o in out
             if o["usable"] and o["H1_share_of_modelled"] is not None)
    summary = {"shapes": out, "in_model_excess_covered_B": tot,
               "H1_share_weighted_by_in_model_excess": h1 / tot if tot else None}
    (d / "gemm_split.json").write_text(json.dumps(summary, indent=2) + "\n")
    for o in out:
        f = o["fit"]
        print(f"{o['shape']} {o['kernel'][:40]} tile {o['tile_from_name']} {f['orient']} a={f['a']:.3f} b={f['b']:.4f} "
              f"rel={f['rel_rmse']:.3f} pts={f['points']}  Wex={o['weight_reread_excess_B']/1e6:.1f}MB "
              f"Aex={o['activation_reread_excess_B']/1e6:.1f}MB meas={((o['standalone_excess_per_call_B'] or 0))/1e6:.1f}MB "
              f"H1={o['H1_share_of_modelled']} usable={o['usable']}")
    print("H1 share weighted by in-model excess:", summary["H1_share_weighted_by_in_model_excess"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
