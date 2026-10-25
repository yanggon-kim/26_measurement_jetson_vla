"""T006 steps 2 and 5-6: analytic floors, the ncu <-> nsys join, the per-kernel and per-part roofline, the plot, for one
model x mode. Rewritten 2026-10-26 after the T006 independent review (protocol §5 / §9 schemas, strict join, in-sequence
bytes, per-kernel analytic, unpadded floor FLOPs).

Inputs:
  --nsys     results/<date>_E5_<m>_<mode>_nsys/       kernels.csv: every kernel (launch order, name, grid, block, nsys
                                                       duration, region)
  --ncu      results/<date>_E6_<m>_<mode>/ncu/         <part>_s<start>_raw.csv[.zst]  per-kernel replay, caches flushed
                                                       (cold-L2 bytes, an upper bound)
                                                       <part>_range_raw.csv[.zst]     range replay, caches not flushed
                                                       (in-sequence bytes), when captured
  --analytic results/<date>_E6_dims/raw/<m>_<mode>/    accounting.json (FLOPs per region, parameters touched, calls),
                                                       dims.csv (C3: every GEMM's M, K, N, operand bytes)
  --t005-row raw/rows/<rung>_N<N>.json of the same configuration (CUDA-event region medians)
  --c1       results/<date>_E4g_c1_<m>_<mode>/ (optional): per-kernel speed-of-light verdict and top stall
  torch_kernels.csv next to ncu/ in raw/ (optional; bench/t006_capture.py torchprof): kernel <-> aten op and shapes

Join (protocol §5: launch order + kernel name + grid): the ncu rows of a part, in order, are matched to the nsys kernels
of the same part in inference_0 by base name AND grid AND block, searching ±8 positions around the expected one (one
stream's start order can differ from ncu's serialised order). A row matching by name only is `name_only` and does not
count toward coverage; coverage = strictly joined nsys time ÷ the part's nsys kernel time.

Floor (protocol §9), per part and per ceiling source: bytes_read = weights (once per call of the part, or per denoising
step, when they exceed L2) + the hand-over / KV read + the GEMM operand activations from C3 (inputs read, outputs
written; protocol §9 "activations if not negligible"); FLOPs = the accounting FLOPs (FlopCounterMode), with the GEMM
FLOPs taken from C3's real (unpadded) M when linear_mpad pads rows ("from the real S and P"). Rows peak_source =
measured (T003) and spec (bandwidth only: NVIDIA publishes no dense per-dtype tensor figure for Thor).

Outputs in --out: kernel_roofline.csv, roofline.csv, floor.csv, coverage.json, roofline.png.
"""
import argparse
import csv
import difflib
import glob
import io
import json
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

TENSOR = {"bf16_dst_fp32": "bf16", "fp16_dst_fp16": "fp16", "fp16_dst_fp32": "fp16", "tf32_dst_fp32": "tf32", "int8": "int8",
          "fp64": "fp64", "fp4_dst_fp32": "fp8", "fp4_fp6_fp8_dst_fp16": "fp8", "fp4_fp6_fp8_dst_fp32": "fp8"}
ES = {"float": 4, "c10::BFloat16": 2, "c10::Half": 2, "double": 8, "long int": 8, "int": 4, "bool": 1, "unsigned char": 1,
      "signed char": 1, "short int": 2}


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return 0.0


def read_text(path):
    if path.endswith(".zst"):
        return subprocess.run(["zstd", "-dc", path], capture_output=True, text=True, check=True).stdout
    return open(path).read()


def read_ncu(path):
    rows = list(csv.reader(io.StringIO(read_text(path))))
    if len(rows) < 3:
        return []
    head, units = rows[0], rows[1]
    out = []
    for r in rows[2:]:
        d = dict(zip(head, r))
        d["_units"] = dict(zip(head, units))
        out.append(d)
    return out


def ns(d, key):
    v, u = num(d.get(key, 0)), d["_units"].get(key, "ns")
    return v * {"ns": 1, "nsecond": 1, "us": 1e3, "usecond": 1e3, "ms": 1e6, "msecond": 1e6, "s": 1e9, "second": 1e9}.get(u, 1)


def mhz(d, key):
    """A clock metric in MHz whatever unit ncu reports it in."""
    v, u = num(d.get(key, 0)), (d["_units"].get(key) or "").lower()
    if not v:
        return None
    for unit, f in (("cycle/nsecond", 1e3), ("cycle/usecond", 1), ("cycle/msecond", 1e-3), ("cycle/second", 1e-6),
                    ("ghz", 1e3), ("mhz", 1), ("khz", 1e-3), ("hz", 1e-6)):
        if unit in u:
            return v * f
    return v


def norm(n):
    """The kernel's base function name. nsys and ncu demangle differently (nsys: `(int)128`, `(bool)0`,
    `at::native::<unnamed>::`, `cudnn::cnn::`; ncu: `128`, `0`, `at::<unnamed>::`, `cudnn::`), so the full strings of
    the same kernel differ; the identifier just before the template / argument list does not."""
    n = re.sub(r"^void\s+", "", (n or "").strip()).replace("<unnamed>", "").replace("(anonymous namespace)", "")
    head = re.split(r"[<(]", n, 1)[0]
    return head.split("::")[-1].strip()


def dim3(s):
    t = tuple(int(x) for x in re.findall(r"\d+", s or ""))
    return t + (1,) * (3 - len(t)) if t else ()


def windows(ncu_dir, part):
    ps = glob.glob(os.path.join(ncu_dir, f"{part}_s*_raw.csv")) + glob.glob(os.path.join(ncu_dir, f"{part}_s*_raw.csv.zst"))
    return sorted(ps, key=lambda p: int(re.search(r"_s(\d+)_raw", p).group(1)))


def join(ncu_rows, ns_part, win=8):
    """[(ncu index, nsys index or None, 'strict' | 'name_only' | 'none')]"""
    used, out, j = set(), [], 0
    for i, d in enumerate(ncu_rows):
        name = norm(d.get("Kernel Name", ""))
        g, b = dim3(d.get("Grid Size", "")), dim3(d.get("Block Size", ""))
        cand = [k for k in range(max(0, j - win), min(len(ns_part), j + win + 1)) if k not in used and norm(ns_part[k]["name"]) == name]
        strict = [k for k in cand if dim3(ns_part[k]["grid"]) == g and dim3(ns_part[k]["block"]) == b]
        pick = min(strict, key=lambda k: abs(k - j)) if strict else (min(cand, key=lambda k: abs(k - j)) if cand else None)
        kind = "strict" if strict else ("name_only" if cand else "none")
        if pick is not None:
            used.add(pick)
            j = pick + 1
        else:
            j += 1
        out.append((i, pick, kind))
    return out


# ------------------------------------------------------------------ per-kernel analytic (torch.profiler linkage)
def op_analytic(op, dims_json, types_json):
    """(M, N, K, flops, bytes, note) from an aten op's input shapes; None where the shapes do not define them."""
    try:
        dims, types = json.loads(dims_json or "[]"), json.loads(types_json or "[]")
    except ValueError:
        return None
    es = lambda i: ES.get(types[i], 2) if i < len(types) else 2  # noqa: E731
    size = lambda s: eval("*".join(str(x) for x in s)) if s else 0  # noqa: E731,S307 (integers only)
    o = op.lower()
    try:
        if o in ("aten::mm",) and len(dims) >= 2 and len(dims[0]) == 2:
            (M, K), (_, N) = dims[0], dims[1]
            return M, N, K, 2 * M * N * K, (M * K + K * N + M * N) * es(0), "mm"
        if o == "aten::addmm" and len(dims) >= 3:
            (M, K), (_, N) = dims[1], dims[2]
            return M, N, K, 2 * M * N * K, (M * K + K * N + M * N + N) * es(1), "addmm"
        if o == "aten::bmm" and len(dims) >= 2 and len(dims[0]) == 3:
            (B, M, K), (_, _, N) = dims[0], dims[1]
            return M, N, K, 2 * B * M * N * K, B * (M * K + K * N + M * N) * es(0), f"bmm, batch {B}"
        if o == "aten::linear" and len(dims) >= 2 and dims[0] and dims[1]:
            K = dims[0][-1]
            M = size(dims[0][:-1])
            N = dims[1][0]
            return M, N, K, 2 * M * N * K, (M * K + N * K + M * N) * es(0), "linear"
        if ("scaled_dot_product" in o or "flash_attn" in o) and len(dims) >= 3 and len(dims[0]) == 4:
            q, k = dims[0], dims[1]
            if "flash_attn" in o:                     # (B, L, H, D)
                B, L, H, D = q
                S = k[1]
            else:                                      # (B, H, L, D)
                B, H, L, D = q
                S = k[2]
            return L, S, D, 4 * B * H * L * S * D, (2 * size(q) + 2 * size(k)) * es(0), "attention (QK^T + PV)"
        if dims and any(dims):
            sizes = [size(d) * es(i) for i, d in enumerate(dims) if d]
            return "", "", "", "", sum(sizes) + max(sizes), "inputs + output (= largest input)"
    except (TypeError, ValueError, IndexError):
        return None
    return None


def torch_link(raw_dir, nk):
    """nsys launch_order -> (op, analytic tuple) by aligning torch.profiler's kernel sequence with nsys's (base names)."""
    p = os.path.join(raw_dir, "torch_kernels.csv")
    if not os.path.exists(p):
        return {}, "no torch_kernels.csv"
    tk = list(csv.DictReader(open(p)))
    a_names = [norm(r["kernel"]) for r in tk]
    b_names = [norm(r["name"]) for r in nk]
    sm = difflib.SequenceMatcher(None, a_names, b_names, autojunk=False)
    link = {}
    for blk in sm.get_matching_blocks():
        for t in range(blk.size):
            r = tk[blk.a + t]
            link[nk[blk.b + t]["launch_order"]] = (r["op"], op_analytic(r["op"], r["input_dims"], r["input_types"]))
    return link, f"{len(link)} of {len(nk)} nsys kernels aligned to torch.profiler kernels"


# ------------------------------------------------------------------ floors
def dims_rows(dims, part, nsteps):
    """C3 rows of a part with a multiplicity (the dims hook records the expert's step 0 only, except GR00T's DiT)."""
    if part == "backbone_lm_head":
        return [(r, 1) for r in dims if r["module"].endswith("lm_head")]
    if part == "ae_total":
        dit = [r for r in dims if r["region"].startswith("ae_dit[")]
        return [(r, 1) for r in dit] + [(r, nsteps) for r in dims if r["region"] == "ae_step[0]"] + \
            [(r, 1) for r in dims if r["region"] == "ae_total"]
    if part == "ae_step[0]":                                    # GR00T: the step's DiT is recorded as ae_dit[0]
        return [(r, 1) for r in dims if r["region"] in ("ae_step[0]", "ae_dit[0]")]
    return [(r, 1) for r in dims if r["region"] == part]


def floors(acc, dims, part, l2):
    pb, kv = acc["param_bytes"], sum(acc["kv_bytes"].values())
    steps = sorted(k for k in pb if k.startswith("ae_step["))
    n = len(steps) or int(acc.get("N") or 1)
    calls = int(acc.get("calls", {}).get(part, 1) or 1)
    note = []
    if part == "ae_total":
        if "ae_total" in pb:                                     # GR00T: the head outside the steps, once
            w = sum(pb[s] for s in steps) + (pb["ae_total"] - pb[steps[0]])
        else:                                                    # LeRobot: Σ steps
            w = sum(pb[s] for s in steps)
        kvb = n * kv
        note.append(f"weights once per denoising step ({n})")
        fl_acc = acc["flops"].get("ae_total") or sum(acc["flops"][s] for s in steps)
        mm_acc = sum(v for s in (["ae_total"] if "ae_total" in acc["flop_ops"] else steps)
                     for k, v in acc["flop_ops"].get(s, {}).items() if k in ("aten.mm", "aten.addmm"))
    else:
        w = pb.get(part, 0)
        if calls > 1 and w > l2:
            w *= calls
            note.append(f"weights once per call ({calls} calls)")
        kvb = kv if part.startswith("ae_step") else 0
        fl_acc = acc["flops"].get(part, 0)
        mm_acc = sum(v for k, v in acc["flop_ops"].get(part, {}).items() if k in ("aten.mm", "aten.addmm"))
    rows = dims_rows(dims, part, n)
    if part == "backbone_lm_head" and not w:
        w = sum(int(r["weight_bytes"]) for r, _ in rows)
        note.append("weights from the C3 lm_head row (the accounting hook attributed them to vlm_backbone)")
    act = sum((int(r["act_bytes"]) + int(r["out_bytes"])) * int(r["calls_per_inference"]) * m for r, m in rows)
    gemm_dims = sum(float(r["flops_per_call"]) * int(r["calls_per_inference"]) * m for r, m in rows)
    padded = any("linear_mpad" in str(p) for p in acc.get("preps", []))
    flops = fl_acc
    if padded and gemm_dims and mm_acc and part != "backbone_lm_head" and abs(gemm_dims - mm_acc) > 0.005 * mm_acc:
        flops = fl_acc - mm_acc + gemm_dims
        note.append(f"GEMM FLOPs from C3's real M ({gemm_dims / 1e9:.1f} GFLOP) instead of the padded executed ones "
                    f"({mm_acc / 1e9:.1f} GFLOP)")
    elif padded and part == "backbone_lm_head" and gemm_dims and abs(gemm_dims - fl_acc) > 0.005 * fl_acc:
        flops = gemm_dims
        note.append("FLOPs from C3's real M (the lm_head GEMM is padded by linear_mpad)")
    return {"bytes_weights": int(w), "bytes_kv": int(kvb), "bytes_activations": int(act), "bytes_read": int(w + kvb + act),
            "flops": int(flops), "flops_accounting_executed": int(fl_acc), "dims_rows": len(rows), "notes": "; ".join(note)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--nsys", required=True)
    ap.add_argument("--ncu", required=True)
    ap.add_argument("--analytic", required=True)
    ap.add_argument("--hw", default=os.path.join(REPO, "results", "20261022_E1_peak", "hw_constants.json"))
    ap.add_argument("--t005-row", default="")
    ap.add_argument("--c1", default="", help="results/<date>_E4g_c1_<m>_<mode> (optional)")
    ap.add_argument("--rung", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from t007_timeline import kclass
    hw = json.load(open(a.hw))
    bw = float(hw["measured_bw_GBps"]["best"]["value"])
    bw_spec = float(hw["memory"]["spec_GBps"])
    l2 = int(hw["l2_bytes"])
    tf = {k: v["value"] for k, v in hw["measured_tflops"].items() if isinstance(v, dict) and "value" in v}
    region_ms = {}
    if a.t005_row:
        row = json.load(open(a.t005_row))
        st = (row.get("by_N") or {}).get(str(row["N"]), {}).get("regions", {})
        for k in ("vision", "vlm_prefix", "ae_total", "backbone_lm_head", "vlm_backbone"):
            if k in st:
                region_ms[k] = st[k]["median_ms"]
        if "ae_step_individual[0]" in st:
            region_ms["ae_step[0]"] = st["ae_step_individual[0]"]["median_ms"]
        a.rung = a.rung or row.get("rung", "")
    nk = [r for r in csv.DictReader(open(os.path.join(a.nsys, "kernels.csv"))) if r["inference"] == "inference_0"]
    link, link_note = torch_link(os.path.join(os.path.dirname(os.path.abspath(a.ncu)), "raw"), nk)
    c1 = {}
    if a.c1 and os.path.exists(os.path.join(a.c1, "ncu", "kernels.csv")):
        for r in csv.DictReader(open(os.path.join(a.c1, "ncu", "kernels.csv"))):
            c1[(r["part"], int(r.get("launch_idx") or r["idx"]))] = r
    acc = json.load(open(os.path.join(a.analytic, "accounting.json")))
    dims = list(csv.DictReader(open(os.path.join(a.analytic, "dims.csv")))) if os.path.exists(os.path.join(a.analytic, "dims.csv")) else []
    parts = sorted({re.sub(r"_s\d+_raw\.csv(\.zst)?$", "", os.path.basename(p))
                    for p in glob.glob(os.path.join(a.ncu, "*_s*_raw.csv*"))})
    krows, prow, cov, frows = [], [], {}, []
    for part in parts:
        nreg = {"ae_step0": "ae_step[0]"}.get(part, part)
        ncu_rows = [d for p in windows(a.ncu, part) for d in read_ncu(p)]
        ns_part = [r for r in nk if r["region"] == nreg]
        tot_ns = sum(int(r["duration_ns"]) for r in ns_part)
        pairs = join(ncu_rows, ns_part)
        kinds = {"strict": 0, "name_only": 0, "none": 0}
        joined_ns, clocks = 0, []
        for i, j, kind in pairs:
            d = ncu_rows[i]
            kinds[kind] += 1
            nsr = ns_part[j] if j is not None else None
            dur = int(nsr["duration_ns"]) if (nsr is not None and kind == "strict") else None
            rd = num(d.get("lts__d_sectors_fill_sysmem.sum")) * 32
            wr = num(d.get("lts__t_sectors_aperture_sysmem_op_write.sum")) * 32
            tens = {TENSOR[k[len("sm__ops_path_tensor_src_"):-4]]: num(v) for k, v in d.items()
                    if k.startswith("sm__ops_path_tensor_src_") and k.endswith(".sum") and k[len("sm__ops_path_tensor_src_"):-4] in TENSOR}
            tsum = sum(tens.values())
            simt = sum(num(d.get(f"smsp__sass_thread_inst_executed_op_{p}{o}_pred_on.sum", 0)) * (2 if o == "fma" else 1)
                       for p in "fhd" for o in ("add", "mul", "fma"))
            fl = tsum + simt
            dt = max(tens, key=tens.get) if tsum > 0 else "fp32_simt"
            peak = tf.get(dt if dt != "fp32_simt" else "fp32", tf.get("fp32"))
            t = (dur or 0) * 1e-9
            byt = rd + wr
            inten = (fl / byt) if byt else None
            ridge = peak * 1e12 / (bw * 1e9)
            ck = mhz(d, "sm__cycles_elapsed.avg.per_second")
            if ck:
                clocks.append(ck)
            if dur:
                joined_ns += dur
            lo = nsr["launch_order"] if nsr is not None else ""
            op, an = link.get(lo, ("", None)) if lo != "" else ("", None)
            cr = c1.get((nreg, i))
            if cr is not None and norm(cr.get("kernel", "")) != norm(d.get("Kernel Name", "")):
                cr = None
            krows.append({"model": a.model, "mode": a.mode, "part": nreg, "launch_idx": lo, "ncu_idx": i, "join": kind,
                          "kernel": d.get("Kernel Name", "")[:200], "grid": nsr["grid"] if nsr else "", "block": nsr["block"] if nsr else "",
                          "op_class": kclass(d.get("Kernel Name", "")), "aten_op": op,
                          "M": an[0] if an else "", "N": an[1] if an else "", "K": an[2] if an else "",
                          "dtype": dt, "flops_analytic": an[3] if an else "n/a", "flops_ncu": int(fl),
                          "bytes_analytic": an[4] if an else "n/a", "bytes_ncu": int(byt), "ms_nsys": round(dur / 1e6, 6) if dur else "",
                          "intensity_ncu": round(inten, 3) if inten is not None else "",
                          "achieved_TFLOPps": round(fl / t / 1e12, 4) if t else "", "achieved_GBps": round(byt / t / 1e9, 3) if t else "",
                          "pct_peak_compute": round(fl / t / 1e12 / peak * 100, 2) if t else "",
                          "pct_peak_bw": round(byt / t / 1e9 / bw * 100, 2) if t else "",
                          "sol_bound": cr["verdict"] if cr else "n/a: no C1 row", "stall_top": cr["top_stall"] if cr else "n/a: no C1 row",
                          "bound_by_intensity": ("compute" if inten >= ridge else "memory") if inten is not None
                          else ("n/a: no FLOPs counted" if byt else "n/a: no DRAM bytes"),
                          "over_100pct": "yes (cold-L2 bytes or tile-padded FLOPs exceed the ceiling)" if t and
                          (byt / t / 1e9 > bw or fl / t / 1e12 > peak) else "",
                          "analytic_note": an[5] if an else ("graph-launched or fused: no shapes" if op else "n/a: not linked"),
                          "dram_read_B": int(rd), "dram_write_B": int(wr), "tensor_flops": int(tsum), "simt_flops": int(simt),
                          "ceiling_TFLOPps": peak, "ceiling_GBps": bw, "ridge_flop_per_B": round(ridge, 2),
                          "sm_clock_MHz": round(ck, 1) if ck else "",
                          "tensor_pipe_pct": num(d.get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active")),
                          "ncu_duration_ns": round(ns(d, "gpu__time_duration.sum")),
                          "sm_pct": num(d.get("sm__throughput.avg.pct_of_peak_sustained_elapsed")),
                          "onchip_mem_pct": num(d.get("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")),
                          "occupancy_pct": num(d.get("sm__warps_active.avg.pct_of_peak_sustained_active"))})
        cov[nreg] = {"nsys_kernels": len(ns_part), "ncu_kernels": len(ncu_rows), "join_strict": kinds["strict"],
                     "join_name_only": kinds["name_only"], "join_none": kinds["none"],
                     "name_mismatches": kinds["none"], "nsys_kernel_ms": tot_ns / 1e6, "joined_kernel_ms": joined_ns / 1e6,
                     "coverage_pct": round(joined_ns / tot_ns * 100, 2) if tot_ns else 0.0,
                     "sm_clock_MHz": {"median": round(statistics.median(clocks), 1), "min": round(min(clocks), 1),
                                      "max": round(max(clocks), 1), "n": len(clocks)} if clocks else "n/a: clock metric not captured"}
        ks = [k for k in krows if k["part"] == nreg]
        F = sum(k["flops_ncu"] for k in ks)
        B = sum(k["bytes_ncu"] for k in ks)
        dts = {}
        for k in ks:
            dts[k["dtype"]] = dts.get(k["dtype"], 0) + k["flops_ncu"]
        dom = max(dts, key=dts.get) if dts else "bf16"
        peak = tf.get(dom if dom != "fp32_simt" else "fp32", tf.get("bf16"))
        ridge = peak * 1e3 / bw

        def prow_add(source, basis, flops, byt, ms, tsrc, n_k, cv):
            if not ms:
                return
            prow.append({"model": a.model, "mode": a.mode, "rung": a.rung, "part": nreg, "flops": int(flops), "bytes": int(byt),
                         "ms": round(ms, 4), "intensity": round(flops / byt, 3) if byt else "",
                         "achieved_TFLOPps": round(flops / ms / 1e9, 4), "achieved_GBps": round(byt / ms / 1e6, 3),
                         "ceiling_TFLOPps": peak, "ceiling_GBps": bw, "ridge": round(ridge, 2),
                         "pct_peak_compute": round(flops / ms / 1e9 / peak * 100, 2), "pct_peak_bw": round(byt / ms / 1e6 / bw * 100, 2),
                         "bound": ("compute" if byt and flops / byt >= ridge else "memory"), "source": source, "bytes_basis": basis,
                         "time_source": tsrc, "n_kernels": n_k, "coverage": cv})
        cvp = cov[nreg]["coverage_pct"]
        for tsrc, ms in (("nsys_kernel_sum", tot_ns / 1e6), ("region_cuda_event", region_ms.get(nreg))):
            prow_add("ncu", "per-kernel replay, caches flushed (cold-L2 bytes: an upper bound)", F, B, ms, tsrc, len(ks), cvp)
        rr = [d for p in (glob.glob(os.path.join(a.ncu, f"{part}_range_raw.csv")) + glob.glob(os.path.join(a.ncu, f"{part}_range_raw.csv.zst")))
              for d in read_ncu(p)]
        if rr:
            Fr = sum(sum(num(v) for k, v in d.items() if k.startswith("sm__ops_path_tensor_src_") and k.endswith(".sum"))
                     + sum(num(d.get(f"smsp__sass_thread_inst_executed_op_{p}{o}_pred_on.sum", 0)) * (2 if o == "fma" else 1)
                           for p in "fhd" for o in ("add", "mul", "fma")) for d in rr)
            Br = sum((num(d.get("lts__d_sectors_fill_sysmem.sum")) + num(d.get("lts__t_sectors_aperture_sysmem_op_write.sum"))) * 32 for d in rr)
            cov[nreg]["range_replay"] = {"results": len(rr), "dram_bytes": int(Br), "flops": int(Fr),
                                         "range_duration_ms": round(sum(ns(d, "gpu__time_duration.sum") for d in rr) / 1e6, 4),
                                         "bytes_vs_per_kernel_replay": round(Br / B, 4) if B else ""}
            for tsrc, ms in (("nsys_kernel_sum", tot_ns / 1e6), ("region_cuda_event", region_ms.get(nreg))):
                prow_add("ncu", "range replay, caches not flushed (in-sequence bytes)", Fr, Br, ms, tsrc, len(ks), cvp)
        fl_ = floors(acc, dims, nreg, l2)
        for tsrc, ms in (("nsys_kernel_sum", tot_ns / 1e6), ("region_cuda_event", region_ms.get(nreg))):
            prow_add("analytic", "weights per call / step + KV + GEMM activations (C3)", fl_["flops"], fl_["bytes_read"], ms, tsrc,
                     len(ks), cvp)
    # floor.csv: every part the accounting knows, not only the profiled ones (protocol §9: one ae_step and ae_total)
    fparts = [p for p in ("vision", "vlm_prefix", "backbone_lm_head", "ae_step[0]", "ae_total")
              if p in acc["flops"] or (p == "ae_total" and any(k.startswith("ae_step[") for k in acc["flops"]))]
    for part in fparts:
        fl_ = floors(acc, dims, part, l2)
        by = acc["param_bytes_by_dtype"].get(part) or acc["param_bytes_by_dtype"].get("ae_step[0]") or {}
        dom = max(by, key=by.get) if by else "torch.bfloat16"
        key = {"torch.bfloat16": "bf16", "torch.float16": "fp16", "torch.float32": "fp32"}.get(dom, "bf16")
        if key == "fp32" and a.mode != "eager_fp32" and "tf32" in tf:
            key = "tf32"
        if a.mode == "eager_fp32":
            key = "fp32"
        peak = tf[key]
        ms = region_ms.get(part)
        if part == "ae_total" and ms is None and a.t005_row:
            st = (json.load(open(a.t005_row)).get("by_N") or {})
            ms = next((v["regions"]["ae_total"]["median_ms"] for v in st.values() if "ae_total" in v.get("regions", {})), None)
        for src, pbw, ptf in (("measured", bw, peak), ("spec", bw_spec, None)):
            fm = fl_["bytes_read"] / (pbw * 1e9) * 1e3
            fc = fl_["flops"] / (ptf * 1e12) * 1e3 if ptf else None
            f_ = max(fm, fc) if fc is not None else fm
            frows.append({"model": a.model, "mode": a.mode, "region": part, "bytes_read": fl_["bytes_read"],
                          "bytes_weights": fl_["bytes_weights"], "bytes_kv": fl_["bytes_kv"], "bytes_activations": fl_["bytes_activations"],
                          "flops": fl_["flops"], "flops_accounting_executed": fl_["flops_accounting_executed"],
                          "peak_GBps": pbw, "peak_TFLOPps": ptf if ptf else "n/a: NVIDIA publishes no dense per-dtype tensor figure",
                          "compute_dtype": key, "floor_mem_ms": round(fm, 4), "floor_compute_ms": round(fc, 4) if fc is not None else "n/a",
                          "floor_ms": round(f_, 4), "bound": ("memory" if fc is None or fm >= fc else "compute"),
                          "floor_mem_weights_only_ms": round(fl_["bytes_weights"] / (pbw * 1e9) * 1e3, 4),
                          "peak_source": src + (" (bandwidth only)" if src == "spec" else ""),
                          "measured_ms": round(ms, 4) if ms else "n/a: region not instrumented in this configuration",
                          "gap_ratio": round(ms / f_, 3) if ms else "n/a", "notes": fl_["notes"]})
    os.makedirs(a.out, exist_ok=True)
    for name, rows in (("kernel_roofline.csv", krows), ("roofline.csv", prow), ("floor.csv", frows)):
        if rows:
            keys = []
            for r in rows:
                keys += [k for k in r if k not in keys]
            with open(os.path.join(a.out, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
    cov["_torch_profiler_link"] = link_note
    json.dump(cov, open(os.path.join(a.out, "coverage.json"), "w"), indent=1)
    plot(a, krows, prow, tf, bw)
    print(json.dumps({k: (v["coverage_pct"], v["join_strict"], v["join_name_only"], v["join_none"]) if isinstance(v, dict) else v
                      for k, v in cov.items()}, indent=1))


def plot(a, krows, prow, tf, bw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = [10 ** (i / 20) for i in range(-40, 101)]
    for dt, ls in (("bf16", "-"), ("tf32", "--"), ("fp32", ":")):
        if dt in tf:
            ax.plot(xs, [min(tf[dt], x * bw / 1e3) for x in xs], ls, color="k", lw=1, label=f"{dt} roof {tf[dt]} TFLOP/s")
    col = {"vision": "tab:blue", "vlm_prefix": "tab:orange", "ae_step[0]": "tab:green", "ae_total": "tab:green",
           "backbone_lm_head": "tab:gray"}
    for k in krows:
        if k["intensity_ncu"] and k["achieved_TFLOPps"]:
            ax.scatter(k["intensity_ncu"], k["achieved_TFLOPps"], s=4, alpha=0.3, color=col.get(k["part"], "tab:purple"))
    for p in prow:
        if p["time_source"] == "region_cuda_event" and p["intensity"] and p["source"] == "ncu":
            mk = "*" if p["bytes_basis"].startswith("per-kernel") else "D"
            ax.scatter(p["intensity"], p["achieved_TFLOPps"], s=80, marker=mk, color=col.get(p["part"], "tab:purple"),
                       edgecolor="k", label=f"{p['part']} ({'cold-L2' if mk == '*' else 'in-sequence'} bytes)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("intensity (FLOP / DRAM byte, ncu)")
    ax.set_ylabel("achieved TFLOP/s (nsys time)")
    ax.set_title(f"{a.model} {a.mode}: kernels (dots), parts (stars: cold-L2 bytes; diamonds: in-sequence bytes)")
    ax.legend(fontsize=6)
    fig.savefig(os.path.join(a.out, "roofline.png"), dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
