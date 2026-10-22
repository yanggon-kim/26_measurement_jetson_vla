"""T004 step 2 (protocol §2): validate the vision / vlm_prefix split of one model.

1. nsys (bench/nsys_regions_target.py capture, `nsys stats` CSVs): the GPU-projected intervals of every `vision` and
   every `vlm_prefix` range; every kernel of cuda_gpu_trace assigned to the ranges whose projected interval contains
   it; the two kernel sets must be disjoint (no kernel in both) and the intervals must not overlap.
2. Timing (the model's E3 latency_samples.csv at its default N, CUDA events): (vision + vlm_prefix) ÷ vlm_backbone per
   iteration, median within 2 % of 1 (GR00T, where vlm_backbone is the enclosing call). For the LeRobot models the
   regions are sequential calls with no enclosing wrapper; only check 1 applies.

python bench/region_split_check.py --nsys-prefix results/<run>/nsys/<model> --e3 results/<E3 dir> --model <m> --out results/<run>
"""
import argparse
import csv
import json
import os
import statistics


def load_csv(p):
    return list(csv.DictReader(open(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsys-prefix", required=True)
    ap.add_argument("--e3", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    proj = load_csv(a.nsys_prefix + "_nvtx_gpu_proj_trace.csv")
    kern = load_csv(a.nsys_prefix + "_cuda_gpu_trace.csv")
    iv = {}
    for r in proj:
        name = r["Name"].lstrip(":")
        if name in ("vision", "vlm_prefix", "vlm_backbone", "backbone_lm_head") and r["Projected Start (ns)"]:
            s = int(r["Projected Start (ns)"])
            iv.setdefault(name, []).append((s, s + int(r["Projected Duration (ns)"])))
    kernels = [(int(k["Start (ns)"]), int(k["Start (ns)"]) + int(k["Duration (ns)"]), k["Name"]) for k in kern
               if k.get("Name") and k.get("GrdX")]
    sets = {}
    for reg in ("vision", "vlm_prefix", "vlm_backbone", "backbone_lm_head"):
        sets[reg] = {i for i, (ks, ke, _) in enumerate(kernels) if any(s <= ks and ke <= e for s, e in iv.get(reg, []))}
    both = sets["vision"] & sets["vlm_prefix"]
    overlap = [(a1, b1) for a1 in iv.get("vision", []) for b1 in iv.get("vlm_prefix", []) if a1[0] < b1[1] and b1[0] < a1[1]]
    ktime = {reg: sum(kernels[i][1] - kernels[i][0] for i in sets[reg]) / 1e6 for reg in sets}
    res = {"model": a.model, "nsys_ranges": {k: len(v) for k, v in iv.items()},
           "kernels_in": {k: len(v) for k, v in sets.items()}, "kernels_in_both_vision_and_prefix": len(both),
           "overlapping_intervals": len(overlap), "kernel_time_ms_in_capture": ktime,
           "disjoint": len(both) == 0 and len(overlap) == 0}
    if sets["vlm_backbone"]:
        rest = sets["vlm_backbone"] - sets["vision"] - sets["vlm_prefix"] - sets["backbone_lm_head"]
        res["kernels_in_backbone_not_in_vision_prefix_or_lm_head"] = len(rest)
        res["backbone_rest_kernel_time_ms"] = sum(kernels[i][1] - kernels[i][0] for i in rest) / 1e6
        res["lm_head_kernel_time_ms_in_capture"] = ktime["backbone_lm_head"]
        rows = [r for r in load_csv(os.path.join(a.e3, "latency_samples.csv")) if int(r["N"]) == a.n]
        by = {}
        for r in rows:
            by.setdefault(int(r["iter"]), {})[r["region"]] = float(r["ms"])
        ratios = [(d["vision"] + d["vlm_prefix"]) / d["vlm_backbone"] for d in by.values()
                  if all(k in d for k in ("vision", "vlm_prefix", "vlm_backbone"))]
        ratios_lm = [(d["vision"] + d["vlm_prefix"] + d.get("backbone_lm_head", 0)) / d["vlm_backbone"] for d in by.values()
                     if all(k in d for k in ("vision", "vlm_prefix", "vlm_backbone"))]
        if any("backbone_lm_head" in d for d in by.values()):
            res["sum_with_lm_head_ratio_median"] = statistics.median(ratios_lm)
            res["backbone_lm_head_ms_median"] = statistics.median(d["backbone_lm_head"] for d in by.values() if "backbone_lm_head" in d)
        res["sum_ratio_median"] = statistics.median(ratios)
        res["sum_ratio_p5_p95"] = [sorted(ratios)[int(.05 * (len(ratios) - 1))], sorted(ratios)[int(.95 * (len(ratios) - 1))]]
        res["sum_within_2pct"] = abs(res["sum_ratio_median"] - 1) <= 0.02
        res["sum_n"] = len(ratios)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"split_check_{a.model}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
