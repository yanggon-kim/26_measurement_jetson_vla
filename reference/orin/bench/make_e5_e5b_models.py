"""reports/E5_E5b_models.md -- the one cross-model table T008 and T009 both name.

T008's acceptance fixes the columns:
  model, S, ms/step eager, ms/step compiled, floor, gap ratio, GPU idle % in ae_step,
  region-weighted SM occupancy, top stall reason and share, LSU load share, dominant kernel

This JOINS committed files and computes nothing new. Every cell names its source in the
"sources" section, and a cell with no source is left EMPTY and listed under "not measured"
rather than filled from a neighbouring configuration. Sources, per column:

  ms/step        Part B ladder rows (UNPROFILED ae_total_ms / N) -- eager rung and the model's
                 best rung. nsys-profiled E5 walls are ~1.5x inflated and are not used here.
  floor, gap     the matched-rung E6 (results/*_E6_rungs/<model>_<rung>/roofline.csv): floor
                 from bytes counted AFTER the rung's casts; gap = ms/step / floor, per rung.
  GPU idle       E5's corrected idle fraction (bench/make_e5_report.corrected_frac), which
                 charges nsys overhead to idle against the matched unprofiled reference.
  SM occupancy   C1's c1_bounds.csv occupancy_pct for the expert region.
  stall / LSU /  E5b stall_attribution.csv, duration-weighted over that configuration's
  dominant       targets (the targets cover >= 60 % of captured expert kernel time, §5).

Mode names differ between sources and are mapped explicitly (MODES below), because "best"
means C1's --compile-mode reduce-overhead in C1/E5b but the ladder's best RUNG in Part B.
That mismatch is stated in the report rather than hidden: the compiled columns are not all
the same compiled configuration.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
import make_e5_report as E5R    # noqa: E402

LABEL = {"pi0": "π0", "pi05": "π0.5", "smolvla": "SmolVLA", "groot_n17": "GR00T N1.7",
         "groot_n16": "GR00T N1.6", "groot_n15": "GR00T N1.5"}
# model -> (ladder root, N, eager rung, best rung, S, S source, expert region in C1/E5b)
MODELS = {
    "pi0": ("results/20260922_E4c_pi0_opt_ladder", 10, "eager_tf32", "bf16_both", 51,
            "E3 meta tokens_detail.suffix_tokens_S", "ae_step_0"),
    "pi05": ("results/20260922_E4c_pi05_opt_ladder", 10, "eager_tf32", "bf16_both_graphs", 50,
             "E6 floor_inputs.json tokens", "ae_step_0"),
    "smolvla": ("results/20260922_E4c_smolvla_opt_ladder", 10, "eager_tf32", "vlm_graphs", 50,
                "E6 floor_inputs.json tokens", "ae_step_0"),
    "groot_n17": ("results/20260923_E4d_groot_n17_ladder", 4, "eager_tf32", "vlm_compile", 41,
                  "T009 (loaded config action_horizon = 40; see report)", "ae_total"),
    "groot_n16": ("results/20260923_E4d_groot_n16_ladder", 4, "eager_tf32",
                  "all_compile_autotune", None, "", "ae_total"),
    "groot_n15": ("results/20260923_E4d_groot_n15_ladder", 4, "eager_tf32",
                  "all_compile_autotune", None, "", "ae_total"),
}


def ladder_ms(root: str, rung: str, n: int):
    for pat in (f"{root}/raw/rows/{rung}_N{n}.json", f"{root}/raw/rows/{rung}.json"):
        for f in glob.glob(pat):
            d = json.load(open(f))
            v = d.get("ae_total_ms")
            if isinstance(v, (int, float)) and v > 0 and not d.get("error"):
                return v / d.get("N", n), f
    return None, ""


def e6_row(model: str, rung: str):
    for f in sorted(glob.glob(f"results/*_E6_rungs/{model}_{rung}/roofline.csv")):
        for r in csv.DictReader(open(f)):
            if r["region_or_kernel"] == "ae_step" and r["source"] == "analytic":
                return r, f
    return None, ""


def e5_idle(model: str, mode: str):
    runs = E5R.runs(f"results/*_E5_{model}_{mode}*")
    # Prefer the N=10 capture (Part B's unprofiled reference is N=10); GR00T's E5 runs are at
    # its own N=4, and skipping them reported "no E5 capture" when one exists -- the real
    # reason the cell is empty is the one corrected_frac gives.
    runs.sort(key=lambda r: str(r["n"]) != "10")
    for run in runs:
        for row in run["rows"]:
            if row.get("region") in ("ae_step", "ae_total"):
                c, why = E5R.corrected_frac(model, mode, run["n"] or 10, row)
                return (c, why, run["dir"])
    return ("", "no E5 capture", "")


def c1_occ(model: str, mode: str, region: str):
    f = "results/20260923_E4g_c1_summary/c1_bounds.csv"
    for r in csv.DictReader(open(f)):
        if r["model"] == model and r["mode"] == mode and r["region"] == region:
            return r["occupancy_pct"], f
    return "", ""


def e5b(model: str, mode: str):
    files = sorted(glob.glob(f"results/*_E5b_{model}_{mode}/stall_attribution.csv"))
    if not files:
        return None
    rows = list(csv.DictReader(open(files[-1])))
    dur = {}
    for r in rows:
        dur[r["kernel"]] = float(r["duration_us"] or 0)
    tot = sum(dur.values()) or 1.0
    stall = defaultdict(float)
    lsu = {}
    for r in rows:
        if r["stall_reason"] == "selected":
            continue
        stall[r["stall_reason"]] += float(r["share_of_waiting_pct"] or 0) * dur[r["kernel"]] / tot
        if r.get("pipe_lsu_pct"):
            lsu[r["kernel"]] = float(r["pipe_lsu_pct"])
    top = max(stall.items(), key=lambda kv: kv[1]) if stall else ("", 0.0)
    lsu_w = sum(lsu[k] * dur[k] for k in lsu) / tot if lsu else None
    dom = max(dur.items(), key=lambda kv: kv[1])[0]
    return {"top": top[0], "top_share": round(top[1], 1),
            "lsu": round(lsu_w, 1) if lsu_w is not None else "",
            "dominant": dom, "src": files[-1]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="reports/E5_E5b_models.md")
    ap.add_argument("--csv", default="reports/E5_E5b_models.csv")
    args = ap.parse_args()

    rows, sources, missing = [], [], []
    for m, (root, n, er, br, S, s_src, reg) in MODELS.items():
        eg, ef = ladder_ms(root, er, n)
        bg, bf = ladder_ms(root, br, n)
        if eg is None and bg is None:
            continue
        e6e, e6ef = e6_row(m, er)
        e6b, e6bf = e6_row(m, br)
        floor = e6e["floor_ms"] if e6e else (e6b["floor_ms"] if e6b else "")
        floor_b = e6b["floor_ms"] if e6b else ""
        idle_e, why_e, idle_src = e5_idle(m, "eager")
        idle_b, why_b, _ = e5_idle(m, "best")
        occ_e, occ_src = c1_occ(m, "eager", reg)
        occ_b, _ = c1_occ(m, "best", reg)
        st_e, st_b = e5b(m, "eager"), e5b(m, "best")
        row = {
            "model": LABEL[m], "S": S if S is not None else "",
            "ms_per_step_eager": round(eg, 3) if eg else "",
            "eager_rung": er,
            "ms_per_step_compiled": round(bg, 3) if bg else "",
            "compiled_rung": br,
            "floor_ms_eager_bytes": floor, "floor_ms_compiled_bytes": floor_b,
            "gap_eager": round(eg / float(floor), 2) if (eg and floor) else "",
            "gap_compiled": round(bg / float(floor_b), 2) if (bg and floor_b) else "",
            "pct_peak_bw_eager": e6e["pct_peak_bw"] if e6e else "",
            "pct_peak_bw_compiled": e6b["pct_peak_bw"] if e6b else "",
            "gpu_idle_pct_eager": round(100 * float(idle_e), 1) if idle_e else "",
            "gpu_idle_pct_best": round(100 * float(idle_b), 1) if idle_b else "",
            "sm_occupancy_pct_eager": occ_e, "sm_occupancy_pct_best": occ_b,
            "top_stall_eager": f"{st_e['top']} {st_e['top_share']}%" if st_e else "",
            "top_stall_best": f"{st_b['top']} {st_b['top_share']}%" if st_b else "",
            "lsu_share_pct_eager": st_e["lsu"] if st_e else "",
            "lsu_share_pct_best": st_b["lsu"] if st_b else "",
            "dominant_kernel_eager": st_e["dominant"] if st_e else "",
            "dominant_kernel_best": st_b["dominant"] if st_b else "",
        }
        rows.append(row)
        sources.append((LABEL[m], {"ladder eager": ef, "ladder compiled": bf,
                                   "E6 eager": e6ef, "E6 compiled": e6bf, "E5": idle_src,
                                   "C1": occ_src, "E5b eager": st_e["src"] if st_e else "",
                                   "E5b best": st_b["src"] if st_b else "", "S": s_src}))
        for k, v in row.items():
            if v in ("", None):
                why = ""
                if k == "gpu_idle_pct_eager":
                    why = why_e
                elif k == "gpu_idle_pct_best":
                    why = why_b
                missing.append((LABEL[m], k, why))

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def cell(v):
        return "—" if v in ("", None) else str(v)

    L = ["# E5 / E5b across models — the action expert, one row per model",
         "",
         "**Generated by `bench/make_e5_e5b_models.py` from committed files only; nothing here "
         "is newly measured.** Columns are T008's acceptance list. `—` means no committed "
         "source exists for that cell; every one is listed under *Not measured* with its reason. "
         f"CSV: `{args.csv}`.",
         "",
         "## Latency, floor and distance from it",
         "",
         "| model | S | ms/step eager | ms/step compiled | floor ms (eager / compiled bytes) "
         "| gap × floor eager | gap × floor compiled | % peak BW eager | % peak BW compiled |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['model']} | {cell(r['S'])} | {cell(r['ms_per_step_eager'])} "
                 f"(`{r['eager_rung']}`) | {cell(r['ms_per_step_compiled'])} "
                 f"(`{r['compiled_rung']}`) | {cell(r['floor_ms_eager_bytes'])} / "
                 f"{cell(r['floor_ms_compiled_bytes'])} | {cell(r['gap_eager'])} | "
                 f"{cell(r['gap_compiled'])} | {cell(r['pct_peak_bw_eager'])} | "
                 f"{cell(r['pct_peak_bw_compiled'])} |")
    L += ["", "## Where the time goes inside the expert", "",
          "| model | GPU idle % eager / best | SM occupancy % eager / best "
          "| top stall eager | top stall best | LSU share % eager / best | dominant kernel eager "
          "| dominant kernel best |",
          "|---|---:|---:|---|---|---:|---|---|"]
    for r in rows:
        L.append(f"| {r['model']} | {cell(r['gpu_idle_pct_eager'])} / "
                 f"{cell(r['gpu_idle_pct_best'])} | {cell(r['sm_occupancy_pct_eager'])} / "
                 f"{cell(r['sm_occupancy_pct_best'])} | {cell(r['top_stall_eager'])} | "
                 f"{cell(r['top_stall_best'])} | {cell(r['lsu_share_pct_eager'])} / "
                 f"{cell(r['lsu_share_pct_best'])} | `{cell(r['dominant_kernel_eager'])[:48]}` "
                 f"| `{cell(r['dominant_kernel_best'])[:48]}` |")
    L += ["", "## Reading it — two things the table cannot say on its own", "",
          "* **\"compiled\" and \"best\" are not one configuration across columns.** ms/step, "
          "floor, gap and % BW use Part B's best RUNG per model (named in the table). GPU idle, "
          "SM occupancy and the E5b columns use C1/E5's `best`, which for the LeRobot models is "
          "`--compile-mode reduce-overhead` and for GR00T is `--head-mode reduce-overhead "
          "--llm-mode max-autotune-no-cudagraphs`. They are the nearest committed matches, not "
          "identical builds.",
          "* **GR00T's stall columns cover `ae_total`'s first 40 captured kernels**, not one "
          "step: its loop has no per-step wrap point. The LeRobot columns cover `ae_step_0`.",
          "", "## Sources, per model", ""]
    for name, src in sources:
        L.append(f"* **{name}** — " + "; ".join(f"{k}: `{v}`" for k, v in src.items() if v))
    L += ["", "## Not measured", ""]
    for name, k, why in missing:
        L.append(f"* {name} · `{k}`" + (f" — {why}" if why else " — no committed source"))
    Path(args.md).write_text("\n".join(L) + "\n")
    print(f"wrote {args.md} and {args.csv} ({len(rows)} models, {len(missing)} empty cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
