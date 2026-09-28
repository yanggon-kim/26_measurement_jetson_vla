"""reports/E5b_<model>.md in docs/03_instruction_level_analysis.md §10's seven parts (2026-09-28).

Before this, reports/E5b_{pi0,pi05,smolvla}.md were verbatim per-model extracts of reports/E5b_summary.md
(bench/make_model_reports.py) plus a hand-written S-sweep section, and reports/E5b_groot.md was hand-written;
none followed §10's structure. This builds each report from committed files only:

  results/*_E5b_<m>_{eager,best}/sass/*        targets.csv, inventories, hot loop, pseudocode, latency budget,
                                               cross-check (bench/e5b_static.py)
  results/*_E5b_<m>_{eager,best}/ncu/parsed.json   per-symbol pipes, LSU split, stalls, occupancy
  results/*_E5_<m>_*/timeline_summary.csv      nsys: step wall, kernel union, GPU idle, kernels per step
  results/20260925_E6_<m>/roofline.csv          analytic floor of the step (E6)
  results/20260928_E5_ncu_<m>_*_aelayer/meta.json   layer-range captures (T008 step 3)
  reports/E5b_parts/*.md                        the hand-written S-sweep sections and GR00T prose, moved here
                                               verbatim from the previous reports
  reports/E5b_summary.md                        per-short-name stall tables (extracted, unchanged)

Every number printed is read from those files or is arithmetic on them that the code below states (sums,
ratios, differences); nothing is typed in. §7's stall -> instruction-class mapping is imported from
bench/analyze_e5b.py, which reproduces docs/03 §7 verbatim.

    python3 bench/make_e5b_reports.py            # writes reports/E5b_{pi0,pi05,smolvla,groot}.md
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
from analyze_e5b import STALL_CAUSE, UNDOCUMENTED  # noqa: E402
import make_model_reports as MMR  # noqa: E402

R = REPO / "results"
PARTS = REPO / "reports" / "E5b_parts"
CORRECTION = ("**Correction, 2026-09-28 (completeness audit).** A table here is per kernel SHORT name. Where several "
              "instantiations share one short name, the stall mix and time are their aggregate — most visibly π0 / "
              "best `Kernel2`, which is four different `cutlass::Kernel2<...>` symbols (two wmma bf16 variants carry "
              "22.3 % + 10.2 % of C1 step time; the 256x64 variant whose SASS was the only one dumped until now is 1 "
              "of 20 launches, 7 %). Per-symbol SASS, inventories, cross-checks, hot loops and latency budgets are now "
              "in each `results/*_E5b_*/sass/` (`bench/e5b_static.py`); `sass/targets.csv` lists every symbol with "
              "its own share. Backward-branch counts before 2026-09-28 were all 0 because of a masking bug in "
              "`bench/sass_backward_bra.py` (fixed); no report quoted them.")

MODELS = {
    "pi0": dict(label="π0", task="T007", e5b=("20260924_E5b_pi0_eager", "20260924_E5b_pi0_best"),
                e5={"eager N=10": "20260922_E5_pi0_eager_N10", "best N=10": "20260924_E5_pi0_best_N10",
                    "eager N=4": "20260922_E5_pi0_eager_N4", "best N=4": "20260924_E5_pi0_best_N4"},
                region="ae_step", e6="20260925_E6_pi0", e6_region="ae_step", ssweep="20260928_E5b_pi0_ssweep",
                aelayer=(), summary_key="pi0", S=51),
    "pi05": dict(label="π0.5", task="T008", e5b=("20260924_E5b_pi05_eager", "20260924_E5b_pi05_best"),
                 e5={"eager N=10": "20260924_E5_pi05_eager_N10", "best N=10": "20260924_E5_pi05_best_N10",
                     "eager N=4": "20260928_E5_pi05_eager_N4", "best N=4": "20260928_E5_pi05_best_N4"},
                 region="ae_step", e6="20260925_E6_pi05", e6_region="ae_step", ssweep="20260925_E5b_pi05_ssweep",
                 aelayer=("20260928_E5_ncu_pi05_eager_aelayer", "20260928_E5_ncu_pi05_best_aelayer"),
                 summary_key="pi05", S=50),
    "smolvla": dict(label="SmolVLA", task="T008", e5b=("20260924_E5b_smolvla_eager", "20260924_E5b_smolvla_best"),
                    e5={"eager N=10": "20260924_E5_smolvla_eager_N10", "best N=10": "20260924_E5_smolvla_best_N10",
                        "eager N=4": "20260928_E5_smolvla_eager_N4", "best N=4": "20260928_E5_smolvla_best_N4"},
                    region="ae_step", e6="20260925_E6_smolvla", e6_region="ae_step",
                    ssweep="20260925_E5b_smolvla_ssweep",
                    aelayer=("20260928_E5_ncu_smolvla_eager_aelayer", "20260928_E5_ncu_smolvla_best_aelayer"),
                    summary_key="smolvla", S=50),
    "groot": dict(label="GR00T N1.7", task="T009", e5b=("20260925_E5b_groot_n17_eager", "20260925_E5b_groot_n17_best"),
                  e5={"eager N=4": "20260924_E5_groot_n17_eager", "best N=4": "20260924_E5_groot_n17_best"},
                  region="ae_total", e6="20260925_E6_groot_n17", e6_region="ae_step",
                  ssweep="20260925_E5b_groot_n17_ssweep", aelayer=(), summary_key=None, S=41),
}


def f(v, k=2):
    try:
        return f"{float(v):.{k}f}"
    except (TypeError, ValueError):
        return str(v)


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_csv(p: Path) -> list[dict]:
    with p.open(newline="") as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def md_block(p: Path, heading: str) -> list[str]:
    """Lines of the `## heading` section of a markdown file, table only."""
    out, on = [], False
    for l in p.read_text().splitlines():
        if l.startswith("## "):
            on = l[3:].startswith(heading)
            continue
        if on and l.startswith("|"):
            out.append(l)
    return out


def demote(text: str) -> str:
    return re.sub(r"(?m)^(#{2,5}) ", lambda m: "#" + m.group(1) + " ", text)


# ------------------------------------------------------------------------------------------------ data

def load_dir(name: str) -> dict:
    d = R / name
    meta = json.loads((d / "meta.json").read_text())
    parsed = json.loads((d / "ncu" / "parsed.json").read_text())
    parsed = next(iter(parsed.values()))
    targets = read_csv(d / "sass" / "targets.csv")
    return {"dir": d, "name": name, "meta": meta, "parsed": parsed, "targets": targets,
            "mode": "eager" if "eager" in name else "best"}


def weight_us(t: dict) -> float:
    """Kernel time of a symbol in the E5b capture window (mean duration x launches captured)."""
    return (num(t.get("e5b_mean_duration_us")) or 0.0) * (num(t.get("e5b_launches")) or 0.0)


def waiting_pct(e: dict) -> dict:
    st = e.get("stalls") or {}
    den = (e.get("stall_sum") or 0) - st.get("selected", 0)
    return {k: 100 * v / den for k, v in st.items() if k != "selected" and den} if den else {}


def timeline(name: str, region: str) -> dict | None:
    p = R / name / "timeline_summary.csv"
    if not p.exists():
        return None
    row = next((r for r in read_csv(p) if r["region"] == region), None)
    if row is None:
        return None
    k = num(row["instances_per_inference"]) or 1
    return {"wall": num(row["wall_ms"]) / k, "union": num(row["kernel_union_ms"]) / k,
            "idle": num(row["gpu_idle_ms"]) / k, "idle_frac": num(row["gpu_idle_frac"]),
            "kernels": num(row["kernels_per_unit"]), "gap_us": num(row["gap_median_us"]), "per": k,
            "unit": row["unit"], "src": f"results/{name}/timeline_summary.csv"}


def e6_floor(name: str, region: str) -> dict:
    row = next(r for r in read_csv(R / name / "roofline.csv") if r["region_or_kernel"] == region)
    return {"floor": num(row["floor_ms"]), "ms": num(row["ms"]), "gap_ratio": num(row["gap_ratio"]),
            "note": row["note"], "src": f"results/{name}/roofline.csv"}


def inner_counts(d: Path, sym: str) -> dict:
    p = d / "sass" / f"{sym}_sm87_inner_inventory.csv"
    if not p.exists():
        return {}
    c: dict = {}
    for r in read_csv(p):
        if not (r.get("count") or "").isdigit():       # repeated header / blank rows inside the file
            continue
        c[r["category"]] = c.get(r["category"], 0) + int(r["count"])
        if re.match(r"(HMMA|IMMA|FFMA|HFMA2|FMUL|FADD|HMUL2|HADD2|DFMA)", r["opcode"]):
            c["MATH"] = c.get("MATH", 0) + int(r["count"])   # arithmetic of the algorithm, not address IMADs
        if r.get("lsu_subclass"):
            c["LSU:" + r["lsu_subclass"]] = c.get("LSU:" + r["lsu_subclass"], 0) + int(r["count"])
    return c


def budget_summary(d: Path, sym: str) -> dict:
    p = d / "sass" / f"{sym}_sm87_latency_budget.csv"
    out = {}
    if not p.exists():
        return out
    on = False
    for l in p.read_text().splitlines():
        if l.startswith("# summary"):
            on = True
            continue
        if on and l.strip():
            k, _, rest = l.partition(",")
            out[k] = rest
    return out


def chain(d: Path, sym: str) -> str:
    p = d / "sass" / f"{sym}_sm87_pseudocode.md"
    if not p.exists():
        return "not produced"
    t = p.read_text()
    m = re.search(r"## Longest dependency chain.*?\n\n`([^`]+)`", t, re.S)
    if not m:
        return "not in the pseudocode file"
    ops = m.group(1).split(" -> ")
    return m.group(1) if len(ops) <= 8 else " -> ".join(ops[:4] + ["…"] + ops[-3:]) + f" ({len(ops)} instructions)"


def algorithm(d: Path, sym: str) -> str:
    p = d / "sass" / f"{sym}_sm87_pseudocode.md"
    if not p.exists():
        return "not produced"
    m = re.search(r"Algorithm \(§5 fingerprints\): \*\*(.+?)\*\*", p.read_text())
    return m.group(1) if m else "not stated"


def hotloop(d: Path, sym: str, cap: int = 70) -> list[str]:
    p = d / "sass" / f"{sym}_sm87_hotloop.sass.md"
    if not p.exists():
        return ["not produced"]
    t = p.read_text()
    m = re.search(r"```\n(.*?)```", t, re.S)
    body = m.group(1).rstrip("\n").splitlines() if m else []
    hdr = t.split("```")[0].strip().splitlines()
    rng = next((l for l in hdr if l.startswith("PC range")), "")
    out = [rng, "", "```"] + body[:cap]
    if len(body) > cap:
        out.append(f"... {len(body) - cap} more lines in the file")
    out += ["```", f"(`{p.relative_to(REPO)}`)"]
    return out


def crosscheck(d: Path, sym: str) -> list[str]:
    p = d / "sass" / f"{sym}_sm87_crosscheck.md"
    if not p.exists():
        return ["not produced"]
    t = p.read_text()
    out = [l for l in t.splitlines() if l.startswith("Static ratios") or l.startswith("**Direction agreement")]
    tbl = [l for l in t.split("## Dynamic")[1].splitlines() if l.startswith("|")] if "## Dynamic" in t else []
    return out + [""] + tbl + ["", f"(`{p.relative_to(REPO)}`)"]


# ------------------------------------------------------------------------------------------------ parts

def deep_targets(D: dict) -> list[dict]:
    out = []
    for t in D["targets"]:
        if not t["role"].startswith("§1 deep"):
            continue
        e = D["parsed"].get(t["kernel_id"])
        out.append({**t, "_e": e, "_w": weight_us(t)})
    return sorted(out, key=lambda x: -x["_w"])


def trace_ok(stall: str, inner: dict) -> tuple[str, bool]:
    """§10 'every stall reason > 5 % traced to instructions': the §7 class must be present in the inner loop."""
    if stall == "not_selected":
        return "ready state (eligible, another warp issued) — like `selected`, not a wait on an instruction", True
    if stall in UNDOCUMENTED:
        return "absent from docs/03 §7 — cannot be traced to an instruction class", False
    cause, cls = STALL_CAUSE.get(stall, ("", "?"))
    need = {"LSU": ("LSU",), "LSU/XU": ("LSU", "Shuffle", "XU"), "FMA": ("FMA", "Tensor core"),
            "control": ("Branch", "Other", "ALU"), "ALU/XU": ("ALU", "XU", "Other")}.get(cls, ())
    have = [c for c in need if inner.get(c)]
    if not inner:
        return f"§7 class {cls}; no inner-loop inventory (no backward branch)", False
    if have:
        return f"§7 class {cls}: inner loop has " + ", ".join(f"{c} {inner[c]}" for c in have), True
    return f"§7 class {cls}: none of {need} in the inner loop", False


def part_answer(m: str, cfg: dict, dirs: list[dict], tl: dict, floor: dict) -> list[str]:
    L = ["## 1. The answer", ""]
    unit = "step" if cfg["region"] == "ae_step" else "action head (N = 4)"
    fl = floor["floor"] * (1 if cfg["region"] == "ae_step" else 4)
    sent = []
    for D in dirs:
        key = next((k for k in tl if k.startswith(D["mode"]) and "N=10" in k), None) or \
            next((k for k in tl if k.startswith(D["mode"])), None)
        t = tl.get(key)
        dts = [x for x in deep_targets(D) if x["_e"]]
        tot = sum(weight_us(x) for x in D["targets"])
        wl = sum(x["_w"] for x in dts)
        mix: dict = {}
        for x in dts:
            for k, v in waiting_pct(x["_e"]).items():
                mix[k] = mix.get(k, 0) + x["_w"] * v / wl
        top = sorted(mix.items(), key=lambda kv: -kv[1])[:2]
        s1 = (f"**{D['mode'].capitalize()}**: the {unit} takes {f(t['wall'])} ms under nsys ({f(t['wall'] / fl, 1)}× the "
              f"E6 floor of {f(fl, 2)} ms) and the GPU is idle {f(100 * t['idle_frac'], 0)} % of it "
              f"({f(t['kernels'], 0)} kernels, median gap {f(t['gap_us'], 1)} µs)" if t else f"**{D['mode']}**")
        if dts:
            x0 = dts[0]
            w0 = waiting_pct(x0["_e"])
            k0 = max(w0.items(), key=lambda kv: kv[1])
            s1 += (f". Inside the kernels, the {len(dts)} deep targets ({f(100 * wl / tot, 0)} % of the capture window) "
                   f"wait mostly on " + " and ".join(f"`{k}` ({f(v, 0)} % of waiting, time-weighted)" for k, v in top)
                   + f"; the heaviest, `{x0['kernel_id'][:70]}`, waits on `{k0[0]}` {f(k0[1], 0)} % at "
                   f"{f(x0['_e'].get('compute_sm_throughput_pct'), 0)} % SM and {f(x0['_e'].get('memory_l2_throughput_pct'), 0)} % "
                   f"L2 throughput ({x0['_e'].get('classification')}).")
        else:
            s1 += "; no symbol passes §1's bar."
        sent.append(s1)
    L += [" ".join(sent), "",
          "Sources: `" + "`, `".join(sorted({(tl[k] or {}).get("src", "") for k in tl if tl[k]})) + f"`, `{floor['src']}`, "
          "`results/<E5b dir>/ncu/parsed.json`. What `wait`, `long_scoreboard` etc. mean is §7's mapping (part 3).", ""]
    syn = (REPO / "reports" / "SYNTHESIS_all_measurements.md").read_text()
    lab = {"pi0": "**π0**", "pi05": "**π0.5**", "smolvla": "**SmolVLA**", "groot": "**GR00T**"}[m]
    row = next((l for l in syn.splitlines() if l.startswith(f"| {lab} |")), None)
    if row:
        cells = [c.strip() for c in row.strip("|").split("|")]
        L += ["The consolidated reading across E5/E5b/E6 (quoted from `reports/SYNTHESIS_all_measurements.md` §3.3):", "",
              f"* eager: {cells[1]}", f"* best: {cells[2]}", ""]
    if m == "groot":
        L += ["GR00T's answer as written in the previous version of this report (kept verbatim):", "",
              demote((PARTS / "groot_answer.md").read_text()).replace("### The answer", "").strip(), ""]
    return L


def part_targets(dirs: list[dict]) -> list[str]:
    L = ["## 2. Targets", "",
         "All symbols the E5b captures contain, per mode, from `sass/targets.csv` (bench/e5b_static.py). "
         "`share` is the symbol's share of C1 `ae_step` kernel time where C1 captured it; `window µs` = E5b mean "
         "duration × launches captured. Deep = past §1's bar; the others are launch-bound (§1) and get §2–§4 only.",
         ""]
    for D in dirs:
        tot = sum(weight_us(t) for t in D["targets"])
        deep_w = sum(weight_us(t) for t in D["targets"] if t["role"].startswith("§1 deep"))
        L += [f"### {D['mode']} — `results/{D['name']}/sass/targets.csv`", "",
              f"Deep targets cover {f(100 * deep_w / tot, 1)} % of the E5b window's kernel time "
              f"({len([t for t in D['targets'] if t['role'].startswith('§1 deep')])} of {len(D['targets'])} symbols).",
              "",
              "| symbol | role | share of C1 step % | launches / step | E5b launches × mean µs | window % | SM % | grid | block | regs |",
              "|---|---|---:|---:|---:|---:|---:|---|---|---:|"]
        for t in sorted(D["targets"], key=lambda x: -weight_us(x)):
            L.append(f"| `{t['kernel_id'][:90]}` | {'deep' if t['role'].startswith('§1 deep') else 'launch-bound'} | "
                     f"{f(t['share_of_ae_step_kernel_time']) if num(t['share_of_ae_step_kernel_time']) is not None else 'not in C1'} | "
                     f"{t['launches_per_step'] if num(t['launches_per_step']) is not None else '—'} | "
                     f"{t['e5b_launches']} × {f(t['e5b_mean_duration_us'], 1)} | {f(100 * weight_us(t) / tot, 1)} | "
                     f"{f(t['e5b_sm_throughput_pct'], 1)} | {t['grid']} | {t['block']} | {f(t['regs_per_thread'], 0)} |")
        L.append("")
    return L


def per_target(D: dict, t: dict) -> list[str]:
    d, sym, e = D["dir"], t["kernel_id"], t["_e"] or {}
    L = [f"#### {D['mode']} — `{sym}`", "",
         f"*Algorithm (§5):* {algorithm(d, sym)}. Demangled `{t['demangled'][:160]}`; {t['library_or_cubin']}.", ""]
    tot = d / "sass" / f"{sym}_sm87_total_inventory.md"
    inn = d / "sass" / f"{sym}_sm87_inner_inventory.md"
    for lab, p in (("whole kernel", tot), ("inner loop", inn)):
        if p.exists():
            cat = md_block(p, "Per-category subtotal")
            lsu = md_block(p, "LSU split")
            first = p.read_text().splitlines()
            hl = next((l for l in first if l.startswith("Total instructions")), "")
            L += [f"*Inventory, {lab}* ({hl.replace('**', '')}; `{p.relative_to(REPO)}`):", ""] + cat + [""] + lsu + [""]
        else:
            L += [f"*Inventory, {lab}:* not produced (no backward branch, so no inner loop).", ""]
    L += ["*Annotated hot loop:*", ""] + hotloop(d, sym) + [""]
    L += [f"*Dependency chain (one iteration, static def-use):* `{chain(d, sym)}`", ""]
    pu = e.get("pipe_utilization") or {}
    ls = e.get("pipe_lsu_split") or {}
    L += ["*Pipes and LSU (ncu, `ncu/parsed.json`):*", "",
          "| LSU % | FMA % | ALU % | XU % | tensor (HMMA) % | load % | store % | atomic % | load:write | SM thr % | L2 thr % | achieved occ % |",
          "|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
          f"| {f(pu.get('lsu_active_pct'), 1)} | {f(pu.get('fma_active_pct'), 1)} | {f(pu.get('alu_active_pct'), 1)} | "
          f"{f(pu.get('xu_active_pct'), 1)} | {f(next(iter((pu.get('tensor_hmma') or {'': 'n/a'}).values())), 1)} | "
          f"{f(ls.get('load_pct'), 1)} | {f(ls.get('store_pct'), 1)} | {ls.get('atomic_pct', 'n/a')} | "
          f"{f(ls.get('load_to_write_ratio'), 2)} | {f(e.get('compute_sm_throughput_pct'), 1)} | "
          f"{f(e.get('memory_l2_throughput_pct'), 1)} | {f((e.get('occupancy') or {}).get('achieved_occupancy_pct'), 1)} |", ""]
    w = waiting_pct(e)
    inner = inner_counts(d, sym)
    L += ["*Stall → instruction (§7; % of waiting, `selected` excluded):*", "",
          "| stall | % of waiting | §7 cause | traced to the inner loop |", "|---|---:|---|---|"]
    for k, v in sorted(w.items(), key=lambda kv: -kv[1]):
        if v < 1:
            continue
        cause = STALL_CAUSE.get(k, ("not in §7", ""))[0] if k not in UNDOCUMENTED else "not in §7"
        tr, _ = trace_ok(k, inner) if v > 5 else ("(≤ 5 %, not required)", True)
        L.append(f"| `{k}` | {f(v, 1)} | {cause} | {tr} |")
    L += [""]
    L += ["*Cross-check (static vs dynamic):*", ""] + crosscheck(d, sym) + [""]
    b = budget_summary(d, sym)
    L += ["*Latency budget (§9, documented latencies only):*", ""]
    for k in ("serial_cycles_per_iteration_DRAM_variant", "serial_cycles_per_iteration_L2_variant",
              "max_documented_cycles_chain_DRAM_variant", "measured_cycles_per_iteration",
              "inner_loop_iterations_per_warp_derived", "achieved_active_warps_per_sm"):
        if k in b:
            L.append(f"* `{k}`: {b[k]}")
    L += [f"(`sass/{sym}_sm87_latency_budget.csv`)", ""]
    return L


def part_deep(m: str, cfg: dict, dirs: list[dict]) -> list[str]:
    L = ["## 3. Per target", ""]
    for D in dirs:
        dts = deep_targets(D)
        L += [f"### {D['mode']} ({len(dts)} deep targets, `results/{D['name']}`)", ""]
        for t in dts:
            L += per_target(D, t)
        lb = [t for t in D["targets"] if not t["role"].startswith("§1 deep")]
        if lb:
            L += [f"**Launch-bound symbols ({D['mode']}), §2–§4 only** — evidence: SM throughput and mean duration "
                  "against the ~62 µs eager launch gap (§1):", "",
                  "| symbol | SM % | mean µs | launches |", "|---|---:|---:|---:|"]
            L += [f"| `{t['kernel_id'][:90]}` | {f(t['e5b_sm_throughput_pct'], 1)} | {f(t['e5b_mean_duration_us'], 1)} | "
                  f"{t['e5b_launches']} |" for t in lb]
            L.append("")
    if cfg["summary_key"]:
        src = (REPO / "reports" / "E5b_summary.md").read_text()
        body, _ = MMR.extract(src, cfg["summary_key"])
        L += ["### Per-short-name stall tables (from `reports/E5b_summary.md`, unchanged)", "",
              "These aggregate every instantiation sharing a short name (see the correction at the top); they are "
              "kept because the previous version of this report consisted of them.", ""]
        L += [demote(l) if l.startswith("#") else l for l in body]
        L.append("")
    else:
        L += ["### Method notes (GR00T, previous version of this report, verbatim)", "",
              demote((PARTS / "groot_method.md").read_text()), "",
              "Per-short-name stall tables: `reports/E5b_groot_analysis.md` (generated by bench/analyze_e5b.py).", ""]
    return L


def part_sweeps(m: str, cfg: dict, dirs: list[dict], tl: dict) -> list[str]:
    L = ["## 4. With S, and under `torch.compile`", "", "### S sweep", ""]
    L += [demote((PARTS / f"{'groot' if m == 'groot' else m}_ssweep.md").read_text()), ""]
    L += ["### Eager vs best", "",
          "| mode | deep targets | deep coverage of window % | time-weighted long scoreboard % of waiting (deep) | "
          "kernels per step (nsys) | GPU idle % (nsys) |", "|---|---:|---:|---:|---:|---:|"]
    for D in dirs:
        dts = deep_targets(D)
        tot = sum(weight_us(t) for t in D["targets"])
        wl = sum(t["_w"] for t in dts if t["_e"])
        lsb = sum(t["_w"] * (t["_e"].get("long_scoreboard_pct_of_waiting") or 0) for t in dts if t["_e"]) / wl if wl else None
        key = next((k for k in tl if k.startswith(D["mode"]) and "N=10" in k), None) or \
            next((k for k in tl if k.startswith(D["mode"])), None)
        t = tl.get(key)
        L.append(f"| {D['mode']} | {len(dts)} | {f(100 * sum(x['_w'] for x in dts) / tot, 1)} | {f(lsb, 1)} | "
                 f"{f(t['kernels'], 1) if t else '—'} ({key}) | {f(100 * t['idle_frac'], 1) if t else '—'} |")
    e_syms = {t["kernel_id"] for t in dirs[0]["targets"]}
    b_syms = {t["kernel_id"] for t in dirs[1]["targets"]}
    L += ["", f"Symbols shared by eager and best: {len(e_syms & b_syms)} "
              f"({', '.join('`' + s[:60] + '`' for s in sorted(e_syms & b_syms)) or 'none'}); best-only: "
              f"{len(b_syms - e_syms)}; eager-only: {len(e_syms - b_syms)}. Under compilation the kernel set changes "
              "(Inductor / Triton kernels and different cuBLAS tiles are new targets, §10), so the per-target "
              "analysis above is per mode.", ""]
    if cfg["aelayer"]:
        L += ["### Layer-range captures (T008 step 3, 2026-09-28)", ""]
        for a in cfg["aelayer"]:
            mt = json.loads((R / a / "meta.json").read_text())
            for c in mt.get("captures", []):
                L.append(f"* `results/{a}` {c['region']}: {c['kernels_profiled']} kernels profiled (cap {c['kernel_cap']}); "
                         f"compile {c['compile_style']}, evidence {c['compile_evidence']}")
            nm = mt.get("not_measured")
            if nm and nm != "none":
                L += [f"* `results/{a}`: {x}" for x in nm]
        L.append("")
    return L


def part_rootcause(m: str, cfg: dict, dirs: list[dict], tl: dict, floor: dict) -> list[str]:
    L = ["## 5. Root cause and gap table (protocol §10)", "",
         "Gap = step wall under nsys − E6 analytic floor. It splits exactly into GPU idle inside the step (wall − kernel "
         "union) and kernel time above the floor (union − floor); the second is attributed to in-kernel stalls through "
         "the deep targets' time-weighted stall mix. *Derived* throughout; inputs are cited per row.", ""]
    unit = "step" if cfg["region"] == "ae_step" else "action head (N = 4)"
    fl = floor["floor"] * (1 if cfg["region"] == "ae_step" else 4)
    for D in dirs:
        key = next((k for k in tl if k.startswith(D["mode"]) and "N=10" in k), None) or \
            next((k for k in tl if k.startswith(D["mode"])), None)
        t = tl.get(key)
        if not t:
            L += [f"### {D['mode']}: no nsys timeline", ""]
            continue
        gap = t["wall"] - fl
        idle, excess = t["idle"], t["union"] - fl
        dts = [x for x in deep_targets(D) if x["_e"]]
        wl = sum(x["_w"] for x in dts)
        mix: dict = {}
        for x in dts:
            for k, v in waiting_pct(x["_e"]).items():
                mix[k] = mix.get(k, 0) + x["_w"] * v / wl
        top = sorted(mix.items(), key=lambda kv: -kv[1])[:3]
        topt = dts[0] if dts else None
        L += [f"### {D['mode']} ({key}, per {unit})", "",
              "| step | observation | evidence |", "|---|---|---|",
              f"| symptom | the {unit} takes {f(t['wall'])} ms against an analytic floor of {f(fl, 3)} ms "
              f"({f(t['wall'] / fl, 2)}×) | `{t['src']}` wall_ms; `{floor['src']}` floor_ms |",
              f"| why 1 | the GPU is idle {f(idle)} ms ({f(100 * t['idle_frac'], 1)} %) of it; {f(t['kernels'], 1)} kernels, "
              f"median gap {f(t['gap_us'], 1)} µs | `{t['src']}` gpu_idle_ms, kernels_per_unit, gap_median_us |",
              f"| why 2 | inside kernels, the deep targets wait on " +
              ", ".join(f"`{k}` {f(v, 1)} %" for k, v in top) + " of waiting (time-weighted) | "
              f"`results/{D['name']}/ncu/parsed.json` stalls |",
              f"| why 3 | the heaviest deep target `{(topt or {}).get('kernel_id', '?')[:70]}` (grid "
              f"{(topt or {}).get('grid')}, block {(topt or {}).get('block')}, {f(((topt or {}).get('_e') or {}).get('occupancy', {}).get('achieved_occupancy_pct'), 1)} % "
              f"achieved occupancy) runs `{chain(D['dir'], topt['kernel_id']) if topt else '?'}` per iteration; "
              f"its inner loop is {inner_counts(D['dir'], topt['kernel_id']).get('LSU', 0) if topt else '?'} LSU of "
              f"{sum(v for k, v in inner_counts(D['dir'], topt['kernel_id']).items() if ':' not in k) if topt else '?'} instructions | "
              f"`sass/*_inner_inventory.csv`, `targets.csv` |",
              f"| root cause | " + (("GPU idle between kernels (host launch / synchronisation) is the larger part of the gap"
                                     + (" — an upper bound under CUDA graphs, see the caveat" if D["mode"] == "best" else ""))
                                    if idle >= excess else
                                    (f"kernel time above the floor is the larger part; inside it the deep targets' "
                                     f"largest wait is `{top[0][0]}` ({f(top[0][1], 0)} % of waiting), §7 class "
                                     f"{STALL_CAUSE.get(top[0][0], ('', 'not in §7'))[1]}" if top else
                                     "kernel time above the floor is the larger part")) +
              " | this table |", "",
              "| factor | evidence | contribution ms | share of gap |", "|---|---|---:|---:|",
              f"| GPU idle inside the {unit} | `{t['src']}` | {f(idle, 3)} | {f(100 * idle / gap, 1)} % |",
              f"| kernel time above the floor | union − floor | {f(excess, 3)} | {f(100 * excess / gap, 1)} % |",
              f"| **total** | wall − floor | **{f(gap, 3)}** | **{f(100 * (idle + excess) / gap, 1)} %** |", ""]
        if D["mode"] == "best" and cfg["region"] == "ae_step":
            L += ["Caveat: under CUDA graphs the step's NVTX range is host time and the graph's kernels run "
                  "asynchronously, so `gpu_idle` for the compiled step is an upper bound "
                  "(`reports/SYNTHESIS_all_measurements.md` §3.4: \"the per-step idle within the graph span is not "
                  "measurable from these traces\").", ""]
    return L


def part_not_measured(m: str, cfg: dict, dirs: list[dict]) -> list[str]:
    L = ["## 6. Not measured / not available on this device", ""]
    for D in dirs:
        np_ = D["meta"].get("not_produced") or {}
        groups: dict = {}
        for path, why in np_.items():
            key = re.sub(r"\[.*?\]|\(/\S+\)|/\S+", "", str(why))[:80]      # group by the reason, not its paths
            groups.setdefault(key, []).append((path, str(why)))
        L.append(f"**{D['mode']}** (`results/{D['name']}/meta.json` `not_produced`, {len(np_)} items):")
        L.append("")
        for _, items in groups.items():
            L.append(f"* {len(items)} × e.g. `{items[0][0]}`: {items[0][1][:320]}")
        L.append("")
    L += ["**Device limits:** `gpc__cycles_active` and `sm__warps_launched` are not exported on this Orin, so no "
          "measured cycles per iteration exist; every latency budget is the documented-latency sum along the chain "
          "(a lower bound). Atomic LSU counters are absent from these captures. No `dram__` counters exist on Orin.", ""]
    if m == "groot":
        L += [demote((PARTS / "groot_mdn.md").read_text()),
              "*(2026-09-28: the π0 S sweep has since been run — `results/20260928_E5b_pi0_ssweep`, `reports/E5b_pi0.md` §4.)*",
              ""]
    return L


def part_review(m: str, cfg: dict, dirs: list[dict], tl: dict, floor: dict) -> list[str]:
    L = ["## 7. Self-review (protocol §11; docs/03 §10 verification list)", ""]
    # §10 verification items, computed
    untraced, traced = [], 0
    mix_rows = []
    for D in dirs:
        for t in deep_targets(D):
            if not t["_e"]:
                continue
            inner = inner_counts(D["dir"], t["kernel_id"])
            for k, v in waiting_pct(t["_e"]).items():
                if v > 5:
                    _, ok = trace_ok(k, inner)
                    if ok:
                        traced += 1
                    else:
                        untraced.append(f"{D['mode']} `{t['kernel_id'][:50]}` {k} {f(v, 1)} %")
            tot = sum(v for kk, v in inner.items() if ":" not in kk) or 0
            if tot:
                mix_rows.append((D["mode"], t["kernel_id"], inner.get("LSU", 0), inner.get("MATH", 0), tot,
                                 (t["_e"].get("memory_l2_throughput_pct") or 0) >= (t["_e"].get("compute_sm_throughput_pct") or 0)))
    mem_bound = "floor bound=memory" in floor["note"]
    lsu_heavy = [r for r in mix_rows if r[2] >= r[3]]
    L += ["| §10 verification item | result |", "|---|---|",
          f"| every stall reason > 5 % traced to instructions | "
          f"{'satisfied' if not untraced else 'NOT satisfied'}: {traced} traced; {len(untraced)} not"
          + (" — " + "; ".join(untraced[:6]) + (" …" if len(untraced) > 6 else "") if untraced else "") + " |",
          "| chain length consistent with the budget | NOT verifiable: measured cycles per iteration are not "
          "exportable on this Orin (§6); the budgets give documented-latency lower bounds only |",
          f"| instruction mix consistent with E6's roofline class | E6 class: "
          f"{'memory' if mem_bound else 'compute'} ({floor['src']}); {len(lsu_heavy)} of {len(mix_rows)} deep-target "
          f"inner loops have at least as many LSU instructions as arithmetic ones (HMMA/FFMA/HFMA2/FMUL/FADD; address IMADs excluded) — "
          f"{'consistent' if len(lsu_heavy) >= len(mix_rows) / 2 else 'not consistent' if not lsu_heavy else 'partly consistent'}"
          f" statically (a GEMM's inner loop issues few, wide loads per many HMMAs, so the static count understates "
          f"memory work); dynamically {sum(1 for r in mix_rows if r[5])} of {len(mix_rows)} have L2 throughput ≥ SM "
          f"throughput (`ncu/parsed.json`) |",
          "| differences across S explained | partly: the swept projections (largest and q) are explained in §4 "
          "(kernel choice per S); the attention / fused / elementwise targets were not swept |",
          "| gap shares sum to 100 % ± 10 | satisfied by construction (idle + kernel excess = wall − floor), §5 |", "",
          "Protocol §11 checklist:", "",
          "* Requested files exist: this report; the §0 file set per `results/<E5b dir>/sass/` (items that cannot "
          "exist are in `meta.json` `not_produced`, §6 above).",
          "* Every number traces to a file in `results/` (cited per table; generated by `bench/make_e5b_reports.py`).",
          "* meta.json complete and raw_files hashed: refreshed by `bench/e5b_static.py` on 2026-09-28.",
          "* Measured (ncu counters, nsys times, SASS) / derived (shares, gap split, verdicts, budgets) / not measured "
          "(§6) are separated.",
          "* Power mode: MAXN for every capture (each run's `meta.json` gate or its recorded reason when absent).",
          "* No file over 50 MB; no weights or secrets staged (checked at commit).", ""]
    if m == "groot":
        L += [demote((PARTS / "groot_disagreements.md").read_text()), ""]
    return L


def build(m: str) -> str:
    cfg = MODELS[m]
    dirs = [load_dir(n) for n in cfg["e5b"]]
    tl = {k: timeline(v, cfg["region"]) for k, v in cfg["e5"].items()}
    floor = e6_floor(cfg["e6"], cfg["e6_region"])
    head = [f"# E5b — {cfg['label']}: what the action-expert kernels wait for", "", CORRECTION, "",
            f"**Task {cfg['task']}. Structured as docs/03_instruction_level_analysis.md §10 (seven parts). Generated by "
            f"`bench/make_e5b_reports.py` from committed files (listed in its docstring); hand-written sections are "
            f"included verbatim from `reports/E5b_parts/`. MAXN. Results: "
            + ", ".join(f"`results/{n}`" for n in cfg["e5b"]) + f", `results/{cfg['ssweep']}`.**", ""]
    if m == "groot":
        head += [(PARTS / "groot_header.md").read_text().strip(), ""]
    body = (part_answer(m, cfg, dirs, tl, floor) + part_targets(dirs) + part_deep(m, cfg, dirs)
            + part_sweeps(m, cfg, dirs, tl) + part_rootcause(m, cfg, dirs, tl, floor)
            + part_not_measured(m, cfg, dirs) + part_review(m, cfg, dirs, tl, floor))
    return "\n".join(head + body) + "\n"


def main() -> int:
    for m in (sys.argv[1:] or list(MODELS)):
        out = REPO / "reports" / f"E5b_{m}.md"
        out.write_text(build(m))
        print(f"wrote {out.relative_to(REPO)} ({len(out.read_text().splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
