"""T011 / T012 — per-model E11 reports in the T010 structure, and the cross-model table in reports/E11_models.md.

T011 and T012 accept `reports/E11_<m>_hybrid_inputs.md` "with the T010 report structure". Until 2026-09-28 those
three files were verbatim extracts of reports/E11_models.md (bench/make_model_reports.py), which have none of
T010's sections. This writes them from the committed result directories:

  * which directory is each part: bench/consolidate_e11.py's parts_for() / pick_run(), imported, so the report
    reads exactly the runs the consolidated model_inputs.csv was built from;
  * the consolidated file: the newest dated results/*_E11_<m>_inputs/model_inputs.csv;
  * each part's own CSVs, meta.json and summary.md.

No measured number is typed here. The only literals are docs/00_context.md's ASSUMED values (cited by line), which
the answer table compares against. A part that is missing prints "pending" where its numbers would be, so the GR00T
report can be generated before its E11a re-run and consolidation exist and regenerated afterwards.

    python bench/make_e11_reports.py                  # pi05, smolvla, groot_n17 + reports/E11_models.md
    python bench/make_e11_reports.py --models pi05    # one model; --no-models-table skips E11_models.md
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from consolidate_e11 import parts_for, pick_run  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
LABEL = {"pi0": "π0", "pi05": "π0.5", "smolvla": "SmolVLA", "groot_n17": "GR00T N1.7"}
TASK = {"pi05": "T011", "smolvla": "T011", "groot_n17": "T012"}
PENDING = "pending"

# docs/00_context.md's ASSUMED values -- the thing each measurement is compared against. Not measurements.
ASSUMED = {
    "t_host_layer_ms": (0.16, "docs/00_context.md:76 (≈ 0.16 ms per π0 layer [ASSUMPTION-based])"),
    "kernels_per_segment": (1, "docs/00_context.md:77 (1 each; 2 for attention_gpu [ASSUMPTION])"),
    "launch_us": (20.0, "docs/00_context.md:77 (20 µs [ASSUMPTION])"),
    "bw_GBps": (143.0, "docs/00_context.md:79 (204.8 GB/s × 0.7 = 143 GB/s [ASSUMPTION])"),
    "prefix_slowdown": (1.0, "docs/00_context.md:92 (slowdown = 1.0 [ASSUMPTION])"),
}

# quantities a consolidated file may lack, and why -- each reason checked against the parts on 2026-09-28
ABSENT_WHY = {
    "t_sync": "E11b (synchronisation) is model-independent and ran once, for π0 (results/*_E11b_xfer); T011 drops "
              "E11b for the later models by design (bench/consolidate_e11.py parts_for docstring)",
    "xfer_latency_fit": "the fitted fixed cost + bandwidth come from E11b's size sweep, which T011 drops for the later "
                        "models by design; this model's own boundary sizes are measured in E11c xfer_exact.csv",
    "t_preprocess": "no part of this model's E11 measures a preprocess region (the LeRobot region set starts at vision; "
                    "GR00T's CPU data processing is restated from T009 only where the part has it)",
    "t_postprocess": "no part of this model's E11 measures a postprocess region",
}


# ------------------------------------------------------------------------------------------------ helpers


def rd(p: Path) -> list[dict]:
    try:
        with p.open(newline="") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def meta_of(d: Path | None) -> dict:
    if d is None:
        return {}
    try:
        return json.loads((d / "meta.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def num(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def f3(x, nd=4):
    v = num(x)
    if v is None:
        return "—"
    if abs(v) >= 1000:
        return f"{v:,.1f}"
    return f"{v:.{nd}g}" if abs(v) < 1 else f"{v:.{nd}f}".rstrip("0").rstrip(".")


def table(rows: list[dict], cols: list[str], heads: list[str] | None = None, maxrows: int | None = None) -> list[str]:
    if not rows:
        return [f"*{PENDING} (no rows)*"]
    heads = heads or cols
    out = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(cols)]
    for r in rows[:maxrows] if maxrows else rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            v = "" if v is None else str(v)
            cells.append(v.replace("|", "/").replace("\n", " "))
        out.append("| " + " | ".join(cells) + " |")
    if maxrows and len(rows) > maxrows:
        out.append(f"| … {len(rows) - maxrows} more rows in the CSV | " + " | " * (len(cols) - 1))
    return out


def rel(p: Path | None) -> str:
    return str(p.relative_to(REPO)) if p else PENDING


def consolidated(model: str) -> Path | None:
    return pick_run(f"*_E11_{model}_inputs")


def parts(model: str) -> dict[str, Path | None]:
    return {p: pick_run(f"*_{p}") for p in parts_for(model)}


def part(ps: dict, prefix: str, contains: str = "") -> list[tuple[str, Path | None]]:
    return [(k, v) for k, v in ps.items() if k.startswith(prefix) and contains in k]


def mi_rows(cdir: Path | None) -> list[dict]:
    return rd(cdir / "model_inputs.csv") if cdir else []


def pick(rows, quantity, key=None, stat="median", style=None, timing=None, var_has=(), var_not=()):
    out = []
    for r in rows:
        if r["quantity"] != quantity or (stat and r["stat"] != stat):
            continue
        if key is not None and r["key"] != key:
            continue
        if style is not None and r["exec_style"] != style:
            continue
        if timing is not None and r["timing"] != timing:
            continue
        if any(s not in r["variant"] for s in var_has) or any(s in r["variant"] for s in var_not):
            continue
        out.append(r)
    return out


def first_val(*a, **k):
    rs = pick(*a, **k)
    return num(rs[0]["value"]) if rs else None


def mode_of(rows, col):
    cnt = {}
    for r in rows:
        if r.get(col):
            cnt[r[col]] = cnt.get(r[col], 0) + 1
    return max(cnt, key=cnt.get) if cnt else ""


def n_layers(model: str, e11a_meta: dict):
    if e11a_meta.get("n_blocks"):
        return e11a_meta["n_blocks"], "E11a meta.json n_blocks"
    d = pick_run(f"*_E5b_{model}_ssweep")
    if d and (d / "shapes.json").exists():
        s = json.loads((d / "shapes.json").read_text())
        if s.get("n_layers"):
            return s["n_layers"], f"{d.name}/shapes.json n_layers (read from the loaded model)"
    return None, "not found"


def notes_2026(metas: list[tuple[str, dict]]) -> list[str]:
    """Every note / backfill / recovery line a part carries that is dated 2026-09-28 -- corrections, verbatim."""
    out = []
    for name, m in metas:
        for k in ("notes", "recovery_notes", "backfill_note", "not_measured"):
            v = m.get(k)
            for s in (v if isinstance(v, list) else [v] if v else []):
                if isinstance(s, str) and "2026-09-28" in s:
                    out.append(f"`{name}`: {s}")
    return out


# ------------------------------------------------------------------------------------------------ cross-model row


def headline(model: str) -> dict:
    """The values T011's cross-model table asks for, from the consolidated file (pending where it does not exist)."""
    c = consolidated(model)
    rows = mi_rows(c)
    ps = parts(model)
    e11a = ps.get(f"E11a_{model}_hostseg")
    h = {"model": LABEL.get(model, model), "run": c.name if c else PENDING}
    e11a_rows = [r for r in rows if "E11a" in r["variant"]]
    h["S"] = mode_of(e11a_rows, "S") or mode_of(rows, "S") or PENDING
    h["P"] = mode_of(e11a_rows, "P") or mode_of(rows, "P") or PENDING
    nl, _ = n_layers(model, meta_of(e11a))
    h["layers"] = nl if nl else PENDING
    first_layer = None
    for r in pick(rows, "t_host_layer", key="sequence"):
        m_ = re.search(r"(layer|block)=(\d+)", r["variant"])
        if m_:
            first_layer = m_.group(0)
            break
    lay = (first_layer,) if first_layer else ()
    for col, key, style in (("t_host_layer eager", "sequence", "eager"),
                            ("t_host_layer compiled", "sequence", "compiled"),
                            ("t_host_layer whole_layer_graph", "whole_layer_graph", "cudagraph")):
        v = first_val(rows, "t_host_layer", key=key, style=style, timing="wall_isolated", var_has=lay)
        h[col] = f3(v) if v is not None else PENDING
    ex = pick(rows, "t_step_extra", style="eager", timing="wall_isolated", var_has=("gap_ms=0",))
    h["t_step_extra (Σ eager)"] = (f3(sum(num(r["value"]) for r in ex)) + f" ({len(ex)} extras)") if ex else PENDING
    for style in ("eager", "compiled"):
        ks = [r for r in pick(rows, "n_kernels_host_segment", stat="value", style=style, var_has=lay)]
        h[f"kernels/layer {style}"] = f3(sum(num(r["value"]) for r in ks)) if ks else PENDING
    v = first_val(rows, "kv_bytes_total", key="padded", stat="value")
    h["kv_bytes_total"] = f"{v:,.0f}" if v is not None else PENDING
    km = pick(rows, "t_kv_materialise", timing="wall_isolated")
    h["t_kv_materialise"] = (f3(km[0]["value"]) + f" ({km[0]['key']})") if km else PENDING
    for q, col in (("t_vision", "t_vision"), ("t_vlm_prefix", "t_vlm_prefix")):
        cells = []
        for style in ("eager", "compiled"):
            rs = [r for r in pick(rows, q, style=style) if r["N"] == "10" and re.search(r"from=\S*_E[34]", r["variant"])]
            cells.append(f3(rs[0]["value"]) if rs else "—")
        h[col + " (N=10 eager / best)"] = " / ".join(cells) if rows else PENDING
    for q in ("prefix_throughput", "e2e_throughput"):
        cells = []
        for style in ("eager", "compiled"):
            rs = pick(rows, q, stat="value", style=style)
            if q == "prefix_throughput":
                rs = [r for r in rs if "load=none" in r["variant"]] or [r for r in rs if "E11f" in r["variant"]]
            cells.append(f3(rs[0]["value"]) if rs else "—")
        h[q + " eager / best"] = " / ".join(cells) if rows else PENDING
    for load in ("gpu_hog_100", "cpu_hog_100"):
        cells = [f3(first_val(rows, "prefix_slowdown", key=load, stat="value", style=s)) for s in ("eager", "compiled")]
        h[f"prefix_slowdown {load} eager / best"] = " / ".join(cells) if rows else PENDING
    return h


CROSS_COLS = ["model", "S", "P", "layers", "t_host_layer eager", "t_host_layer compiled",
              "t_host_layer whole_layer_graph", "t_step_extra (Σ eager)", "kernels/layer eager", "kernels/layer compiled",
              "kv_bytes_total", "t_kv_materialise", "t_vision (N=10 eager / best)", "t_vlm_prefix (N=10 eager / best)",
              "prefix_throughput eager / best", "e2e_throughput eager / best",
              "prefix_slowdown gpu_hog_100 eager / best", "prefix_slowdown cpu_hog_100 eager / best", "run"]


def cross_table(models=("pi0", "pi05", "smolvla", "groot_n17")) -> list[str]:
    rows = [headline(m) for m in models]
    return [
        "## Cross-model table (T011 acceptance; GR00T row per T012)", "",
        "Every cell is read from the model's newest consolidated `results/*_E11_<m>_inputs/model_inputs.csv` "
        "(`run` column) by `bench/make_e11_reports.py`; `pending` means that file or quantity does not exist yet. "
        "Times in ms, throughput in inferences/s, bytes in B. Definitions: `t_host_layer` = the first measured "
        "layer/block's `sequence` (eager, compiled) and `whole_layer_graph` (cudagraph), wall_isolated median; "
        "`t_step_extra (Σ eager)` = the SUM of the per-step extras' eager wall_isolated medians (derived: a sum); "
        "`kernels/layer` = the sum of that layer's host-segment kernel counts (derived: a sum); `t_kv_materialise` = "
        "the first key's wall_isolated median (the key is named); `t_vision` / `t_vlm_prefix` = the restated E3/E4 "
        "(T006/T009) region latencies at N = 10 — for GR00T `t_vlm_prefix` is the backbone EXCLUDING vision "
        "(`vlm_prefix_excl_vision`); `prefix_throughput` = E11e with no load; `prefix_slowdown` = E11e.", "",
    ] + table(rows, CROSS_COLS) + [""]


# ------------------------------------------------------------------------------------------------ per-model report


def report(model: str) -> str:
    lab = LABEL.get(model, model)
    ps = parts(model)
    c = consolidated(model)
    rows = mi_rows(c)
    cmeta = meta_of(c)
    e11a_name = f"E11a_{model}_hostseg"
    e11a = ps.get(e11a_name)
    am = meta_of(e11a)
    gate = am.get("gate") or {}
    metas = [(v.name, meta_of(v)) for v in ps.values() if v]
    L = [f"# E11 — measured inputs for the hybrid GPU + PIM model ({lab})", ""]
    ck = cmeta.get("checkpoint") or am.get("checkpoint") or PENDING
    commit = (cmeta.get("model_commit") or am.get("model_commit") or "")[:7] or PENDING
    L += [f"**Task {TASK[model]}** · {lab} (`{ck}` @ `{commit}`) · {gate.get('power_mode', am.get('power_mode', PENDING))}, "
          f"{gate.get('sm_count', am.get('sm_count', '?'))} SMs, GPU {num(gate.get('gpu_cur_freq_hz')) and f3(num(gate.get('gpu_cur_freq_hz')) / 1e6)} MHz"
          f", EMC {num(am.get('emc_hz')) and f3(num(am.get('emc_hz')) / 1e6)} MHz · bf16 · batch 1 · "
          f"P = {mode_of([r for r in rows if 'E11a' in r['variant']], 'P') or am.get('prefix_tokens', PENDING)}, "
          f"S = {mode_of([r for r in rows if 'E11a' in r['variant']], 'S') or am.get('suffix_tokens', PENDING)}", ""]
    chk = ""
    if c and (c / "raw" / "check_model_inputs.log").exists():
        chk = (c / "raw" / "check_model_inputs.log").read_text()
    exit_ok = "exit=0" in chk
    L += [f"Deliverable: `{rel(c)}/model_inputs.csv` — "
          + (f"{cmeta.get('total_rows', len(rows)):,} rows, {len(cmeta.get('quantities_present', []))} of "
             f"{len(cmeta.get('quantities_present', [])) + len(cmeta.get('quantities_absent', []))} quantities §13.8 "
             f"names, from {len(cmeta.get('provenance', []))} source runs. "
             f"`docs/reference_scripts/check_model_inputs.py` run in place: "
             f"**{'exit 0' if exit_ok else 'see raw/check_model_inputs.log'}**." if c else
             f"**{PENDING}** — the consolidated file is built by `bench/consolidate_e11.py --model {model}` once every "
             "part exists."), "",
          "*Generated by `bench/make_e11_reports.py` from the committed result directories; no number is typed. "
          "Replaces the earlier extract of `reports/E11_models.md`, which had none of T010's sections.*", "", "---", ""]

    # ---------------------------------------------------------------- the answer first
    L += ["## The answer first", "",
          "docs/00_context.md prices the GPU side of the hybrid model analytically with ASSUMED constants. Each "
          "row puts this model's measurement next to the assumption it replaces (ratio = measured / assumed).", "",
          "| term | assumed (source) | measured | ratio |", "|---|---|---|---:|"]
    lay = None
    for r in pick(rows, "t_host_layer", key="sequence"):
        m_ = re.search(r"(layer|block)=(\d+)", r["variant"])
        if m_:
            lay = m_.group(0)
            break
    ta, tsrc = ASSUMED["t_host_layer_ms"]
    for key, style in (("sequence", "eager"), ("sequence", "compiled"), ("sum_of_segments", "eager"),
                       ("sum_of_segments", "cudagraph"), ("whole_layer_graph", "cudagraph")):
        v = first_val(rows, "t_host_layer", key=key, style=style, var_has=(lay,) if lay else ())
        L.append(f"| `T_host_layer`, {key} / {style} ({lay or PENDING}) | {ta} ms ({tsrc}) | "
                 f"{f3(v) + ' ms' if v is not None else PENDING} | {f3(v / ta, 3) + '×' if v else '—'} |")
    ks = pick(rows, "n_kernels_host_segment", stat="value", style="eager", var_has=(lay,) if lay else ())
    if ks:
        vals = sorted((num(r["value"]), r["key"]) for r in ks)
        L.append(f"| kernels per host segment, eager ({lay}) | {ASSUMED['kernels_per_segment'][0]} "
                 f"({ASSUMED['kernels_per_segment'][1]}) | {f3(vals[0][0])} (`{vals[0][1]}`) – {f3(vals[-1][0])} "
                 f"(`{vals[-1][1]}`) | up to {f3(vals[-1][0] / ASSUMED['kernels_per_segment'][0], 3)}× |")
    else:
        L.append(f"| kernels per host segment | 1 | {PENDING} | — |")
    lk = pick(rows, "t_launch_per_kernel")
    if lk:
        v = num(lk[0]["value"])
        L.append(f"| launch cost per kernel (`{lk[0]['key'] or lk[0]['variant'][:40]}`) | {ASSUMED['launch_us'][0]} µs "
                 f"({ASSUMED['launch_us'][1]}) | {f3(v * 1e3)} µs | {f3(v * 1e3 / ASSUMED['launch_us'][0], 3)}× |")
    kvb = first_val(rows, "kv_bytes_total", key="padded", stat="value")
    ba, bsrc = ASSUMED["bw_GBps"]
    for direction in ("h2d", "d2h"):
        rs = [r for r in pick(rows, "xfer_latency", key=f"pinned:{direction}")
              if "size_label=kv_bytes_total" in r["variant"]]
        if rs and kvb:
            v = num(rs[0]["value"])
            gbps = kvb / (v / 1e3) / 1e9
            L.append(f"| moving the hand-over object ({kvb:,.0f} B), pinned {direction} | "
                     f"{f3(kvb / ba / 1e6)} ms at {ba} GB/s ({bsrc}) | {f3(v)} ms ({f3(gbps)} GB/s, derived) | "
                     f"{f3(v / (kvb / ba / 1e6), 3)}× |")
    kc = first_val(rows, "kv_copied_bytes_per_step", stat="value")
    if kc is not None:
        L.append(f"| `kv_copied_bytes_per_step` | 0 (the hybrid model assumes the hand-over is loaded once) | "
                 f"{kc:,.0f} B | not zero |")
    for r in pick(rows, "t_kv_materialise", timing="wall_isolated"):
        L.append(f"| `t_kv_materialise`, {r['variant'].split(';from=')[0] or r['key']} | 0 | "
                 f"{f3(r['value'])} ms (n = {r['n']}) | not zero |")
    sa, ssrc = ASSUMED["prefix_slowdown"]
    for load in ("gpu_hog_100", "cpu_hog_100", "gpu_spin_control"):
        for style in ("eager", "compiled"):
            v = first_val(rows, "prefix_slowdown", key=load, stat="value", style=style)
            if v is not None:
                L.append(f"| `prefix_slowdown` under {load}, {style} | {sa} ({ssrc}) | {f3(v, 3)}× | {f3(v / sa, 3)}× |")
    if not rows:
        L.append(f"| (all terms) | | {PENDING} | |")
    L += ["", "The E11e spin controls take GPU time slices (`gpu_spin_control`) or CPU cores (`cpu_spin_control`) "
              "without taking memory bandwidth; a slowdown they also produce is not lost bandwidth (protocol §13.5).", ""]

    # ---------------------------------------------------------------- method notes
    L += ["## Method notes that shaped every number", ""]
    corr = CORRECTIONS.get(model, [])
    if corr:
        L += ["**Corrections found and fixed on 2026-09-28 (completeness audit).** Each superseded run is kept; the "
              "consolidated file reads only the newest dated directory per part.", ""]
        L += [f"* {x}" for x in corr]
        L.append("")
    n26 = notes_2026(metas)
    if n26:
        L += ["What the parts' own `meta.json` record about it, verbatim:", ""] + [f"* {x}" for x in n26] + [""]
    # E11a's generic meta notes are NOT copied: host_segments.py writes the same π0-worded notes for every LeRobot
    # model (they name pi_gemma.py and adarms_cond for SmolVLA too), so quoting them here would mislead.
    L += ["", "Timing definitions (protocol §13.1): `wall_isolated` = synchronize, time, synchronize per call; "
              "`gpu_event` = CUDA events around the call; `wall_launch` = host time to enqueue; `wall_loop` = a "
              "60 s back-to-back loop. Every timed callable returns its result (a compiled callable whose output "
              "is discarded is dead-code-eliminated — host_segments.py).", ""]

    # ---------------------------------------------------------------- E11a
    L += ["## E11a — the host segments", ""]
    if e11a is None:
        L += [f"*{PENDING}: no E11a directory.*", ""]
    else:
        L += [f"Run: `{rel(e11a)}`.", ""]
        seg = rd(e11a / "segments.csv")
        if seg:
            L += ["### Segment list (derived from the model code; shapes read from the loaded model)", ""]
            cols = [c for c in ("segment", "layer_kind", "source_ref", "bytes_read", "bytes_written", "flops",
                                "in_model_code") if c in seg[0]]
            L += table(seg, cols) + [""]
        ss = rd(e11a / "segment_summary.csv")
        if ss:
            lc = "layer" if "layer" in ss[0] else ("block" if "block" in ss[0] else None)
            piv = {}
            for r in ss:
                if r.get("timing") not in ("wall_isolated", "gpu_event") or "gap_ms=2" in r.get("variant", ""):
                    continue
                k = (r["segment"], r.get(lc, "") if lc else "")
                d = piv.setdefault(k, {"segment": r["segment"], "layer": k[1]})
                d[f"{r['exec_style']} {r['timing']}"] = f3(r["median_ms"])
                if r.get("n_kernels") not in (None, ""):
                    d[f"kernels {r['exec_style']}"] = f3(r["n_kernels"])
            cols = ["segment", "layer", "eager wall_isolated", "eager gpu_event", "compiled wall_isolated",
                    "cudagraph wall_isolated", "kernels eager", "kernels compiled", "kernels cudagraph"]
            L += ["### Per-segment time (median ms) and kernels per call", ""] + table(list(piv.values()), cols) + [
                "", "Kernel counts from `kernels_per_call.csv` (a separate `--mode count` pass as root; attaching "
                "a profiler inflates every later call, so it never shares a process with the timing).", ""]
        hl = rd(e11a / "host_layer_summary.csv")
        if hl:
            L += ["### Per-layer host sequence", ""] + table(hl, list(hl[0].keys())) + [""]
        ex = pick(rows, "t_step_extra", timing="wall_isolated", var_has=("gap_ms=0",))
        if ex:
            L += ["### Per-step extras (outside the layers)", ""]
            L += table([{"extra": r["key"], "exec_style": r["exec_style"], "median ms": f3(r["value"]),
                         "variant": r["variant"].split(";from=")[0]} for r in ex],
                       ["extra", "exec_style", "median ms", "variant"]) + [""]
        nm = am.get("not_measured")
        if nm:
            L += ["Not measured in E11a (verbatim from its meta.json):", ""]
            L += [f"* {x}" for x in (nm if isinstance(nm, list) else [nm])] + [""]

    # ---------------------------------------------------------------- E11c
    L += ["## E11b / E11c — transfers and the hand-over object", ""]
    for name, d in part(ps, "E11c_"):
        if d is None:
            L += [f"*{name}: {PENDING}.*", ""]
            continue
        if name.endswith("_xfer"):
            xs = rd(d / "xfer_exact.csv")
            L += [f"### Transfers at this model's exact boundary sizes — `{rel(d)}/xfer_exact.csv`", "",
                  "T010's three methods (pageable, pinned, zero-copy through one managed allocation) plus the "
                  "zero-copy device control, at the sizes this model's mapping moves (T011 step 2).", ""]
            L += table([{k: (f3(v) if k in ("median_ms", "p5_ms", "p95_ms", "GBps") else v) for k, v in r.items()}
                        for r in xs], ["method", "direction", "size_label", "size_bytes", "median_ms", "p5_ms",
                                       "p95_ms", "GBps", "n"]) + [""]
            continue
        L += [f"### Hand-over object — `{rel(d)}`", ""]
        kr = d / "kv_reuse.md"
        if kr.exists():
            txt = kr.read_text().strip().splitlines()
            ans = [x for x in txt if x.startswith("**")][:1]
            L += (ans or txt[1:3]) + ["", f"(full evidence: `{rel(kr)}`)", ""]
        for f, title in (("handover.csv", "What is handed over"), ("boundary.csv", "Per-step boundary tensors"),
                         ("kv_copy.csv", "Copy / re-materialisation cost"), ("cross_kv.csv", "Cross-attention K/V per block")):
            t = rd(d / f)
            if t:
                cols = [c for c in t[0].keys() if c not in ("stride", "contiguous", "source")]
                L += [f"**{title}** (`{f}`)", ""] + table(t, cols, maxrows=20) + [""]

    # ---------------------------------------------------------------- E11d
    L += ["## E11d — how the prefix scales with its input", ""]
    for name, d in part(ps, "E11d_"):
        if d is None:
            L += [f"*{name}: {PENDING}.*", ""]
            continue
        sw = rd(d / "prefix_sweep.csv")
        md = meta_of(d)
        L += [f"### {name.split(model + '_')[-1].replace('_prefixsweep', '')} — `{rel(d)}`", ""]
        piv = {}
        for r in sw:
            k = (r.get("n_images"), r.get("prompt_label"))
            dd = piv.setdefault(k, {"n_images": r.get("n_images"), "prompt": r.get("prompt_label"),
                                    "P_processed": r.get("P_processed"), "P_valid": r.get("P_valid"),
                                    "first_call_s": r.get("first_call_s")})
            dd[r["region"]] = f3(r["median_ms"])
        regs = []
        for r in sw:
            if r["region"] not in regs:
                regs.append(r["region"])
        L += table(list(piv.values()), ["n_images", "prompt", "P_processed", "P_valid"] + regs + ["first_call_s"]) + [""]
        fit = rd(d / "prefix_fit.csv")
        if fit:
            L += ["Linear fits (`prefix_fit.csv`):", ""] + table(fit, [c for c in fit[0] if c != "residuals"]) + [""]
        ref = rd(d / "raw" / "refused_points.csv")
        if ref:
            L += ["Points the checkpoint refused (`raw/refused_points.csv`, first line of each error):", ""]
            L += table([{"point": r["point"], "error": r["error"].splitlines()[0][:160]} for r in ref],
                       ["point", "error"]) + [""]
        for k in ("tokenizer_max_length", "prompts", "empty_cameras", "camera_keys_declared"):
            if md.get(k) not in (None, "", [], {}):
                L.append(f"* `{k}`: {str(md[k])[:300]}")
        if model.startswith("groot"):
            L.append("* GR00T's `vlm_prefix` here is the WHOLE backbone with vision inside (T012 step 4's definition); "
                     "the LeRobot `vlm_prefix` excludes the image encoder.")
        L.append("")

    # ---------------------------------------------------------------- E11e
    L += ["## E11e — what memory pressure costs the prefix", ""]
    for name, d in part(ps, "E11e_"):
        if d is None:
            L += [f"*{name}: {PENDING}.*", ""]
            continue
        L += [f"### {name.split(model + '_')[-1].replace('_mempressure', '')} — `{rel(d)}`", ""]
        mp = rd(d / "mempressure.csv")
        L += table([{**r, "median_ms": f3(r.get("median_ms")), "slowdown": f3(r.get("slowdown"), 4)} for r in mp],
                   ["load_label", "bg_GBps_alone", "bg_GBps_during", "pct_of_measured_peak", "median_ms", "p5_ms",
                    "p95_ms", "n", "slowdown"]) + [""]
        cu = [r for r in rd(d / "cpu_util.csv") if not r.get("scope", "").startswith("cpu")]
        if cu:
            L += ["**CPU utilisation per phase** (`cpu_util.csv`)", ""] + table(
                cu, ["phase", "scope", "pct", "median_pct", "p5_pct", "p95_pct", "n", "cores_busy"]) + [""]
        pw = [r for r in rd(d / "power.csv") if r.get("rail") == "module"]
        if pw:
            L += ["**Power per phase, module total** (`power.csv`; per-rail rows in the file)", ""] + table(
                pw, ["phase", "rail", "mean_mW", "median_mW", "p5_mW", "p95_mW", "n"]) + [""]

    # ---------------------------------------------------------------- E11f
    L += ["## E11f — throughput, GPU idle, and sharing the GPU", ""]
    for name, d in part(ps, "E11f_"):
        if d is None:
            L += [f"*{name}: {PENDING}.*", ""]
            continue
        L += [f"### `{rel(d)}`", ""]
        for f, title, cols in (
                ("throughput.csv", "Throughput (60 s back-to-back loops)",
                 ["mode", "phase", "median_ms", "p5_ms", "p95_ms", "n", "throughput_inf_per_s", "effective_exec_style"]),
                ("gpu_idle.csv", "GPU idle inside the action-expert region (nsys)",
                 ["mode", "N", "ae_total_ms_unprofiled", "kernel_busy_ms", "gpu_idle_frac", "gpu_idle_frac_corrected",
                  "profiling_inflation", "n_kernels_in_region", "note"]),
                ("concurrency.csv", "Sharing the GPU: host-segment sequence vs a concurrent prefix loop",
                 ["mode", "activity", "condition", "sharing", "median_ms", "non_sleep_ms", "slowdown",
                  "slowdown_non_sleep"])):
            t = rd(d / f)
            if t:
                L += [f"**{title}** (`{f}`)", ""] + table(t, [c for c in cols if c in t[0]]) + [""]

    # ---------------------------------------------------------------- consequences
    L += ["## What this changes for the hybrid model", ""]
    bl = []
    v = first_val(rows, "t_host_layer", key="sequence", style="eager", var_has=(lay,) if lay else ())
    g = first_val(rows, "t_host_layer", key="whole_layer_graph", style="cudagraph", var_has=(lay,) if lay else ())
    if v is not None and g is not None:
        bl.append(f"`T_host_layer`: the eager per-layer host sequence costs {f3(v)} ms ({f3(v / ta, 3)}× the "
                  f"{ta} ms assumption); one CUDA graph over the layer costs {f3(g)} ms ({f3(g / ta, 3)}×). The mapping "
                  "should be priced with the implementation it will actually run, not with one launch per segment.")
    if ks:
        bl.append(f"Kernels per host segment are {f3(vals[0][0])}–{f3(vals[-1][0])} eager, not 1: a per-kernel launch "
                  "term scales with these counts, not with the number of segments.")
    if kvb:
        rs = [r for r in pick(rows, "xfer_latency", key="pinned:h2d") if "size_label=kv_bytes_total" in r["variant"]]
        if rs:
            bl.append(f"Moving the hand-over object ({kvb:,.0f} B) once costs {f3(rs[0]['value'])} ms pinned; "
                      f"at {ba} GB/s it would cost {f3(kvb / ba / 1e6)} ms.")
    if kc:
        bl.append(f"{kc:,.0f} B are re-materialised every denoising step (`kv_copied_bytes_per_step`), which the "
                  "hybrid model currently treats as zero.")
    for style in ("eager", "compiled"):
        g100 = first_val(rows, "prefix_slowdown", key="gpu_hog_100", stat="value", style=style)
        sp = first_val(rows, "prefix_slowdown", key="gpu_spin_control", stat="value", style=style)
        if g100 is not None:
            bl.append(f"`prefix_slowdown` ({style}) is {f3(g100, 3)}× under a GPU bandwidth hog at 100 % duty, of "
                      f"which the GPU spin control (no bandwidth taken) alone gives {f3(sp, 3) if sp else '—'}×; "
                      "the assumption is 1.0.")
    L += [f"* {x}" for x in bl] if bl else [f"*{PENDING}*"]
    L += ["", "These are derived statements: each ratio divides a measured value in this report by the "
              "docs/00_context.md assumption named in *The answer first*.", ""]

    # ---------------------------------------------------------------- measured / derived / not measured
    L += ["## Measured, derived, not measured", "", "**Measured** (one result directory per part; newest dated "
          "directory as `bench/consolidate_e11.py` picks it):", ""]
    L += [f"* `{k}` → `{v.name if v else PENDING}`" for k, v in ps.items()]
    L += ["", "**Derived**: the ratios above; `t_ae_step` in the consolidated file as the least-squares slope of "
              "`ae_total` over N (`timing=slope`, bench/consolidate_e11.py); sums in the cross-model table of "
              "`reports/E11_models.md`.", "", "**Not measured**:", ""]
    absent = cmeta.get("quantities_absent", []) if c else []
    for q in absent:
        L.append(f"* `{q}` — {ABSENT_WHY.get(q, 'not produced by any part; see the parts summary.md files')}.")
    for name, m in metas:
        nm = m.get("not_measured")
        for s in (nm if isinstance(nm, list) else [nm] if nm and nm != "none" else []):
            if isinstance(s, str):
                L.append(f"* `{name}`: {s[:400]}")
    if not c:
        L.append(f"* the consolidated file: {PENDING}")
    miss = cmeta.get("parts_missing")
    if miss:
        L.append(f"* parts missing from the consolidation: {miss}")
    L += ["", "Reproduce: `python bench/consolidate_e11.py --model " + model + " --outdir results/<date>_E11_"
          + model + "_inputs` then `python bench/make_e11_reports.py --models " + model + "`.", ""]
    return "\n".join(L) + "\n"


# Corrections found on 2026-09-28, with the commit that fixed each. These are facts about the code history (git log),
# not measurements; the measured values they changed are in the tables above.
CORRECTIONS = {
    "pi05": [
        "E11a had no kernel counts: every `n_kernels` was 0 and `t_launch_per_kernel` was absent in "
        "results/20260924_E11a_pi05_hostseg. The 2026-09-28 run adds the `--mode count` pass (root) and the nsys "
        "host-sequence capture, then re-times with the counts; its layer timings reproduce the 2026-09-24 run within "
        "0.5 % (its meta.json notes).",
        "The exact-size transfers (T011 step 2) were never measured before; they are in "
        "results/20260928_E11c_pi05_handover_xfer (bench/xfer_exact.py, 91fbe0f).",
        "The consolidated file now includes the xfer_exact part and the restated T006 region latencies "
        "(bench/consolidate_e11.py, 2505107).",
    ],
    "smolvla": [
        "`attention_whole` in results/20260924_E11a_smolvla_hostseg timed a Python return: its callable returned the "
        "captured attention output (0.0156 ms, 0 kernels). It now replays the model's own forward_*attn_layer on the "
        "captured inputs (e23d4b3).",
        "Kernel counts were keyed by (segment, style), so layer 9's count overwrote layer 0's; SmolVLA's layers 0 and 9 "
        "are different kinds. Keyed by layer since be4b3b7.",
        "E11a had no kernel counts before 2026-09-28 (same gap as π0.5).",
        "The exact-size transfers (T011 step 2) are new: results/20260928_E11c_smolvla_handover_xfer.",
    ],
    "groot_n17": [
        "T012 was first closed with E11a only (a scope statement); steps 2–7 were measured on 2026-09-28 with "
        "bench/groot_e11.py (ace31e2).",
        "GR00T's DiT has 32 blocks (16 cross-attention, 16 self-attention), not the 16 T012 assumed "
        "(results/20260928_E11c_groot_n17_handover/kv_reuse.md, hook counts).",
        "GR00T hands the action head the backbone's hidden states, not a KV cache, and every cross-attention block "
        "re-projects them through to_k / to_v at every denoising step (kv_reuse.md).",
        "The first GR00T prefix sweep was superseded: its short and typical prompts were the same length and its "
        "3-image points were silently 2-image points (5359842).",
        "E11a is re-run with kernel counts, the nsys capture, the host sequence, whole_layer_graph and the per-step "
        "extras (bench/groot_segments.py, ee3d997).",
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="pi05,smolvla,groot_n17")
    ap.add_argument("--no-models-table", action="store_true",
                    help="do not regenerate reports/E11_models.md")
    args = ap.parse_args()
    for m in [x for x in args.models.split(",") if x]:
        out = REPO / "reports" / f"E11_{m}_hybrid_inputs.md"
        out.write_text(report(m))
        print(f"wrote {out.relative_to(REPO)}")
    if not args.no_models_table:
        import make_e11_models_report as MER  # noqa: PLC0415
        sys.argv = [sys.argv[0]]
        MER.main()
        p = REPO / "reports" / "E11_models.md"
        txt = p.read_text().splitlines()
        i = next((k for k, x in enumerate(txt) if x.startswith("## Conditions")), len(txt))
        p.write_text("\n".join(txt[:i] + cross_table() + txt[i:]) + "\n")
        print("inserted the cross-model table into reports/E11_models.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
