"""T005 steps 3-5: choose best_bf16 per model from the committed rows, write best_bf16.json, phase_balance.csv and
candidates.csv into results/<date>_E4i_summary/, and generate reports/E4d_phase_balance.md (every number from the rows).

Sources per model (raw/rows/*.json written by bench/opt_ladder.py, equivalence.csv by bench/t005_finalise.py):
  the ladder   results/*_E4c_<m>_opt_ladder/ (LeRobot) or results/*_E4d_<m>_ladder/ (GR00T)
  the search   results/*_E4i_<m>_best_search/
  the confirmation run of the chosen candidate, if present: results/*_E4i_<m>_best_confirm/

Eligibility for best_bf16 (rules fixed before choosing; every exclusion is listed with its reason in candidates.csv):
  1. measured at the model's default N, with the per-part split (regions on: vision, vlm_prefix, ae_total present);
  2. no graph compiled during the timed loop; not throttled; the timed block started in the normal memory-path state;
  3. BF16 weights: fp32 parameters <= 1 % of all parameter bytes (the rest stays fp32 for a stated reason);
  4. actions finite, and relative L2 vs eager_fp32 (10 seeded inferences, equivalence.csv) <= max(1e-2, 1.5 x the
     model's compile-only reference): `ae_graphs` measured in the search directory (GR00T: with fallback_random, so
     seeded noise matches eager; the ladder's compiled GR00T rows drew different noise) or in the ladder (LeRobot).
     Fixed before the search results were read; plain compilation alone sets each model's numerical spread.
best_bf16 = the eligible row with the lowest e2e median. Reported beside it: the fastest row overall (any rung, including
e2e-only whole-policy compiles), the fastest eligible row on LeRobot's own code path (no attention-backend switch), and
the eligible row with the largest action-expert share.

python bench/e4i_summary.py --date 20261022 --out results/20261022_E4i_summary --report reports/E4d_phase_balance.md
"""
import argparse
import csv
import glob
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["pi0", "pi05", "smolvla", "groot_n17", "groot_n16", "groot_n15"]
NAMES = {"pi0": "π0", "pi05": "π0.5", "smolvla": "SmolVLA", "groot_n17": "GR00T N1.7", "groot_n16": "GR00T N1.6",
         "groot_n15": "GR00T N1.5"}
N_DEFAULT = {"pi0": 10, "pi05": 10, "smolvla": 10, "groot_n17": 4, "groot_n16": 4, "groot_n15": 4}
REL_L2_MAX, FP32_MAX = 1e-2, 0.01


def dirs_for(m):
    pats = ([f"results/*_E4c_{m}_opt_ladder"] if not m.startswith("groot") else [f"results/*_E4d_{m}_ladder"]) + \
        [f"results/*_E4i_{m}_best_search", f"results/*_E4i_{m}_best_confirm"]
    return [d for p in pats for d in sorted(glob.glob(os.path.join(REPO, p)))]


def load_eq(d):
    eq = {}
    p = os.path.join(d, "equivalence.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            eq.setdefault(r["rung_N"], {})[r["reference"].split(":")[-1].rsplit("_N", 1)[0]] = r
    return eq


def rows_for(m):
    out = []
    for d in dirs_for(m):
        eq = load_eq(d)
        for f in sorted(glob.glob(os.path.join(d, "raw", "rows", "*.json"))):
            r = json.load(open(f))
            if int(r["N"]) != N_DEFAULT[m]:
                continue
            r["_dir"] = os.path.relpath(d, REPO)
            r["_eq"] = eq.get(f"{r['rung']}_N{r['N']}", {})
            out.append(r)
    return out


def part(r, k, q="median_ms"):
    b = (r.get("by_N") or {}).get(str(r["N"])) or (r.get("by_N") or {}).get(r["N"]) or {}
    st = b.get("regions", {}).get(k)
    return st[q] if st else None


def block(r):
    return (r.get("by_N") or {}).get(str(r["N"])) or (r.get("by_N") or {}).get(r["N"]) or {}


def eligibility(r):
    why = []
    if r["status"] != "measured":
        return [f"{r['status']}: {(r.get('not_applicable_reason') or r.get('error') or '')[:120]}"]
    if not all(part(r, k) is not None for k in ("vision", "vlm_prefix", "ae_total")):
        why.append("no per-part split (whole-policy compile, e2e only)")
    b = block(r)
    if b.get("graphs_compiled_during_timed", 0):
        why.append(f"{b['graphs_compiled_during_timed']} graphs compiled during the timed loop")
    if b.get("throttle", {}).get("throttled"):
        why.append("throttled")
    if b.get("memory_state_before", {}).get("state") != "normal":
        why.append("did not start in the normal memory-path state")
    pb = r.get("param_dtype_bytes", {})
    tot = sum(pb.values()) or 1
    fp32 = pb.get("torch.float32", 0) / tot
    if fp32 > FP32_MAX:
        why.append(f"not BF16: {fp32 * 100:.1f} % of parameter bytes fp32")
    if r.get("actions_finite") is False:
        why.append("non-finite actions")
    e = r["_eq"].get("eager_fp32")
    bound = r.get("_bound", REL_L2_MAX)
    if e is None:
        why.append("no equivalence row vs eager_fp32")
    else:
        try:
            if float(e["rel_l2"]) > bound:
                why.append(f"actions differ from eager_fp32 by rel L2 {float(e['rel_l2']):.2e} > bound {bound:.2e}")
        except ValueError:
            why.append(f"equivalence: {e['max_abs']}")
    return why


def own_path(r):
    return not any("attn:" in p or "thor_" in p for p in ((r.get("rung_desc") or {}).get("preps") or [])) and \
        "_attn_" not in r["rung"]


def orin_tables():
    """The Orin's E4d rung tables and E4i best rows (reference/orin/reports), parsed, not typed."""
    txt = open(os.path.join(REPO, "reference/orin/reports/E4d_phase_balance.md")).read()
    key = {"π0": "pi0", "π0.5": "pi05", "SmolVLA": "smolvla", "GR00T N1.7": "groot_n17", "GR00T N1.6": "groot_n16",
           "GR00T N1.5": "groot_n15"}
    out = {}
    for sec in re.split(r"\n### ", txt)[1:]:
        head = sec.split("\n", 1)[0]
        mm = re.match(r"(.+?) \(N = (\d+)\)", head)
        if not mm or mm.group(1) not in key:
            continue
        rungs = {}
        for ln in sec.splitlines():
            c = [x.strip() for x in ln.strip().strip("|").split("|")]
            if len(c) >= 5 and c[0].startswith("`"):
                try:
                    rungs[c[0].strip("`")] = {k: float(v) if v not in ("—", "") else None
                                              for k, v in zip(("e2e", "vision", "prefix", "expert"), c[1:5])}
                except ValueError:
                    pass
        out[(key[mm.group(1)], int(mm.group(2)))] = rungs
    best = {}
    t2 = open(os.path.join(REPO, "reference/orin/reports/E4i_best_combinations.md")).read()
    sec = t2.split("## Best configuration per model")[1].split("##")[0]
    for ln in sec.splitlines():
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) >= 7 and c[0] in key:
            best[key[c[0]]] = {"rung": c[2].strip("`"), "e2e": float(c[3].strip("*")), "vision": float(c[4]),
                               "prefix": float(c[5]), "expert": float(c[6])}
    return out, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    orin_rungs, orin_best = orin_tables()
    best_json = {"rules": {"eligibility": __doc__.split("Eligibility for best_bf16")[1].split("python bench")[0].strip()},
                 "models": {}}
    pb_rows, cand_rows, sections = [], [], []
    for m in MODELS:
        rows = rows_for(m)
        if not rows:
            continue
        refs = [r for r in rows if r["rung"] == "ae_graphs" and r["status"] == "measured" and r["_eq"].get("eager_fp32")
                and (not m.startswith("groot") or "_E4i_" in r["_dir"])]
        ref_l2 = float(refs[-1]["_eq"]["eager_fp32"]["rel_l2"]) if refs else 0.0
        bound = max(REL_L2_MAX, 1.5 * ref_l2)
        for r in rows:
            r["_bound"] = bound
            r["_why"] = eligibility(r)
        elig = [r for r in rows if not r["_why"]]
        meas = [r for r in rows if r["status"] == "measured"]
        eager = next((r for r in rows if r["rung"] == "eager_fp32" and r["status"] == "measured"), None)
        best = min(elig, key=lambda r: part(r, "e2e_wall")) if elig else None
        fastest_any = min(meas, key=lambda r: part(r, "e2e_wall")) if meas else None
        own = [r for r in elig if own_path(r)]
        best_own = min(own, key=lambda r: part(r, "e2e_wall")) if own else None
        share = lambda r: part(r, "ae_total") / part(r, "model_get_action" if m.startswith("groot") else "e2e_wall")  # noqa: E731
        max_share = max(elig, key=share) if elig else None
        for r in rows:
            cand_rows.append({"model": m, "N": r["N"], "rung": r["rung"], "dir": r["_dir"], "status": r["status"],
                              "e2e_ms": part(r, "e2e_wall") or "", "eligible": not r["_why"],
                              "why_not": "; ".join(r["_why"]),
                              "rel_l2_vs_eager_fp32": r["_eq"].get("eager_fp32", {}).get("rel_l2", ""),
                              "own_lerobot_path": own_path(r)})
        if best:
            d = best.get("rung_desc") or {}
            best_json["models"][m] = {
                "rung": best["rung"], "N": best["N"], "timed_in": best["_dir"],
                "label": d.get("label"), "compiled": d.get("compiled"), "preps": d.get("preps"),
                "precision_flags": d.get("precision_flags"), "autotune": d.get("autotune") or "off",
                "regions": d.get("regions"), "mpad_table": best.get("mpad_table", "n/a"),
                "e2e_ms": part(best, "e2e_wall"), "equivalence_bound_rel_l2": bound,
                "compile_reference": refs[-1]["_dir"] + ":ae_graphs" if refs else "none", "reproduce": f"bench/opt_ladder.py --model {m} --rung best_bf16 --out <dir>",
                "fastest_overall": {"rung": fastest_any["rung"], "e2e_ms": part(fastest_any, "e2e_wall"),
                                    "dir": fastest_any["_dir"], "eligible": not fastest_any["_why"],
                                    "why_not": "; ".join(fastest_any["_why"])},
                "fastest_on_lerobot_own_path": ({"rung": best_own["rung"], "e2e_ms": part(best_own, "e2e_wall")}
                                                if best_own else "n/a"),
                "max_ae_share": {"rung": max_share["rung"], "ae_share": share(max_share)}}
        for mode, r in (("eager_fp32", eager), ("best_bf16", best)):
            if r is None:
                continue
            pb = {"model": m, "N": r["N"], "mode": mode, "rung": r["rung"], "dir": r["_dir"]}
            for k, col in (("vision", "vision"), ("vlm_prefix", "prefix"), ("ae_total", "expert"), ("other", "other"),
                           ("e2e_wall", "e2e"), ("model_get_action", "model_call")):
                if part(r, k) is not None:
                    pb[f"{col}_ms"] = round(part(r, k), 3)
                    pb[f"{col}_p5_ms"] = round(part(r, k, "p5_ms"), 3)
                    pb[f"{col}_p95_ms"] = round(part(r, k, "p95_ms"), 3)
            pb["n"] = part(r, "e2e_wall", "n")
            pb["ae_share_of_e2e"] = round(part(r, "ae_total") / part(r, "e2e_wall"), 4)
            if part(r, "model_get_action") is not None:
                pb["ae_share_of_model_call"] = round(part(r, "ae_total") / part(r, "model_get_action"), 4)
            pb_rows.append(pb)
        sections.append((m, rows, eager, best, fastest_any, best_own, max_share))
    json.dump(best_json, open(os.path.join(a.out, "best_bf16.json"), "w"), indent=1, default=str)
    for name, rows in (("phase_balance.csv", pb_rows), ("candidates.csv", cand_rows)):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    write_report(a.report, sections, pb_rows, orin_rungs, orin_best)
    print(json.dumps({m: v["rung"] for m, v in best_json["models"].items()}, indent=1))


def f2(x):
    return "—" if x is None else f"{x:.1f}"


def write_report(path, sections, pb_rows, orin_rungs, orin_best):
    L = ["# E4d — the phase balance on Thor: baseline vs `best_bf16`, and every rung (T005)", "",
         "**Generated by `bench/e4i_summary.py` from the committed rows** (`raw/rows/*.json` of each ladder and search "
         "directory). No number is typed. Medians of n = 100 timed inferences, each configuration in its own process; "
         "MAXN, `jetson_clocks`, gate passed. GR00T: e2e = `Gr00tPolicy.get_action` (includes CPU processing); the AE "
         "share is also given of the model call (`model.get_action`), which is what the Orin's GR00T e2e measured.", "",
         "## 1. `eager_fp32` vs `best_bf16` per model", "",
         "| model | N | mode | rung | e2e | vision | prefix | expert | other | AE % of e2e | AE % of model call |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in pb_rows:
        L.append(f"| {NAMES[r['model']]} | {r['N']} | {r['mode']} | `{r['rung']}` | {r.get('e2e_ms', '—')} | "
                 f"{r.get('vision_ms', '—')} | {r.get('prefix_ms', '—')} | {r.get('expert_ms', '—')} | {r.get('other_ms', '—')} | "
                 f"{r['ae_share_of_e2e'] * 100:.1f} | {r['ae_share_of_model_call'] * 100:.1f} |"
                 if "ae_share_of_model_call" in r else
                 f"| {NAMES[r['model']]} | {r['N']} | {r['mode']} | `{r['rung']}` | {r.get('e2e_ms', '—')} | "
                 f"{r.get('vision_ms', '—')} | {r.get('prefix_ms', '—')} | {r.get('expert_ms', '—')} | {r.get('other_ms', '—')} | "
                 f"{r['ae_share_of_e2e'] * 100:.1f} | — |")
    L += ["", "## 2. Thor ÷ Orin per part (Orin values parsed from `reference/orin/reports/E4d_phase_balance.md` and "
          "`E4i_best_combinations.md`; GR00T e2e on Thor = the model call, the Orin's definition)", "",
          "| model | mode | Orin rung | e2e | vision | prefix | expert |", "|---|---|---|---|---|---|---|"]
    for m, rows, eager, best, *_ in sections:
        for mode, r in (("eager_fp32", eager), ("best_bf16", best)):
            if r is None:
                continue
            o = orin_rungs.get((m, r["N"]), {}).get("eager_fp32") if mode == "eager_fp32" else orin_best.get(m)
            if not o:
                continue
            e2e = part(r, "model_get_action") if m.startswith("groot") else part(r, "e2e_wall")
            pre = (part(r, "vlm_backbone") - part(r, "vision")) if m.startswith("groot") else part(r, "vlm_prefix")
            vals = [(o["e2e"], e2e), (o["vision"], part(r, "vision")), (o["prefix"], pre), (o["expert"], part(r, "ae_total"))]
            L.append(f"| {NAMES[m]} | {mode} | `{o.get('rung', 'eager_fp32')}` | " +
                     " | ".join(f"{a / b:.2f}×" if a and b else "—" for a, b in vals) + " |")
    L += ["", "GR00T prefix in this table = Thor's `vlm_backbone − vision` (the Orin's definition, which includes the "
          "unused `lm_head`).", ""]
    L += ["## 3. Does the action expert dominate the optimised frame?", ""]
    for m, rows, eager, best, fastest_any, best_own, max_share in sections:
        if not best:
            L.append(f"- {NAMES[m]}: no eligible best_bf16 row yet.")
            continue
        parts = {"vision": part(best, "vision"), "prefix": part(best, "vlm_prefix"), "expert": part(best, "ae_total")}
        big = max(parts, key=parts.get)
        den = part(best, "model_get_action") if m.startswith("groot") else part(best, "e2e_wall")
        L.append(f"- {NAMES[m]}: best_bf16 `{best['rung']}` {part(best, 'e2e_wall'):.1f} ms; expert "
                 f"{parts['expert']:.1f} ms = {parts['expert'] / den * 100:.1f} % of "
                 f"{'the model call' if m.startswith('groot') else 'e2e'}; largest part: **{big}**"
                 f" ({parts[big]:.1f} ms). Fastest of any rung: `{fastest_any['rung']}` {part(fastest_any, 'e2e_wall'):.1f} ms"
                 + ("" if not fastest_any["_why"] else f" (not eligible: {'; '.join(fastest_any['_why'])})")
                 + (f"; fastest on LeRobot's own code path: `{best_own['rung']}` {part(best_own, 'e2e_wall'):.1f} ms"
                    if best_own and best_own is not best else "") + ".")
    L += ["", "## 4. Every rung and candidate (default N)", ""]
    for m, rows, *_ in sections:
        L += [f"### {NAMES[m]}", "", "| rung | dir | status | e2e | vision | prefix | expert | other | eligible for best_bf16 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: (part(r, "e2e_wall") is None, part(r, "e2e_wall") or 0)):
            L.append(f"| `{r['rung']}` | `{os.path.basename(r['_dir'])}` | {r['status']} | {f2(part(r, 'e2e_wall'))} | "
                     f"{f2(part(r, 'vision'))} | {f2(part(r, 'vlm_prefix'))} | {f2(part(r, 'ae_total'))} | {f2(part(r, 'other'))} | "
                     f"{'yes' if not r['_why'] else 'no: ' + '; '.join(r['_why'])[:150]} |")
        L.append("")
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
