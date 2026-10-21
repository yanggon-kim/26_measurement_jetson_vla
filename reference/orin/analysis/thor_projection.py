#!/usr/bin/env python3
"""Orin -> Jetson AGX Thor latency projection (static analysis; nothing here is a Thor measurement).

Every output number is PROJECTED, NOT MEASURED.  Method and caveats: analysis/thor_projection.md.

Inputs read from committed files of this repo (paths relative to the repo root):
  reports/SYNTHESIS_all_measurements.md  §1.1  per-stage Orin medians, six models, base and best
  results/20260925_E12b_pi0_green_corun_N10_MAXN/corun_ratios.csv  pi0 A_iso, B_iso, O (green 12/4)
  reports/E1_E2_microbench.md            measured Orin roof 187.47 GB/s, pi0 expert weights 622.85 MB/step
External constants (hard-coded below, each with source and access date 2026-10-08).

Writes analysis/thor_projection/{factors,per_model,geomean,pi0_aim,validation}.csv and prints the tables.
Run from anywhere:  python3 analysis/thor_projection.py
"""
import csv
import math
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "thor_projection")

# ---------------------------------------------------------------------------------------------
# External calibration constants (accessed 2026-10-08)
# ---------------------------------------------------------------------------------------------
# (T) NVIDIA Jetson AI Lab, "GR00T N1.7 on Thor", https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/
#     Jetson AGX Thor devkit, JetPack 7.2, MAXN, GR00T-N1.7-LIBERO/libero_10, LIBERO_PANDA, 1 camera,
#     4 denoising steps, batch 1, medians of 20 after 5 warm-up.  (backbone, action head) ms.
THOR_JP72 = {"eager": (47.7, 68.2), "compile": (48.6, 46.8)}
# (N) NVIDIA Isaac-GR00T scripts/deployment/README.md @ 376ba89 (the pinned clone,
#     microarch/01_models/Isaac-GR00T; https://github.com/NVIDIA/Isaac-GR00T/blob/376ba89/scripts/deployment/README.md),
#     "GR00T N1.7 Inference Timing (4 denoising steps, 1 camera)", same harness (benchmark_inference.py),
#     same checkpoint libero_10 on both devices.  Thor there is JetPack 7.1 (README "Tested with JetPack 7.1").
NV_ORIN = {"eager": (127.6, 205.39), "compile": (128.59, 78.94)}
NV_THOR_JP71 = {"eager": (55.26, 81.65), "compile": (55.59, 64.66)}

# Hardware (Thor T5000): 2560 CUDA cores in 10 TPCs -> 20 SMs (derived, 128 cores/SM); GPU up to 1.57 GHz
# (ASPLOS'27 sweep top 1575 MHz); 273 GB/s, 256-bit LPDDR5X, EMC 4266 MHz.  BF16 peak not published: none derived.
#   https://flopper.io/gpu/nvidia-jetson-agx-thor-t5000-128gb/spec-sheet.pdf
#   https://www.servethehome.com/nvidia-jetson-agx-thor-developer-kit-blackwell-for-robotics/2/
#   Jung et al., ASPLOS'27, arXiv 2610.05062 (local: microarch/00_doc/02_refs/related_work/asplos_hyuk.pdf) §4.3
THOR_SM, THOR_MHZ, THOR_GBPS = 20, 1575.0, 273.0
ORIN_SM, ORIN_MHZ, ORIN_GBPS = 16, 1300.5, 204.8   # E0 (MAXN gate) / spec
SMCLK = (THOR_SM * THOR_MHZ) / (ORIN_SM * ORIN_MHZ)  # 1.514
BW = THOR_GBPS / ORIN_GBPS                           # 1.333

OTHER_F = 1.5  # [ASSUMPTION] host pre/post-processing between timed regions speeds up 1.5x (CPU uplift)

# Validation targets.  ASPLOS'27 Table 7 RTTmax (ms), Orin / Thor, LIBERO checkpoints, compiled, JetPack 6.2 / 7.0;
# RTT covers observation transfer + server pre/post-processing + compiled inference.
ASPLOS_RTT = {"π0.5": (457.3, 180.6), "GR00T N1.6": (206.8, 126.7)}
# NVIDIA openpi pi0.5 on Thor, https://www.jetson-ai-lab.com/tutorials/openpi_on_thor (JP 7.2, MAXN, pi05_libero,
# torch.compile BF16): total 132.00 ms, model 128.34 ms.  openpi implementation, not LeRobot: context only.
OPENPI_THOR = (132.00, 128.34)

# PIM side, from the workspace's evaluation/p1/numbers.md (ids c5_fd_5ns.FD-16x800.step_ms,
# c5_fd_c32.5ns/FD-16x800-c32.step_ms, constants.gpu_optimistic_2x_step_ms).  [ASSUMPTION] unchanged on
# LPDDR5X: the AiM step is set by the DRAM-core column period (5 ns), which does not scale with the I/O rate.
AIM_STEP = {16: 9.31, 32: 5.87}
GPU_OPT_2X_ORIN = 6.30  # "optimistic bound (2x floor)", ms/step, Orin
N_PI0 = 10


# ---------------------------------------------------------------------------------------------
# Committed Orin inputs
# ---------------------------------------------------------------------------------------------
def read_synthesis():
    txt = open(os.path.join(ROOT, "reports", "SYNTHESIS_all_measurements.md")).read()
    sec = txt.split("### 1.1", 1)[1].split("\n### ", 1)[0]
    rows, model, nsteps = {}, None, None
    for line in sec.splitlines():
        if not line.startswith("|") or line.startswith("|---") or "config" in line:
            continue
        c = [x.strip().replace("*", "") for x in line.strip().strip("|").split("|")]
        if c[0]:
            m = re.match(r"(.+?) \((\d+)\)", c[0])
            model, nsteps = m.group(1), int(m.group(2))
        cfg = "base" if c[1].startswith("base") else "best"
        label = re.sub(r"^(base|best)\s*", "", c[1]).strip("` ") or "eager_fp32"
        v, p, a, _step, e = (float(c[i]) for i in range(2, 7))
        rows.setdefault(model, {"N": nsteps})[cfg] = dict(vis=v, pre=p, ae=a, e2e=e, label=label)
    assert len(rows) == 6, rows.keys()
    return rows


def read_e12b():
    p = os.path.join(ROOT, "results", "20260925_E12b_pi0_green_corun_N10_MAXN", "corun_ratios.csv")
    for r in csv.DictReader(open(p)):
        if r["split"] == "12/4" and r["valid"] == "True":
            return float(r["A_iso_ms"]), float(r["B_iso_ms"]), float(r["O"])
    raise SystemExit("E12b green 12/4 row not found")


def read_e1():
    t = open(os.path.join(ROOT, "reports", "E1_E2_microbench.md")).read()
    roof = float(re.search(r"\| memory bandwidth \| 204\.8 GB/s \| \*\*([\d.]+) GB/s", t).group(1))
    mb = float(re.search(r"analytic floor \(([\d.]+) MB ÷", t).group(1))
    return roof, mb


def check_aim():
    """If the workspace's evaluation file is reachable, confirm the hard-coded PIM constants."""
    p = os.path.join(ROOT, "..", "evaluation", "p1", "numbers.md")
    if not os.path.exists(p):
        return "evaluation/p1/numbers.md not reachable; constants as hard-coded"
    t = open(p).read()
    for ch, ident in ((16, "c5_fd_5ns.FD-16x800.step_ms"), (32, "c5_fd_c32.5ns/FD-16x800-c32.step_ms")):
        m = re.search(r"\| ([\d.]+) ms \| `" + re.escape(ident) + "`", t)
        assert m and float(m.group(1)) == AIM_STEP[ch], (ch, m and m.group(1))
    m = re.search(r"\| ([\d.]+) ms \| `constants\.gpu_optimistic_2x_step_ms\.value`", t)
    assert m and float(m.group(1)) == GPU_OPT_2X_ORIN
    return "AiM 9.31 / 5.87 ms and 6.30 ms confirmed against evaluation/p1/numbers.md"


# ---------------------------------------------------------------------------------------------
def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def write(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([f"{x:.4g}" if isinstance(x, float) else x for x in r])


def main():
    syn = read_synthesis()
    A, B, O = read_e12b()
    roof, ae_mb = read_e1()
    aim_note = check_aim()
    n17 = syn["GR00T N1.7"]

    # --- calibration factors (front, expert) per config; base <- eager, best <- torch.compile ---------------
    mode = {"base": "eager", "best": "compile"}
    ratio = lambda o, t: (o[0] / t[0], o[1] / t[1])
    ours = {c: (n17[c]["vis"] + n17[c]["pre"], n17[c]["ae"]) for c in ("base", "best")}
    scen = {  # name: {cfg: (front, expert)}
        "central": {c: ratio(NV_ORIN[mode[c]], THOR_JP72[mode[c]]) for c in mode},
        "low": {"base": (SMCLK, SMCLK), "best": (SMCLK, BW)},
        "jp71": {c: ratio(NV_ORIN[mode[c]], NV_THOR_JP71[mode[c]]) for c in mode},
        "ours_vs_jp72": {c: ratio(ours[c], THOR_JP72[mode[c]]) for c in mode},
    }
    desc = {
        "central": "NVIDIA Orin (README @376ba89) / NVIDIA Thor JP 7.2 tutorial; same input (libero_10, 1 camera, 4 steps)",
        "low": "hardware ratios: front SMxclock 20x1575/(16x1300.5); expert bandwidth 273/204.8 (best), SMxclock (base)",
        "jp71": "NVIDIA Orin / NVIDIA Thor JP 7.1, both from README @376ba89 (same harness, same input)",
        "ours_vs_jp72": "main's draft: our Orin N1.7 (2 cameras x 2 frames, P=273, S=41) / NVIDIA Thor JP 7.2 (1 camera): input-confounded",
    }
    fac_rows = []
    print(f"SMxclock ratio {SMCLK:.3f}   bandwidth ratio {BW:.3f}")
    print("\ncalibration factors (front = vision + VLM prefix, expert = action head)")
    for s, d in scen.items():
        for c in ("base", "best"):
            fac_rows.append([s, c, d[c][0], d[c][1], desc[s]])
            print(f"  {s:13s} {c:4s} front x{d[c][0]:.3f}  expert x{d[c][1]:.3f}")
    write("factors.csv", ["scenario", "config", "front_factor", "expert_factor", "source"], fac_rows)

    def proj(r, f):
        v, p, a = r["vis"] / f[0], r["pre"] / f[0], r["ae"] / f[1]
        o = (r["e2e"] - r["vis"] - r["pre"] - r["ae"]) / OTHER_F
        e = v + p + a + o
        return v, p, a, o, e

    # --- per model -------------------------------------------------------------------------------------------
    pm, geo = [], []
    for c in ("best", "base"):
        print(f"\n== {c} (PROJECTED, NOT MEASURED) — central: vision / prefix / AE / e2e ms, Hz, AE share, ms/step;"
              " then low-scenario e2e and AE share; Orin e2e and Orin/Thor ratio")
        shares = {"orin": [], "central": [], "low": [], "jp71": [], "ours_vs_jp72": []}
        for m, d in syn.items():
            r = d[c]
            v, p, a, o, e = proj(r, scen["central"][c])
            lv, lp, la, lo, le = proj(r, scen["low"][c])
            _, _, ja, _, je = proj(r, scen["jp71"][c])
            _, _, xa, _, xe = proj(r, scen["ours_vs_jp72"][c])
            shares["jp71"].append(ja / je); shares["ours_vs_jp72"].append(xa / xe)
            sh_o = r["ae"] / r["e2e"]
            shares["orin"].append(sh_o); shares["central"].append(a / e); shares["low"].append(la / le)
            pm.append([m, c, r["label"], d["N"], r["vis"], r["pre"], r["ae"], r["e2e"], sh_o,
                       v, p, a, o, e, 1000 / e, a / e, a / d["N"], r["e2e"] / e,
                       la, le, 1000 / le, la / le, r["e2e"] / le, je, xe])
            print(f"  {m:11s} {v:6.1f} {p:7.1f} {a:6.1f} {e:7.1f} {1000/e:5.1f} Hz {100*a/e:5.1f} % {a/d['N']:6.2f} | "
                  f"low {le:7.1f} {100*la/le:5.1f} % | Orin {r['e2e']:7.1f}  x{r['e2e']/e:.2f}")
        g = {k: gmean(v) for k, v in shares.items()}
        geo.append([c, g["orin"], g["central"], g["low"], g["jp71"], g["ours_vs_jp72"]])
        print(f"  geomean AE share: Orin {100*g['orin']:.1f} %  Thor central {100*g['central']:.1f} %  low {100*g['low']:.1f} %"
              f"  jp71 {100*g['jp71']:.1f} %  ours_vs_jp72 {100*g['ours_vs_jp72']:.1f} %")
    write("per_model.csv",
          ["model", "config", "config_label", "N", "orin_vision_ms", "orin_prefix_ms", "orin_ae_ms", "orin_e2e_ms",
           "orin_ae_share", "thor_vision_ms", "thor_prefix_ms", "thor_ae_ms", "thor_other_ms", "thor_e2e_ms",
           "thor_hz", "thor_ae_share", "thor_ae_ms_per_step", "orin_over_thor_e2e",
           "low_ae_ms", "low_e2e_ms", "low_hz", "low_ae_share", "low_orin_over_thor_e2e",
           "jp71_e2e_ms", "ours_vs_jp72_e2e_ms"], pm)
    write("geomean.csv", ["config", "orin_ae_share", "thor_central_ae_share", "thor_low_ae_share",
                          "thor_jp71_ae_share", "thor_ours_vs_jp72_ae_share"], geo)

    # --- validation ------------------------------------------------------------------------------------------
    val = []
    print("\nvalidation (best config, Orin/Thor e2e ratio)")
    for m, (ro, rt) in ASPLOS_RTT.items():
        r = syn[m]["best"]
        rc, rl = r["e2e"] / proj(r, scen["central"]["best"])[4], r["e2e"] / proj(r, scen["low"]["best"])[4]
        rj, rx = r["e2e"] / proj(r, scen["jp71"]["best"])[4], r["e2e"] / proj(r, scen["ours_vs_jp72"]["best"])[4]
        val.append([m, ro, rt, ro / rt, rc, rl, rj, rx])
        print(f"  {m:11s} ASPLOS RTTmax {ro}/{rt} = x{ro/rt:.2f};  projected central x{rc:.2f}  low x{rl:.2f}"
              f"  jp71 x{rj:.2f}  ours_vs_jp72 x{rx:.2f}")
    p05 = proj(syn["π0.5"]["best"], scen["central"]["best"])[4]
    print(f"  pi0.5 best central projection {p05:.1f} ms vs NVIDIA openpi Thor torch.compile {OPENPI_THOR[0]} ms total"
          f" ({OPENPI_THOR[1]} model) — different implementation, context only")
    write("validation.csv", ["model", "asplos_rtt_orin_ms", "asplos_rtt_thor_ms", "asplos_ratio",
                             "proj_ratio_central", "proj_ratio_low", "proj_ratio_jp71", "proj_ratio_ours_vs_jp72"], val)

    # --- pi0 GPU vs AiM on Thor (E12b basis) -----------------------------------------------------------------
    floor_orin = ae_mb / roof                 # ms/step at the measured Orin roof
    floor_thor = floor_orin / BW              # [ASSUMPTION] Thor reaches the same 0.915 fraction of its spec
    opt2x_thor = GPU_OPT_2X_ORIN / BW
    prefix_floor_thor = 21.64e3 / (roof * BW)  # E12b prefix bytes/frame (21.64 GB) if the re-reads persist on Thor
    fs = scen["central"]["best"]
    cases = {"central": fs, "low": (SMCLK, BW), "high_expert": (fs[0], SMCLK * BW),
             "jp71": scen["jp71"]["best"], "ours_vs_jp72": scen["ours_vs_jp72"]["best"]}
    aim_rows = []
    print(f"\npi0 on Thor, E12b basis (A_iso {A:.2f}, B_iso {B:.2f} ms, O {O:.3f}); {aim_note}")
    print(f"  expert floor {floor_orin:.3f} -> {floor_thor:.2f} ms/step; optimistic bound (2x floor) "
          f"{GPU_OPT_2X_ORIN:.2f} -> {opt2x_thor:.2f} ms/step; prefix floor if re-reads persist {prefix_floor_thor:.1f} ms")
    for lab, (fa, fb) in cases.items():
        At, Bt = A / fa, B / fb
        ser, co = At + Bt, max(At, Bt) * O
        row = [lab, fa, fb, At, Bt, Bt / N_PI0, ser, co]
        line = f"  [{lab:12s}] A {At:6.1f}  B {Bt:6.1f} ({Bt/N_PI0:.2f}/step)  serial {ser:6.1f}  co-run~{co:6.1f}"
        for ch in (16, 32):
            lat, per = At + N_PI0 * AIM_STEP[ch], max(At, N_PI0 * AIM_STEP[ch])
            row += [lat, lat / ser, per]
            line += f" | AiM{ch} lat {lat:6.1f} (x{lat/ser:.3f}) period>={per:5.1f}"
        aim_rows.append(row)
        print(line)
    write("pi0_aim.csv", ["scenario", "front_factor", "expert_factor", "thor_A_ms", "thor_B_gpu_expert_ms",
                          "thor_B_ms_per_step", "gpu_serial_ms", "gpu_corun_est_ms",
                          "aim16_latency_ms", "aim16_over_gpu_serial", "aim16_period_lb_ms",
                          "aim32_latency_ms", "aim32_over_gpu_serial", "aim32_period_lb_ms"], aim_rows)
    write("pi0_aim_constants.csv", ["quantity", "value", "unit"], [
        ["orin_expert_floor", floor_orin, "ms/step"], ["thor_expert_floor", floor_thor, "ms/step"],
        ["orin_optimistic_bound_2x_floor", GPU_OPT_2X_ORIN, "ms/step"],
        ["thor_optimistic_bound_2x_floor", opt2x_thor, "ms/step"],
        ["thor_prefix_floor_if_rereads_persist", prefix_floor_thor, "ms"],
        ["aim_step_16ch", AIM_STEP[16], "ms"], ["aim_step_32ch", AIM_STEP[32], "ms"],
        ["E12b_A_iso", A, "ms"], ["E12b_B_iso", B, "ms"], ["E12b_O_green_12_4", O, ""]])


if __name__ == "__main__":
    main()
