"""Shared pieces of E12b (T015): gate, rung, loops, finite check, provenance.

Kept in one place so M1-M5 all go through ONE loader, ONE gate and ONE loop timer; a second copy
would eventually differ in a detail and the ratios between measurements would stop meaning what
they say (the lesson of groot_phases.load_for_segments).
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))

import pi0_common as P          # noqa: E402

E7_META = REPO / "results/20260924_E7_pi0_N10_bf16_both_MAXN/meta.json"
LEROBOT_COMMIT = "f0d2b37bebddf7e8852cb624712da0ad564601b6"
CKPT_COMMIT = "45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f"

# Variant tokens (spec §11). Every row and meta.json carries the '+'-joined label.
VARIANT_TOKENS = {
    "bf16_both": "E7's rung exactly (casts + model.denoise_step@reduce-overhead, "
                 "pg.embed_image@reduce-overhead+clone, pg.forward@max-autotune-no-cudagraphs)",
    "stock": "shipping sample_actions timestep: torch.tensor(<float>, device) rebuilt every step",
    "hoist": "the N timestep tensors built once and reused (corun.timestep_tensors); nothing else",
    "shipping_full": "the whole frame through policy.predict_action_chunk (E7's `full` loop)",
    "stages": "the whole frame as corun.prefix_stage + corun.expert_stage (same compiled callables)",
    "owngraph_expert": "denoise_step compiled WITHOUT cudagraph trees (same Inductor kernels) and the "
                       "N-step expert loop captured as ONE torch.cuda.CUDAGraph with its OWN pool",
    "kv_copy": "each frame's KV / pad / noise copied into the expert graph's static input buffers "
               "ON THE EXPERT STREAM (counted in the expert stage time)",
    "prefix_first": "per frame the host issues prefix(k) before expert(k-1)",
    "expert_first": "per frame the host issues expert(k-1) before prefix(k)",
    "green_<p>_<e>": "prefix in a green-context partition of <p> SMs, expert in a disjoint one of <e>; "
                     "each stage's graph CAPTURED inside its own green context (M0: a graph captured "
                     "on the primary context is NOT confined when replayed on a green stream)",
    "hog_<rate>": "in-process bandwidth hog (device copy graph) on the expert partition's stream",
}


def gate_full() -> dict:
    """protocol §1 gate plus the fields NN4 requires non-null: EMC rate, temperatures."""
    g = P.gate()
    try:
        emc = subprocess.run(["sudo", "-n", "/bin/sh", "-c",
                              "cat /sys/kernel/debug/bpmp/debug/clk/emc/rate"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        g["emc_rate_hz"] = int(emc)
    except (ValueError, subprocess.SubprocessError):
        g["emc_rate_hz"] = "unreadable"
    g["temps_mC"] = temps()
    g["t_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return g


def temps() -> dict:
    out = {}
    for z in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            out[(z / "type").read_text().strip()] = int((z / "temp").read_text().strip())
        except Exception:        # noqa: BLE001 -- some zones return EAGAIN, surfacing as
            continue             # TypeError inside codecs (it crashed the first M1 start)
    return out


def throttle_record(g0: dict) -> dict:
    """Measured, not asserted: GPU clock at the end vs the pinned clock, and peak temperature."""
    g1 = P.gate()
    t1 = temps()
    hot = max(t1.values()) if t1 else 0
    throttled = (g1["gpu_cur_freq_hz"] != g0["gpu_cur_freq_hz"]) or hot >= 95000
    return {"throttled": bool(throttled),
            "throttled_evidence": (f"GPU clock start {g0['gpu_cur_freq_hz']} end "
                                   f"{g1['gpu_cur_freq_hz']} (pinned min==max: "
                                   f"{g0['gpu_min_freq_hz'] == g0['gpu_max_freq_hz']}); peak "
                                   f"thermal zone at end {hot / 1000:.1f} C (trip threshold used "
                                   f"95 C); temps at end {t1}")}


def load_rung(steps: int = 10, rung: str = "bf16_both", compile_expert: str = "rung"):
    """π0 with the bf16_both rung applied EXACTLY as E7 recorded it.

    compile_expert="rung": the rung as-is (denoise_step@reduce-overhead, cudagraph trees).
    compile_expert="nographs": identical casts and prefix compilation, but denoise_step compiled
    with mode="default" -- the same Inductor kernels with no cudagraph trees -- so the caller can
    capture the expert itself (owngraph_expert). The difference is a labelled harness change.
    """
    import model_registry as MR                                        # noqa: PLC0415
    import opt_ladder as OL                                            # noqa: PLC0415
    spec = MR.get("pi0")
    policy, cfg, _ = MR.load_policy(spec, n=steps, compile_mode=None)
    P.assert_real_weights(policy, spec=spec)
    compiled, prep, prec, comp_s, fails, _ = OL.apply_rung(policy, spec, rung)
    if fails:
        raise SystemExit(f"rung {rung} failed: {fails}")
    e7 = json.loads(E7_META.read_text())
    want = e7["rung_compiled"]
    got = ";".join(compiled)
    if got != want:
        raise SystemExit(f"rung does not match E7's recorded rung:\n  E7  {want}\n  now {got}")
    note = {"rung": rung, "prep": prep, "compiled": got, "matmul_precision": prec,
            "matches_E7": True}
    if compile_expert == "nographs":
        m = policy.model
        orig = getattr(m.denoise_step, "_torchdynamo_orig_callable", None)
        if orig is None:
            orig = getattr(m.denoise_step, "__wrapped__", None)
        base = orig if orig is not None else type(m).denoise_step.__get__(m)
        m.denoise_step = torch.compile(base, mode="default")
        note["expert_recompiled"] = ("model.denoise_step re-compiled mode=default (no cudagraph "
                                     "trees) for owngraph_expert")
    return policy, spec, cfg, note


def loop(fn, min_frames=100, min_seconds=60.0, warm=20, sync_each=True, nonfinite=None):
    """E7-style back-to-back loop. Returns per-iteration ms. A device-side non-finite count is
    accumulated by `fn` itself (it receives nothing); read `nonfinite` after the loop."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    t0 = time.perf_counter()
    while len(out) < min_frames or time.perf_counter() - t0 < min_seconds:
        a = time.perf_counter()
        fn()
        if sync_each:
            torch.cuda.synchronize()
        out.append((time.perf_counter() - a) * 1e3)
    return out


class Finite:
    """Device-side count of non-finite action elements -- no host sync inside a timed loop."""

    def __init__(self):
        self.n = torch.zeros((), dtype=torch.int64, device="cuda")
        self.checked = 0

    def add(self, t):
        self.n += (~torch.isfinite(t)).sum()
        self.checked += 1

    def read(self) -> int:
        torch.cuda.synchronize()
        return int(self.n.item())


def summ(v):
    s = sorted(v)
    n = len(s)
    return {"n": n, "median_ms": round(statistics.median(s), 4),
            "p5_ms": round(s[max(0, int(0.05 * n))], 4),
            "p95_ms": round(s[min(n - 1, int(0.95 * n))], 4),
            "mean_ms": round(statistics.fmean(s), 4)}


def bench_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                          text=True).stdout.strip()


def write_meta(path: Path, meta: dict, required=("bench_commit", "model_commit",
               "checkpoint_commit", "rung", "variant", "power_mode", "sm_count",
               "granted_partitions", "gpu_clock_hz", "emc_rate_hz", "throttled")):
    """NN4: refuse to write a meta.json with a null in a required field."""
    missing = [k for k in required if meta.get(k) in (None, "")]
    if missing:
        raise SystemExit(f"meta.json would have null/empty {missing} -- NN4 forbids it")
    meta.setdefault("variant_tokens", VARIANT_TOKENS)
    meta.setdefault("lerobot_commit", LEROBOT_COMMIT)
    files = sorted(p for p in path.parent.rglob("*") if p.is_file() and p.name != "meta.json")
    import hashlib                                                     # noqa: PLC0415
    meta["raw_files"] = [{"path": str(p.relative_to(path.parent)), "bytes": p.stat().st_size,
                          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
