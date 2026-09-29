"""E12d arm (c) -- the eager co-run as TWO PROCESSES with slim weights, KV handed over in shared memory.

T014 tried two processes and was OOM-killed: each process loaded the whole π0. Here each process keeps
only the weights its stage reads (read from modeling_pi0.py, lerobot f0d2b37):
  prefix process  embed_prefix + paligemma_with_expert.forward(inputs_embeds=[prefix, None]) touch only
                  paligemma (vision tower, projector, language model) -> gemma_expert is dropped;
  expert process  denoise_step touches only gemma_expert, state_proj, action_in_proj,
                  action_time_mlp_*, action_out_proj (forward's inputs_embeds=[None, suffix] branch)
                  -> paligemma is dropped.
Each process has its own Python interpreter, so the two stages are ISSUED by two CPU cores in parallel
-- the host-issue coupling T014 blamed is removed by construction, which is the point of this arm.

CUDA IPC does not work on this Orin (torch.multiprocessing CUDA tensor share -> cudaErrorInvalidValue,
scratch/ipc_probe.py, 2026-09-28), so each frame's KV cache + pad mask crosses the process boundary
through POSIX shared memory: a double buffer (slot k % 2), producer/consumer semaphores per slot
(prefix(k) waits until the expert has copied slot k % 2 out, i.e. frame k-2). The device->host copy is
counted in the prefix stage, host->device in the expert stage; both are also timed separately. The
shared region is registered with cudaHostRegister when the driver allows it (recorded).

Conditions (orchestrator, one pair of long-lived processes per run; `--mps` starts the MPS daemon
first so the two processes share the GPU through one MPS server instead of time-sliced contexts):
  prefix_alone   the prefix process alone, F frames (the expert process idle, no GPU work)
  expert_alone   the expert process alone, F frames, reading the two slots filled once
  serial         lockstep across the processes: prefix(k+1) starts after expert(k) ends
  pipeline       free-running double buffer; T_pipe from the expert's completion times
  pipeline_60s   the same, >= 60 s
Bracketed pre / post. Equivalence: the expert's actions for 6 frames vs a single-process serial run of
the full model (corun.run_serial), bitwise expected; stale control = the expert reuses the other
buffer's cache (frame k-1) without copying and MUST change the actions. Non-finite count on every
action. Stage start is taken after the semaphore is acquired, so waiting for the other process is not
counted as work (E12 §3).

    python bench/e12d_twoproc.py --outdir results/<date>_E12d_pi0_eager_twoproc_N10_MAXN [--mps]
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))

N = 10
FIXTURES = REPO / "bench/fixtures/e12"
ALIGN = 256


# ------------------------------------------------------------------------------------------- helpers
def meminfo_kb(key="MemAvailable"):
    for ln in Path("/proc/meminfo").read_text().splitlines():
        if ln.startswith(key + ":"):
            return int(ln.split()[1])
    return -1


def rss_kb():
    for ln in Path("/proc/self/status").read_text().splitlines():
        if ln.startswith("VmRSS:"):
            return int(ln.split()[1])
    return -1


def kv_pairs(cache):
    if hasattr(cache, "layers"):
        return [(l.keys, l.values) for l in cache.layers]
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return [(k, v) for k, v in cache]


def make_cache(pairs):
    from transformers import DynamicCache                               # noqa: PLC0415
    c = DynamicCache()
    for i, (k, v) in enumerate(pairs):
        c.update(k, v, i)
    return c


def load_role(role: str):
    """The full policy loaded once (as T014 did), then the other stage's weights DROPPED."""
    import torch                                                        # noqa: PLC0415
    import model_registry as MR                                         # noqa: PLC0415
    import corun as CR                                                  # noqa: PLC0415
    import pi0_common as P                                              # noqa: PLC0415
    spec = MR.get("pi0")
    policy, cfg, _ = MR.load_policy(spec, n=N, compile_mode=None)
    if role != "expert":
        P.assert_real_weights(policy, spec=spec)          # reads paligemma tensors; before slimming
    note = CR.patch_pi0_nosync(policy)
    pwe = policy.model.paligemma_with_expert
    dropped = "nothing (reference)"
    if role == "prefix":
        n = sum(p.numel() for p in pwe.gemma_expert.parameters())
        pwe.gemma_expert = torch.nn.Module()
        dropped = f"paligemma_with_expert.gemma_expert ({n} params)"
    elif role == "expert":
        n = sum(p.numel() for p in pwe.paligemma.parameters())
        pwe.paligemma = torch.nn.Module()
        dropped = f"paligemma_with_expert.paligemma ({n} params)"
    gc.collect()
    torch.cuda.empty_cache()
    kept = sum(p.numel() for p in policy.parameters())
    frames, manifest = CR.load_frames(policy, spec, FIXTURES, 5, seed0=0)
    return policy, spec, frames, manifest, {"nosync": note, "dropped": dropped, "params_kept": kept}


def footprint():
    import torch                                                        # noqa: PLC0415
    torch.cuda.synchronize()
    return {"cuda_allocated_GiB": round(torch.cuda.memory_allocated() / 2**30, 3),
            "cuda_reserved_GiB": round(torch.cuda.memory_reserved() / 2**30, 3),
            "cuda_max_allocated_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "rss_GiB": round(rss_kb() / 2**20, 3)}


class Slots:
    """Two KV slots in POSIX shared memory, as typed CPU tensor views."""

    def __init__(self, shm, layout):
        import torch                                                    # noqa: PLC0415
        self.shm, self.layout = shm, layout
        base = torch.frombuffer(shm.buf, dtype=torch.uint8)
        self.base = base
        self.pinned = False
        rc = torch.cuda.cudart().cudaHostRegister(base.data_ptr(), base.numel(), 0)
        self.pin_rc = int(rc) if not hasattr(rc, "value") else int(rc.value)
        self.pinned = self.pin_rc == 0
        self.views = []
        for s in range(2):
            vs = []
            for ent in layout["tensors"]:
                off = s * layout["slot_bytes"] + ent["offset"]
                dt = getattr(torch, ent["dtype"])
                vs.append(base[off:off + ent["nbytes"]].view(dt).view(ent["shape"]))
            self.views.append(vs)

    def close(self):
        import torch                                                    # noqa: PLC0415
        if self.pinned:
            torch.cuda.cudart().cudaHostUnregister(self.base.data_ptr())
        self.views = None
        self.base = None


def layout_of(pairs, pad):
    ents, off = [], 0
    for k, v in pairs:
        for t in (k, v):
            n = t.numel() * t.element_size()
            ents.append({"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
                         "nbytes": n, "offset": off})
            off += (n + ALIGN - 1) // ALIGN * ALIGN
    n = pad.numel() * pad.element_size()
    ents.append({"shape": list(pad.shape), "dtype": str(pad.dtype).replace("torch.", ""),
                 "nbytes": n, "offset": off})
    off += (n + ALIGN - 1) // ALIGN * ALIGN
    return {"tensors": ents, "slot_bytes": off, "n_layers": len(pairs)}


# ------------------------------------------------------------------------------------------- roles
def prefix_role(conn, sems):
    import torch                                                        # noqa: PLC0415
    from multiprocessing import shared_memory                           # noqa: PLC0415
    import corun as CR                                                  # noqa: PLC0415
    empty, full, done = sems
    policy, spec, frames, manifest, info = load_role("prefix")
    with torch.no_grad():
        for fr in frames[:2]:
            pkv, pad = CR.prefix_stage(fr["ctx"])
        torch.cuda.synchronize()
    lay = layout_of(kv_pairs(pkv), pad)
    shm = shared_memory.SharedMemory(create=True, size=2 * lay["slot_bytes"])
    slots = Slots(shm, lay)
    conn.send({"ready": "prefix", "layout": lay, "shm": shm.name, "footprint": footprint(),
               "info": info, "pinned": slots.pinned, "pin_rc": slots.pin_rc})

    def one(k, b):
        fr = frames[k % len(frames)]
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        t0 = time.monotonic()
        e0.record()
        pkv, pad = CR.prefix_stage(fr["ctx"])
        e1.record()
        srcs = [t for kv in kv_pairs(pkv) for t in kv] + [pad]
        for d, s in zip(slots.views[b], srcs):
            d.copy_(s, non_blocking=slots.pinned)
        e2.record()
        torch.cuda.synchronize()
        t1 = time.monotonic()
        return {"frame": k, "start": t0, "end": t1, "stage_ms": (t1 - t0) * 1e3,
                "compute_ms": e0.elapsed_time(e1), "copy_ms": e1.elapsed_time(e2)}

    with torch.no_grad():
        while True:
            cmd = conn.recv()
            c = cmd["cmd"]
            if c == "quit":
                break
            rows = []
            if c == "alone":
                for k in range(cmd["F"]):
                    rows.append(one(k, k % 2))
            elif c == "fill":
                for k in range(2):
                    rows.append(one(k, k))
            elif c == "pipe":
                lock = cmd.get("lockstep", False)
                for k in range(cmd["F"]):
                    if lock and k > 0:
                        done.acquire()
                    empty[k % 2].acquire()
                    rows.append(one(k, k % 2))
                    full[k % 2].release()
                if lock:
                    done.acquire()
            conn.send({"rows": rows, "footprint": footprint()})
    slots.close()
    del slots
    gc.collect()
    try:
        shm.close()
    except BufferError:
        pass
    shm.unlink()


def expert_role(conn, sems, layout, shm_name):
    import numpy as np                                                  # noqa: PLC0415
    import torch                                                        # noqa: PLC0415
    from multiprocessing import shared_memory                           # noqa: PLC0415
    import corun as CR                                                  # noqa: PLC0415
    empty, full, done = sems
    policy, spec, frames, manifest, info = load_role("expert")
    shm = shared_memory.SharedMemory(name=shm_name)
    slots = Slots(shm, layout)
    gbuf = [[torch.empty(e["shape"], dtype=getattr(torch, e["dtype"]), device="cuda")
             for e in layout["tensors"]] for _ in range(2)]
    nonfinite = torch.zeros((), dtype=torch.int64, device="cuda")
    conn.send({"ready": "expert", "footprint": footprint(), "info": info,
               "pinned": slots.pinned, "pin_rc": slots.pin_rc})

    def one(k, b, keep=None, use=None):
        """Copy slot b to device buffer b; compute from buffer `use` (default b). The stale control
        passes use = the other buffer, which holds frame k-1's cache (copied on the previous frame)."""
        fr = frames[k % len(frames)]
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        t0 = time.monotonic()
        e0.record()
        for d, s in zip(gbuf[b], slots.views[b]):
            d.copy_(s, non_blocking=slots.pinned)
        e1.record()
        e1.synchronize()                        # slot b is free once it is on the device
        t_copied = time.monotonic()
        ts = gbuf[b if use is None else use]
        pairs = [(ts[2 * i], ts[2 * i + 1]) for i in range(layout["n_layers"])]
        pad = ts[-1]
        act = CR.expert_stage(fr["ctx"], make_cache(pairs), pad, fr["noise"], N)
        nonfinite.add_((~torch.isfinite(act)).sum())
        e2.record()
        torch.cuda.synchronize()
        t1 = time.monotonic()
        if keep is not None and k < keep:
            keep_out[k] = act.float().cpu().numpy()
        return {"frame": k, "start": t0, "end": t1, "copied": t_copied, "stage_ms": (t1 - t0) * 1e3,
                "copy_ms": e0.elapsed_time(e1), "compute_ms": e1.elapsed_time(e2)}

    with torch.no_grad():
        while True:
            cmd = conn.recv()
            c = cmd["cmd"]
            if c == "quit":
                break
            rows, keep_out = [], {}
            if c == "alone":
                for k in range(cmd["F"]):
                    rows.append(one(k, k % 2))
            elif c == "pipe":
                lock, stale, keep = cmd.get("lockstep", False), cmd.get("stale", False), cmd.get("keep")
                for k in range(cmd["F"]):
                    b = k % 2
                    full[b].acquire()
                    if stale and k > 0:
                        # NEGATIVE CONTROL: frame k's cache is copied as usual, but the expert computes
                        # from the other device buffer = frame k-1's cache -> the actions MUST change
                        r = one(k, b, keep=keep, use=(b + 1) % 2)
                    else:
                        r = one(k, b, keep=keep)
                    empty[b].release()
                    if lock:
                        done.release()
                    rows.append(r)
            torch.cuda.synchronize()
            conn.send({"rows": rows, "footprint": footprint(), "nonfinite": int(nonfinite.item()),
                       "actions": {k: v for k, v in keep_out.items()}})
    slots.close()
    del slots
    gc.collect()
    try:
        shm.close()
    except BufferError:
        pass
    _ = np


def ref_role(conn):
    """Single process, full model: the serial reference actions for the equivalence check."""
    import torch                                                        # noqa: PLC0415
    import corun as CR                                                  # noqa: PLC0415
    policy, spec, frames, manifest, info = load_role("reference")
    with torch.no_grad():
        fr6 = [frames[k % len(frames)] for k in range(6)]
        CR.run_serial(fr6[:2], N)
        acts, samples = CR.run_serial(fr6, N, keep_actions=True)
    conn.send({"actions": {k: a.float().cpu().numpy() for k, a in enumerate(acts)},
               "samples": samples, "footprint": footprint(), "info": info,
               "manifest": manifest})


# ------------------------------------------------------------------------------------------- orchestrator
def med(v):
    return statistics.median(v) if v else float("nan")


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))] if s else float("nan")


def main() -> int:
    import multiprocessing as mp                                        # noqa: PLC0415
    import numpy as np                                                  # noqa: PLC0415
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--mps", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = REPO / args.outdir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty (protocol §5)")
    (out / "raw").mkdir(parents=True, exist_ok=True)
    import pi0_common as P                                              # noqa: PLC0415
    g0 = P.gate()
    mem0 = meminfo_kb()
    ctx = mp.get_context("spawn")
    log = []

    def say(s):
        print(s, flush=True)
        log.append(s)

    # 1. the single-process reference, in its own process so its memory is returned
    a, b = ctx.Pipe()
    pr = ctx.Process(target=ref_role, args=(b,))
    pr.start()
    ref = a.recv()
    pr.join(120)
    say(f"  reference serial (full model, one process): frame {med([s['frame_ms'] for s in ref['samples'][2:]]):.1f} ms, "
        f"footprint {ref['footprint']}")

    mps_log = ""
    if args.mps:
        mdir = Path("/tmp/e12d_mps")
        (mdir / "pipe").mkdir(parents=True, exist_ok=True)
        (mdir / "log").mkdir(parents=True, exist_ok=True)
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = str(mdir / "pipe")
        os.environ["CUDA_MPS_LOG_DIRECTORY"] = str(mdir / "log")
        r = subprocess.run(["nvidia-cuda-mps-control", "-d"], capture_output=True, text=True)
        say(f"  MPS daemon start rc={r.returncode} {r.stdout.strip()} {r.stderr.strip()}")
        if r.returncode != 0:
            raise SystemExit("MPS daemon did not start")

    sems = ([ctx.Semaphore(1), ctx.Semaphore(1)], [ctx.Semaphore(0), ctx.Semaphore(0)], ctx.Semaphore(0))
    pa, pb = ctx.Pipe()
    pp = ctx.Process(target=prefix_role, args=(pb, sems))
    pp.start()
    pre_ready = pa.recv()
    say(f"  prefix process ready: {pre_ready['footprint']} dropped {pre_ready['info']['dropped']} "
        f"slot {pre_ready['layout']['slot_bytes']} B pinned={pre_ready['pinned']} (rc {pre_ready['pin_rc']})")
    ea, eb = ctx.Pipe()
    pe = ctx.Process(target=expert_role, args=(eb, sems, pre_ready["layout"], pre_ready["shm"]))
    pe.start()
    exp_ready = ea.recv()
    mem_both = meminfo_kb()
    say(f"  expert process ready: {exp_ready['footprint']} dropped {exp_ready['info']['dropped']} "
        f"pinned={exp_ready['pinned']}; MemAvailable {mem0 / 2**20:.2f} -> {mem_both / 2**20:.2f} GiB")
    if args.mps:
        try:
            mps_log = (Path(os.environ["CUDA_MPS_LOG_DIRECTORY"]) / "control.log").read_text()
        except OSError:
            mps_log = "control.log unreadable"
        say(f"  MPS control.log: {mps_log.count('NEW CLIENT')} NEW CLIENT line(s)")

    def cmd_both(cp, ce):
        if cp is not None:
            pa.send(cp)
        if ce is not None:
            ea.send(ce)
        rp = pa.recv() if cp is not None else None
        re_ = ea.recv() if ce is not None else None
        return rp, re_

    F = 12 if args.smoke else args.frames
    blocks = {}
    samples = []

    def record(key, rp, re_):
        blocks[key] = {"prefix": rp["rows"] if rp else [], "expert": re_["rows"] if re_ else [],
                       "nonfinite": re_.get("nonfinite", 0) if re_ else 0}
        for stage, rows in (("prefix", blocks[key]["prefix"]), ("expert", blocks[key]["expert"])):
            for r in rows:
                samples.append({"block": key, "stage": stage, **{k: (round(v, 6) if isinstance(v, float) else v)
                                                                  for k, v in r.items()}})

    def iso(tag):
        rp, _ = cmd_both({"cmd": "alone", "F": 10}, None)
        rp, _ = cmd_both({"cmd": "alone", "F": F}, None)
        record(f"prefix_alone{tag}", rp, None)
        cmd_both({"cmd": "fill"}, None)
        cmd_both(None, {"cmd": "alone", "F": 10})
        _, re_ = cmd_both(None, {"cmd": "alone", "F": F})
        record(f"expert_alone{tag}", None, re_)
        rp, re_ = cmd_both({"cmd": "pipe", "F": F, "lockstep": True}, {"cmd": "pipe", "F": F, "lockstep": True})
        record(f"serial{tag}", rp, re_)
        pm = med([r["stage_ms"] for r in blocks[f"prefix_alone{tag}"]["prefix"][5:]])
        em = med([r["stage_ms"] for r in blocks[f"expert_alone{tag}"]["expert"][5:]])
        sp_, se_ = blocks[f"serial{tag}"]["prefix"], blocks[f"serial{tag}"]["expert"]
        fm = med([e["end"] * 1e3 - p["start"] * 1e3 for p, e in zip(sp_, se_)][5:])
        say(f"  {tag:5s} A_iso {pm:8.2f}  B_iso {em:8.2f}  T_serial {fm:8.2f} ms")
        return pm, em, fm

    # 2. equivalence
    rp, re_ = cmd_both({"cmd": "pipe", "F": 6}, {"cmd": "pipe", "F": 6, "keep": 6})
    good = re_["actions"]
    rp, re_ = cmd_both({"cmd": "pipe", "F": 6}, {"cmd": "pipe", "F": 6, "keep": 6, "stale": True})
    bad = re_["actions"]
    eq_rows = []
    for k in range(6):
        r = ref["actions"][k]
        for chk, acts in (("pipeline_vs_single_process_serial", good), ("stale_control", bad)):
            y = acts[k]
            d = np.abs(y - r)
            bit = bool(np.array_equal(y, r))
            nf = int((~np.isfinite(y)).sum())
            if chk == "stale_control":
                verdict = "n/a (frame 0)" if k == 0 else ("DETECTED" if not bit else "MISSED")
            else:
                verdict = "PASS" if (bit and nf == 0) else "FAIL"
            eq_rows.append({"check": chk, "frame": k, "max_abs_diff": float(d.max()),
                            "bitwise_identical": bit, "n_nonfinite": nf, "verdict": verdict})
    ok = (all(r["verdict"] == "PASS" for r in eq_rows if r["check"].startswith("pipeline"))
          and all(r["verdict"] in ("DETECTED", "n/a (frame 0)") for r in eq_rows if r["check"] == "stale_control"))
    for r in eq_rows:
        say(f"    {r['check']:36s} frame {r['frame']} max|d| {r['max_abs_diff']:.3e} -> {r['verdict']}")
    say(f"  equivalence {'PASS' if ok else 'FAIL'}")

    # 3. timings, bracketed
    A1, B1, T1 = iso("_pre")
    rp, re_ = cmd_both({"cmd": "pipe", "F": F}, {"cmd": "pipe", "F": F})
    record("pipeline", rp, re_)
    ends = [r["end"] for r in blocks["pipeline"]["expert"]]
    Tp_est = (ends[-1] - ends[5]) * 1e3 / max(1, len(ends) - 6)
    if not args.smoke:
        F60 = max(F, int(args.seconds * 1000 / max(Tp_est, 1)) + 6)
        rp, re_ = cmd_both({"cmd": "pipe", "F": F60}, {"cmd": "pipe", "F": F60})
        record("pipeline_60s", rp, re_)
    A2, B2, T2 = iso("_post")
    pa.send({"cmd": "quit"})
    ea.send({"cmd": "quit"})
    pp.join(120)
    pe.join(120)
    mps_stop = ""
    if args.mps:
        r = subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", capture_output=True, text=True)
        time.sleep(2)
        left = subprocess.run(["pgrep", "-x", "nvidia-cuda-mps-server"], capture_output=True, text=True).stdout.strip()
        mps_stop = f"quit rc={r.returncode}; mps-server processes left: '{left or 'none'}'"
        say(f"  MPS stopped: {mps_stop}")
        try:
            mps_log = (Path(os.environ["CUDA_MPS_LOG_DIRECTORY"]) / "control.log").read_text()
        except OSError:
            pass

    A, B, Tser = (A1 + A2) / 2, (B1 + B2) / 2, (T1 + T2) / 2
    nonfinite = sum(v["nonfinite"] for v in blocks.values())
    ratios = []
    for key in [k for k in ("pipeline", "pipeline_60s") if k in blocks]:
        pre = [r["stage_ms"] for r in blocks[key]["prefix"][5:]]
        exp = [r["stage_ms"] for r in blocks[key]["expert"][5:]]
        ends = [r["end"] for r in blocks[key]["expert"]]
        Tp = (ends[-1] - ends[5]) * 1e3 / (len(ends) - 6)
        mx = max(A, B)
        ratios.append({
            "experiment": "E12d", "arm": "twoproc_mps" if args.mps else "twoproc",
            "variant": "eager+pi0_nosync+hoist+slim_weights+shm_dbuf_kv" + ("+mps" if args.mps else ""),
            "power_mode": g0["power_mode"], "sm_count": g0["sm_count"], "N": N,
            "A_iso_ms": round(A, 4), "B_iso_ms": round(B, 4), "T_serial_ms": round(Tser, 4),
            "A_co_ms": round(med(pre), 4), "A_co_p5_ms": round(pct(pre, .05), 4), "A_co_p95_ms": round(pct(pre, .95), 4),
            "B_co_ms": round(med(exp), 4), "B_co_p5_ms": round(pct(exp, .05), 4), "B_co_p95_ms": round(pct(exp, .95), 4),
            "T_pipe_ms": round(Tp, 4), "s": round(med(exp) / B, 4), "p": round(med(pre) / A, 4),
            "r": round(A / B, 4), "O": round(Tp / mx, 4), "phi": round((Tser - Tp) / (Tser - mx), 4),
            "speedup_vs_serial": round(Tser / Tp, 4),
            "prefix_copy_d2h_ms": round(med([r["copy_ms"] for r in blocks[key]["prefix"][5:]]), 4),
            "expert_copy_h2d_ms": round(med([r["copy_ms"] for r in blocks[key]["expert"][5:]]), 4),
            "n": len(exp), "reference": "same_session_same_harness (two processes)",
            "valid": bool(ok and nonfinite == 0), "block": key})
    for r in ratios:
        say(f"  {r['block']:13s} O={r['O']} phi={r['phi']} p={r['p']} s={r['s']} r={r['r']} T_pipe={r['T_pipe_ms']} "
            f"(A {r['A_iso_ms']}, B {r['B_iso_ms']}, Tser {r['T_serial_ms']}; copies d2h {r['prefix_copy_d2h_ms']} "
            f"h2d {r['expert_copy_h2d_ms']})")
    for name, rows in (("corun_ratios.csv", ratios), ("equivalence.csv", eq_rows)):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    with (out / "raw" / "latency_samples.csv").open("w", newline="") as f:
        keys = []
        for s in samples:
            for k in s:
                if k not in keys:
                    keys.append(k)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(samples)
    (out / "raw" / "orchestrator_log.txt").write_text("\n".join(log) + "\n")
    if args.mps:
        (out / "raw" / "mps_control.log").write_text(mps_log or "empty")
    (out / "equivalence.md").write_text("\n".join([
        "# E12d arm (c) — equivalence, two processes", "",
        "The expert process's actions for 6 frames against a single-process serial run of the full model "
        "(corun.run_serial), same frames and seeds.", "",
        f"* pipeline vs single-process serial: **{'bitwise identical on every frame' if all(r['verdict']=='PASS' for r in eq_rows if r['check'].startswith('pipeline')) else 'NOT identical — INVALID'}**",
        f"* stale control (expert reuses frame k-1's cache): **{sum(r['verdict']=='DETECTED' for r in eq_rows)} of 5 detected**",
        f"* non-finite action elements over every condition: {nonfinite}", "", "Rows: `equivalence.csv`."]) + "\n")
    import e12b_common as C                                             # noqa: PLC0415
    g1 = P.gate()
    C.write_meta(out / "meta.json", {
        "experiment": "E12d arm (c) -- eager co-run as two processes with slim weights, KV via shared memory"
                      + (" under MPS" if args.mps else " (no MPS)"),
        "task": "owner request 2026-09-28 (T014 follow-up)",
        "bench_commit": C.bench_commit(), "model_commit": C.CKPT_COMMIT, "checkpoint": "lerobot/pi0_libero_finetuned_v044",
        "checkpoint_commit": C.CKPT_COMMIT, "rung": "eager (no torch.compile)", "compile_mode": "none (eager)",
        "variant": ratios[0]["variant"] if ratios else "none", "power_mode": g0["power_mode"],
        "sm_count": g0["sm_count"], "granted_partitions": "none (two processes, whole GPU each)",
        "gpu_clock_hz": g0["gpu_cur_freq_hz"],
        "emc_rate_hz": C.gate_full().get("emc_rate_hz", "unreadable"),
        "throttled": g1["gpu_cur_freq_hz"] != g0["gpu_cur_freq_hz"],
        "throttled_evidence": f"GPU clock start {g0['gpu_cur_freq_hz']} end {g1['gpu_cur_freq_hz']}",
        "gate_start": g0, "gate_end": g1, "N": N, "batch": 1, "frames_per_condition": F, "smoke": bool(args.smoke),
        "mps": bool(args.mps), "mps_stop": mps_stop or "not used",
        "equivalence_pass": ok, "nonfinite_total": nonfinite,
        "footprints": {"reference_single_process_full_model": ref["footprint"],
                       "prefix_process_slim": pre_ready["footprint"], "expert_process_slim": exp_ready["footprint"],
                       "MemAvailable_GiB_before": round(mem0 / 2**20, 3),
                       "MemAvailable_GiB_both_processes_ready": round(mem_both / 2**20, 3)},
        "slimming": {"prefix": pre_ready["info"]["dropped"], "expert": exp_ready["info"]["dropped"],
                     "params_kept_prefix": pre_ready["info"]["params_kept"],
                     "params_kept_expert": exp_ready["info"]["params_kept"]},
        "kv_handoff": {"layout_slot_bytes": pre_ready["layout"]["slot_bytes"],
                       "n_tensors": len(pre_ready["layout"]["tensors"]),
                       "host_registered_pinned": {"prefix": pre_ready["pinned"], "expert": exp_ready["pinned"],
                                                  "cudaHostRegister_rc": pre_ready["pin_rc"]},
                       "why_not_cuda_ipc": "torch.multiprocessing CUDA tensor share fails on this Orin with "
                                           "cudaErrorInvalidValue (scratch/ipc_probe.py, 2026-09-28)"},
        "bracket": {"A_pre": A1, "A_post": A2, "B_pre": B1, "B_post": B2, "T_serial_pre": T1, "T_serial_post": T2},
        "reference_serial_frame_ms": med([s["frame_ms"] for s in ref["samples"][2:]]),
        "decisions": {"stage_time": "wall (CLOCK_MONOTONIC) from after the semaphore is acquired to after the "
                                    "stage's stream synchronises; D2H copy in the prefix stage, H2D in the expert",
                      "T_pipe": "(expert end of last frame - expert end of frame 5) / (frames - 6)",
                      "T_serial": "lockstep across the processes: median of expert end(k) - prefix start(k)",
                      "first_5_frames_dropped": "per-stage medians skip frames 0-4 of each block"},
        "command": " ".join([sys.executable, *sys.argv]),
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
