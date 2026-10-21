"""T016 step 5 (EV-13): which of E12b's two suspects produced T014's compiled co-run corruption?

2 x 2, one process per cell (a torch.compile choice cannot be undone in-process):
  --pool shared   T014-faithful: the expert calls the rung's own model.denoise_step@reduce-overhead --
                  a CUDAGraphTrees callable sharing the tree pool with the prefix's
                  pg.embed_image@reduce-overhead -- eagerly on the expert stream (no own graph).
  --pool own      E12b: denoise_step recompiled mode="default" and the N-step loop captured as one
                  torch.cuda.CUDAGraph in its OWN pool (owngraph_expert).
  --kv single     one KV buffer; prefix(k) copies its cache into it with NO wait on expert(k-1): the
                  next prefix may overwrite the cache the expert is still reading.
  --kv double     E12b dbuf_kv: buffer k % 2, prefix(k) waits for expert(k-2).
Per cell, >= 100 frames, each a DISTINCT observation (the 5 E12 fixtures, each image perturbed by a
seeded +-1/255 pattern per frame) with distinct seeded noise; the pipeline's actions are compared with
the SAME cell's serial run (bitwise), every action is finite-checked on the device, and a stale control
(the expert handed another frame's cache) must change the actions.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
import corun as CR               # noqa: E402
import e12b_common as C          # noqa: E402
import e12b_corun as H           # noqa: E402

N = 10


class Diag:
    def __init__(self, policy, frames, pool, kv):
        self.m, self.frames, self.pool, self.kvmode = policy.model, frames, pool, kv
        self.sp, self.se = torch.cuda.Stream(), torch.cuda.Stream()
        fr0 = frames[0]
        with torch.no_grad():
            pkv0, pad0 = CR.prefix_stage(fr0["ctx"])
            torch.cuda.synchronize()
        nb = 3                                            # buffers: 0/1 working, 2 = stale holder
        self.kv = [copy.deepcopy(pkv0) for _ in range(nb)]
        self.pad = [pad0.clone() for _ in range(nb)]
        self.noise = fr0["noise"].clone()
        self.state = fr0["ctx"]["state"].clone()
        self.ts = CR.timestep_tensors(N, pad0.shape[0], pad0.device)
        self.g, self.out = {}, {}
        if pool == "own":
            pe = torch.cuda.graph_pool_handle()
            with torch.no_grad(), torch.cuda.stream(self.se):
                for b in range(nb):
                    for _ in range(3):
                        self._body(b)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=pe, stream=torch.cuda.current_stream()):
                        self.out[b] = self._body(b)
                    self.g[b] = g
                torch.cuda.synchronize()

    def _body(self, b):
        x = self.noise
        for step in range(N):
            v = self.m.denoise_step(self.state, self.pad[b], self.kv[b], x, self.ts[step])
            x = x + (-1.0 / N) * v
        return x

    def expert(self, b):
        if self.pool == "own":
            self.g[b].replay()
            return self.out[b]
        return self._body(b)                          # rung graph-tree denoise_step, shared pool

    def run(self, F, fin, mode):
        """mode: serial | pipeline | stale. Returns list of per-frame action tensors (clones)."""
        acts = [None] * F
        ev_p = [torch.cuda.Event() for _ in range(F)]
        ev_e = [torch.cuda.Event() for _ in range(F)]
        with torch.no_grad():
            for k in range(F + 1):
                if k < F:                               # prefix(k)
                    with torch.cuda.stream(self.sp):
                        s = torch.cuda.current_stream()
                        b = 0 if self.kvmode == "single" else k % 2
                        if self.kvmode == "double" and k >= 2 and mode != "serial":
                            s.wait_event(ev_e[k - 2])
                        if mode == "serial" and k >= 1:
                            s.wait_event(ev_e[k - 1])
                        pkv, pad = CR.prefix_stage(self.frames[k]["ctx"])
                        H.copy_tree(self.kv[b], pkv)
                        self.pad[b].copy_(pad, non_blocking=True)
                        if mode == "stale" and k % 3 == 0:
                            H.copy_tree(self.kv[2], pkv)
                            self.pad[2].copy_(pad, non_blocking=True)
                        ev_p[k].record(s)
                    if mode == "serial":
                        ev_p[k].synchronize()
                if k >= 1:                              # expert(k-1)
                    j = k - 1
                    with torch.cuda.stream(self.se):
                        s = torch.cuda.current_stream()
                        s.wait_event(ev_p[j])
                        self.noise.copy_(self.frames[j]["noise"], non_blocking=True)
                        self.state.copy_(self.frames[j]["ctx"]["state"], non_blocking=True)
                        b = 2 if mode == "stale" else (0 if self.kvmode == "single" else j % 2)
                        out = self.expert(b)
                        fin.add(out)
                        acts[j] = out.clone()
                        ev_e[j].record(s)
                    if mode == "serial":
                        ev_e[j].synchronize()
            torch.cuda.synchronize()
        return acts


def distinct_frames(policy, spec, F):
    base, manifest = CR.load_frames(policy, spec, REPO / "bench/fixtures/e12", F, seed0=1000)
    g = torch.Generator(device="cuda").manual_seed(4242)
    out = []
    for k, fr in enumerate(base):
        ctx = dict(fr["ctx"])
        imgs = ctx["images"]
        if isinstance(imgs, (list, tuple)):
            ctx["images"] = [t + (torch.randint(-1, 2, t.shape, generator=g, device=t.device) / 255.0
                                  * 2).to(t.dtype) for t in imgs]
        else:
            ctx["images"] = imgs + (torch.randint(-1, 2, imgs.shape, generator=g, device=imgs.device)
                                    / 255.0 * 2).to(imgs.dtype)
        out.append({**fr, "ctx": ctx})
    return out, manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=("shared", "own"), required=True)
    ap.add_argument("--kv", choices=("single", "double"), required=True)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    out = REPO / a.outdir
    out.mkdir(parents=True, exist_ok=True)
    g0 = C.gate_full()
    if "MAXN" not in g0["power_mode"]:
        raise SystemExit("MAXN only")
    policy, spec, cfg, note = C.load_rung(N)
    if a.pool == "own":
        note["expert"] = H.recompile_default(policy, spec, "denoise_step")
    frames, manifest = distinct_frames(policy, spec, a.frames)
    d = Diag(policy, frames, a.pool, a.kv)
    fin_s, fin_p, fin_x = C.Finite(), C.Finite(), C.Finite()
    serial = d.run(a.frames, fin_s, "serial")
    pipe = d.run(a.frames, fin_p, "pipeline")
    stale = d.run(a.frames, fin_x, "stale")
    rows, eq_fail, st_det, st_n = [], 0, 0, 0
    for k in range(a.frames):
        s, p, x = serial[k], pipe[k], stale[k]
        same = bool(torch.equal(s, p))
        eq_fail += (not same)
        src = 3 * (k // 3)
        stale_differs = bool(not torch.equal(s, x))
        if src != k:
            st_n += 1
            st_det += stale_differs
        rows.append({"frame": k, "pipeline_bitwise_equal_serial": same,
                     "pipeline_max_abs_diff": float((p.float() - s.float()).abs().max()),
                     "pipeline_nonfinite": int((~torch.isfinite(p)).sum()),
                     "stale_source_frame": src, "stale_differs": stale_differs})
    res = {"cell": f"{a.pool}_pool+{a.kv}_kv", "frames": a.frames,
           "nonfinite_serial": fin_s.read(), "nonfinite_pipeline": fin_p.read(), "nonfinite_stale": fin_x.read(),
           "equivalence_failures_vs_serial": eq_fail,
           "stale_control_detected": f"{st_det}/{st_n}",
           "max_abs_diff_pipeline_vs_serial": max(r["pipeline_max_abs_diff"] for r in rows)}
    with (out / "frames.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
    thr = C.throttle_record(g0)
    C.write_meta(out / "meta.json", {
        "experiment": "T016 step 5 (EV-13): E12b compiled co-run corruption, 2x2 diagnosis",
        "bench_commit": C.bench_commit(), "model_commit": C.CKPT_COMMIT, "checkpoint": spec.checkpoint,
        "checkpoint_commit": C.CKPT_COMMIT, "rung": "bf16_both", "rung_note": note,
        "variant": f"bf16_both+hoist+{'rung_graphtree_expert' if a.pool == 'shared' else 'owngraph_expert'}"
                   f"+{'single_kv' if a.kv == 'single' else 'dbuf_kv'}+prefix_first",
        "power_mode": g0["power_mode"], "sm_count": g0["sm_count"], "granted_partitions": "none (whole GPU)",
        "gpu_clock_hz": g0["gpu_cur_freq_hz"], "emc_rate_hz": g0["emc_rate_hz"], "gate_start": g0, **thr,
        "frames_manifest": manifest, "frame_distinctness": "5 fixtures x seeded +-2/255 image perturbation per "
        "frame (generator seed 4242) + per-frame noise seeds 1000+k", "result": res,
        "command": " ".join([sys.executable, *sys.argv]),
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
