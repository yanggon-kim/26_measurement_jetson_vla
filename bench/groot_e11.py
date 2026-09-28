"""T012 steps 2-6 / E11c-f for GR00T N1.7 (2026-09-28 completeness audit).

T012 was closed with "GR00T gets E11a only"; its acceptance asks for E11c (hand-over + exact transfers), E11d
(prefix vs input), E11e (memory pressure + CPU) and E11f (pipelining) "as T010". This is that port. It is called
from bench/groot_phases.py (`--e11 <part>`) AFTER that script has loaded the model, applied the rung's compile flags
and installed its region timers, so every part measures exactly the model the phase split, E7 and E7v measured.

Reused, not re-implemented:
  mempressure.measure_loads / measure_cpu_util / measure_power / emit_rows   (E11e, LeRobot path unchanged)
  pipelining.measure_concurrency / gpu_idle_from_nsys / emit_model_inputs     (E11f, LeRobot path unchanged)
  prefix_sweep.linfit, handover.timed_wall / timed_event, phase_loops.run_loop
  groot_phases.build_phases (the E7 prefix / ae / full loops), segments_groot (the E11a DiT block capture)

    . ~/venvs/groot/activate_orin_venv.sh
    RUNG_EAGER="--rung eager --tf32 on"
    RUNG_BEST="--rung vlm_compile --tf32 on --head-mode reduce-overhead --llm-mode max-autotune-no-cudagraphs"
    python bench/groot_phases.py $RUNG_EAGER --e11 handover    --outdir results/<d>_E11c_groot_n17_handover
    python bench/groot_phases.py $RUNG_X     --e11 prefixsweep --outdir results/<d>_E11d_groot_n17_<rung>_prefixsweep --warmup 5 --timed 30
    python bench/groot_phases.py $RUNG_X     --e11 mempressure --outdir results/<d>_E11e_groot_n17_<rung>_mempressure --seconds 60
    python bench/groot_phases.py $RUNG_X     --e11 pipelining  --outdir results/<d>_E11f_groot_n17_<rung>_pipelining  --seconds 60
    ~/venvs/pi0/bin/python bench/xfer_exact.py --sizes-json results/<d>_E11c_groot_n17_handover/xfer_sizes.json \\
        --model-name groot_n17 --outdir results/<d>_E11c_groot_n17_handover_xfer
"""
from __future__ import annotations

import csv
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REPO, collect_raw, compress_and_record, sh_out, software_versions,  # noqa: E402
                    summarize, write_json, write_summary)
from e11_summaries import rd, table  # noqa: E402
from host_segments import ModelInputs  # noqa: E402

GP_FILE = REPO / "bench" / "groot_phases.py"
PM = "MAXN+jetson_clocks"


# ----------------------------------------------------------------------------------------------- shared helpers

def _gp():
    import groot_phases as GP                                          # noqa: PLC0415
    return GP


def _model_name(args) -> str:
    return f"groot_{args.version}"


def _style(ctx) -> str:
    return "compiled" if ctx["compiled_desc"] else "eager"


def _mode_label(args, ctx) -> str:
    # pipelining.emit_model_inputs derives exec_style from mode == "eager"
    return "eager" if not ctx["compiled_desc"] else args.rung


def _call(model, bo, ai, action_head_fn=None):
    """action_head.get_action with N1.7's extra `options` argument when it has one (as run_power checks)."""
    fn = action_head_fn or model.action_head.get_action
    n_params = len(inspect.signature(fn).parameters)
    return fn(bo, ai, None) if n_params >= 3 else fn(bo, ai)


def _bf(bo):
    """A shallow copy of a BatchFeature: the head's process_backbone_output REPLACES backbone_features in the dict it
    is given (gr00t_n1d7.py:179), so every call needs its own dict or the next one would apply vlln twice."""
    from transformers.feature_extraction_utils import BatchFeature     # noqa: PLC0415
    return BatchFeature(data=dict(bo))


def _eager_head(model):
    """The action head's CLASS get_action, bound: bypasses a torch.compile'd instance attribute and the region timer,
    so the DiT's Python body really runs (a CUDA-graph replay runs none and a hook would record nothing)."""
    ah = model.action_head
    return type(ah).get_action.__get__(ah, type(ah))


def _step(args, model, inputs):
    """One inference, as groot_phases.main's step() (which closes over its own inputs, so a new batch needs this)."""
    if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()
    if args.api == "old":
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model.get_action(inputs)
    with torch.no_grad():
        return model.get_action(inputs)


def _finite(out) -> tuple[bool, int]:
    a = out["action_pred"] if hasattr(out, "__getitem__") and "action_pred" in out else out
    a = a.float()
    bad = int((~torch.isfinite(a)).sum().item())
    return bad == 0, bad


def _ps(model, inputs) -> dict:
    """P, P_valid, image / text tokens and S, read from one real backbone pass and the head's own feature encoder."""
    with torch.no_grad():
        bi, ai = model.prepare_input(inputs)
        bo = model.backbone(bi)
        mask = bo["backbone_attention_mask"].bool()
        im = bo["image_mask"].bool()
        feats = model.action_head._encode_features(_bf(bo), ai)
    S = int(feats["state_features"].shape[1]) + int(model.action_head.config.action_horizon)
    P, P_valid = int(mask.shape[1]), int(mask.sum().item())
    img = int((im & mask).sum().item())
    return {"P": P, "P_valid": P_valid, "image_tokens": img, "text_tokens_valid": P_valid - img, "S": S,
            "state_tokens": int(feats["state_features"].shape[1]),
            "action_horizon": int(model.action_head.config.action_horizon),
            "bo": bo, "ai": ai, "feats": feats, "bi": bi}


def _dit_dims(model) -> dict:
    dit = model.action_head.model
    cfg = dit.config
    H, HD = int(cfg.num_attention_heads), int(cfg.attention_head_dim)
    blk = dit.transformer_blocks[0]
    lin = [m for m in blk.ff.modules() if isinstance(m, torch.nn.Linear)]
    return {"H": H, "HD": HD, "D": H * HD, "I": int(lin[0].out_features), "n_blocks": len(dit.transformer_blocks),
            "cross_blocks": [i for i, b in enumerate(dit.transformer_blocks)
                             if getattr(b.attn1, "is_cross_attention", False)],
            "ff_source": f"first nn.Linear of transformer_blocks[0].ff: {lin[0]}"}


def _nonull(o, path="", found=None):
    found = [] if found is None else found
    if isinstance(o, dict):
        for k in list(o):
            if o[k] is None:
                o[k] = "none"
                found.append(f"{path}.{k}")
            else:
                _nonull(o[k], f"{path}.{k}", found)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            if v is None:
                o[i] = "none"
                found.append(f"{path}[{i}]")
            else:
                _nonull(v, f"{path}[{i}]", found)
    return found


def _meta(args, ctx, experiment: str, extra: dict) -> dict:
    GP = _gp()
    g = ctx["gate"]
    sw = software_versions()
    prov = GP._provenance(args, g)
    m = {"experiment": experiment, "task": "T012",
         "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "model": _model_name(args), "checkpoint": args.checkpoint, "model_repo": args.checkpoint,
         "embodiment_tag": args.embodiment_tag, "rung": args.rung,
         "compile": ";".join(ctx["compiled_desc"]) or "eager",
         "compile_seconds": round(float(ctx["compile_s"]), 2),
         "precision": "bfloat16", "matmul_precision": "high" if args.tf32 == "on" else "highest",
         "batch": 1, "num_denoise_steps": ctx["n"],
         "observation_shapes": {k: {kk: list(vv) for kk, vv in v.items()} for k, v in ctx["shapes"].items()},
         "observation_values": "random pixels / state from np.random (latency has no data-dependent control flow; "
                               "see groot_phases.build_observation)",
         "power_mode": g["power_mode"], "sm_count": g.get("sm_count"), "gpu_hz": g.get("gpu_cur_freq_hz"),
         "gate": g, "emc_hz": int(sh_out("sudo -n sh -c 'cat /sys/kernel/debug/bpmp/debug/clk/emc/rate'") or 0),
         "jetpack": "6.2.3+b81", "l4t": "R36.5.0", "cuda": sw.get("torch_cuda"), "torch": sw.get("torch"),
         "triton": sw.get("triton"), "python": sw.get("python"), "venv": sw.get("executable"),
         "bench_files_sha256": sw.get("bench_files_sha256"),
         "gr00t_root": args.gr00t_root or "default",
         "command": "python bench/groot_phases.py " + " ".join(sys.argv[1:]),
         **prov, **extra}
    return m


def _write_meta(out: Path, meta: dict) -> None:
    replaced = _nonull(meta)
    meta["null_fields_replaced"] = replaced or "none"
    write_json(out / "meta.json", meta)
    meta["raw_files"] = collect_raw(out, meta["command"])
    write_json(out / "meta.json", meta)


def _csv(path: Path, rows: list[dict]) -> None:
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in keys} for r in rows])


def _ev_ms(fn, warmup: int, n: int) -> list[float]:
    from handover import timed_event                                   # noqa: PLC0415
    return timed_event(fn, warmup, n)


def _wall_ms(fn, warmup: int, n: int) -> list[float]:
    from handover import timed_wall                                    # noqa: PLC0415
    return timed_wall(fn, warmup, n)


def _desc(t) -> dict:
    return {"shape": str(tuple(t.shape)), "dtype": str(t.dtype).split(".")[-1],
            "stride": str(tuple(t.stride())), "contiguous": t.is_contiguous(),
            "bytes": t.numel() * t.element_size()}


# ----------------------------------------------------------------------------------------------- E11c handover

def handover(args, ctx) -> int:
    if ctx["compiled_desc"]:
        raise SystemExit("E11c describes the model's own code path: run --e11 handover with the eager rung "
                         "(no --head-mode / --llm-mode); hooks cannot see inside a CUDA-graph replay")
    model, out, n = ctx["model"], ctx["out"], ctx["n"]
    ah, dit = model.action_head, model.action_head.model
    reps = max(args.timed, 200)
    ps = _ps(model, ctx["inputs"])
    bo, ai = ps["bo"], ps["ai"]
    dims = _dit_dims(model)

    # --- handover.csv: what the backbone returns (gr00t_n1d7.py:603-612), then what the head derives ONCE per
    # inference from it (process_backbone_output, gr00t_n1d7.py:175-179: vlln + vl_self_attention)
    hrows = []
    for k, v in bo.items():
        if torch.is_tensor(v):
            hrows.append({"model": _model_name(args), "stage": "backbone_output", "tensor": k, **_desc(v),
                          "valid_tokens": ps["P_valid"], "padded_tokens": ps["P"] - ps["P_valid"],
                          "image_tokens": ps["image_tokens"], "text_tokens_valid": ps["text_tokens_valid"],
                          "per": "inference", "notes": "returned by model.backbone (the hand-over object)"
                          if k == "backbone_features" else "mask carried with the features"})
    for k in ("backbone_features", "state_features"):
        v = ps["feats"][k]
        hrows.append({"model": _model_name(args), "stage": "after_encode_features", "tensor": k, **_desc(v),
                      "valid_tokens": ps["P_valid"] if k == "backbone_features" else "",
                      "padded_tokens": ps["P"] - ps["P_valid"] if k == "backbone_features" else "",
                      "per": "inference",
                      "notes": ("vlln + vl_self_attention applied once per inference "
                                "(gr00t_n1d7.py:175-179, :307); this is what every DiT step reads as "
                                "encoder_hidden_states") if k == "backbone_features"
                      else "state_encoder once per inference (gr00t_n1d7.py:321); concatenated with the "
                           "action tokens every step (:412)"})
    feat = bo["backbone_features"]
    tot_bytes = feat.numel() * feat.element_size()

    # --- one real inference through the head's own code, instrumented
    log = {"to_k_calls": {}, "to_v_calls": {}, "enc_ptrs": [], "steps": [], "dit_in": [], "dit_out": [],
           "dec_out": [], "enc_in": []}
    undo = []

    def wrap(obj, attr, fn):
        orig = getattr(obj, attr)
        setattr(obj, attr, fn(orig))
        undo.append(lambda o=obj, a=attr, f=orig: setattr(o, a, f))

    blocks = dit.transformer_blocks
    for i, b in enumerate(blocks):
        a = b.attn1
        for nm, d in (("to_k", log["to_k_calls"]), ("to_v", log["to_v_calls"])):
            lin = getattr(a, nm)

            def mk(orig, i=i, d=d):
                def f(x, *aa, **kk):
                    d[i] = d.get(i, 0) + 1
                    return orig(x, *aa, **kk)
                return f
            wrap(lin, "forward", mk)

    def dit_fwd(orig):
        def f(*aa, **kk):
            hs = kk.get("hidden_states", aa[0] if aa else None)
            ehs = kk.get("encoder_hidden_states", aa[1] if len(aa) > 1 else None)
            ts = kk.get("timestep", aa[2] if len(aa) > 2 else None)
            log["dit_in"].append({"hidden_states": _desc(hs), "timestep": _desc(ts),
                                  "encoder_hidden_states": _desc(ehs), "enc_ptr": ehs.data_ptr()})
            log["ehs"] = ehs
            r = orig(*aa, **kk)
            log["dit_out"].append(_desc(r))
            return r
        return f
    wrap(dit, "forward", dit_fwd)

    def enc_fwd(orig):
        def f(actions, ts, emb, *aa, **kk):
            log["enc_in"].append({"actions": _desc(actions), "timestep": _desc(ts)})
            return orig(actions, ts, emb, *aa, **kk)
        return f
    wrap(ah.action_encoder, "forward", enc_fwd)

    def dec_fwd(orig):
        def f(*aa, **kk):
            r = orig(*aa, **kk)
            log["dec_out"].append(_desc(r))
            return r
        return f
    wrap(ah.action_decoder, "forward", dec_fwd)
    try:
        with torch.no_grad():
            res = _call(model, _bf(bo), ai, _eager_head(model))
        torch.cuda.synchronize()
    finally:
        for u in reversed(undo):
            u()
    ok, bad = _finite(res)
    steps_seen = len(log["dit_in"])
    cross = dims["cross_blocks"]                 # attn1.is_cross_attention, read from the modules
    log["to_k_calls"] = {i: c for i, c in log["to_k_calls"].items() if i in cross}
    print(f"  handover: {steps_seen} DiT calls; to_k calls per block {log['to_k_calls']}; finite={ok}", flush=True)

    # --- cross-attention K/V: bytes and time per step, on the REAL encoder states the DiT received
    ehs = log["ehs"]
    kv_rows, kv_step_bytes = [], 0
    for i in cross:
        a = blocks[i].attn1
        inner_k, inner_v = a.to_k.out_features, a.to_v.out_features
        nb = ehs.shape[0] * ehs.shape[1] * (inner_k + inner_v) * ehs.element_size()
        kv_step_bytes += nb
        ms = _ev_ms(lambda a=a: (a.to_k(ehs), a.to_v(ehs)), 20, reps)
        s = summarize(ms)
        kv_rows.append({"block": i, "to_k": f"{a.to_k.in_features}->{inner_k}",
                        "to_v": f"{a.to_v.in_features}->{inner_v}", "tokens": ehs.shape[1],
                        "kv_bytes_written": nb, "weight_bytes_read": (a.to_k.weight.numel() + a.to_v.weight.numel())
                        * a.to_k.weight.element_size(),
                        "calls_per_inference": log["to_k_calls"].get(i, 0),
                        "median_ms": round(s["median_ms"], 6), "p5_ms": round(s["p5_ms"], 6),
                        "p95_ms": round(s["p95_ms"], 6), "n": s["n"], "timing": "gpu_event"})

    def all_kv():
        return [(blocks[i].attn1.to_k(ehs), blocks[i].attn1.to_v(ehs)) for i in cross]
    kv_all = summarize(_ev_ms(all_kv, 20, reps))
    kv_all_wall = summarize(_wall_ms(all_kv, 20, reps))

    # --- materialise analogues: a copy of the hand-over object, and the once-per-inference feature encoder
    def clone_bo():
        return {k: v.clone() for k, v in bo.items() if torch.is_tensor(v)}

    def encode():
        with torch.no_grad():
            return ah._encode_features(_bf(bo), ai)
    mat = {}
    for what, fn in (("clone_backbone_output", clone_bo), ("encode_features_once", encode)):
        for timing, tf in (("wall_isolated", _wall_ms), ("gpu_event", _ev_ms)):
            mat[(what, timing)] = summarize(tf(fn, 20, reps))
            print(f"  {what:24s} {timing:14s} {mat[(what, timing)]['median_ms']:8.4f} ms", flush=True)

    # --- source evidence for "K/V recomputed every step"
    proc = type(blocks[cross[0]].attn1.processor) if cross else None
    src_ref = "n/a"
    if proc is not None:
        lines, start = inspect.getsourcelines(proc.__call__)
        hit = [start + j for j, ln in enumerate(lines) if "to_k(encoder_hidden_states)" in ln]
        src_ref = f"{inspect.getsourcefile(proc)}:{hit[0] if hit else start} ({proc.__name__}.__call__)"
    ptrs = sorted({d["enc_ptr"] for d in log["dit_in"]})

    # --- boundary.csv, per step, from the hooks
    e0, d0, o0 = log["enc_in"][0], log["dit_in"][0], log["dec_out"][0]
    brows = [
        {"model": _model_name(args), "tensor": "backbone_features", "direction": "gpu_to_action_head",
         **{k: _desc(feat)[k] for k in ("shape", "dtype", "bytes")}, "per": "inference",
         "notes": "the hand-over object (backbone output)"},
        {"model": _model_name(args), "tensor": "noisy_actions", "direction": "into_action_head",
         **{k: e0["actions"][k] for k in ("shape", "dtype", "bytes")}, "per": "step",
         "notes": "action_encoder input each step (gr00t_n1d7.py:405)"},
        {"model": _model_name(args), "tensor": "timestep", "direction": "into_action_head",
         **{k: e0["timestep"][k] for k in ("shape", "dtype", "bytes")}, "per": "step",
         "notes": "discretised timestep tensor (gr00t_n1d7.py:401-404)"},
        {"model": _model_name(args), "tensor": "state_features", "direction": "inside_action_head",
         **{k: _desc(ps["feats"]["state_features"])[k] for k in ("shape", "dtype", "bytes")}, "per": "inference",
         "notes": "state encoded ONCE per inference (gr00t_n1d7.py:321) and concatenated each step (:412)"},
        {"model": _model_name(args), "tensor": "dit_hidden_states", "direction": "into_dit",
         **{k: d0["hidden_states"][k] for k in ("shape", "dtype", "bytes")}, "per": "step",
         "notes": "sa_embs = cat(state_features, action_features)"},
        {"model": _model_name(args), "tensor": "velocity", "direction": "out_of_action_head",
         **{k: o0[k] for k in ("shape", "dtype", "bytes")}, "per": "step",
         "notes": "action_decoder output; the last action_horizon rows are the Euler update's velocity "
                  "(gr00t_n1d7.py:430-435)"},
    ]
    per_step_total = sum(int(r["bytes"]) for r in brows if r["per"] == "step" and r["tensor"] != "dit_hidden_states")
    brows.append({"model": _model_name(args), "tensor": "total", "direction": "both", "shape": "", "dtype": "",
                  "bytes": per_step_total, "per": "step",
                  "notes": "noisy_actions + timestep + velocity (dit_hidden_states is inside the head)"})

    _csv(out / "handover.csv", hrows)
    _csv(out / "boundary.csv", brows)
    _csv(out / "cross_kv.csv", kv_rows)
    with (out / "kv_copy.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["what", "timing", "median_ms", "p5_ms", "p95_ms", "min_ms", "n", "bytes", "per", "source"])
        w.writerow(["cross-attention to_k/to_v on the features, all cross blocks", "gpu_event",
                    round(kv_all["median_ms"], 6), round(kv_all["p5_ms"], 6), round(kv_all["p95_ms"], 6),
                    round(kv_all["min_ms"], 6), kv_all["n"], kv_step_bytes, "step", src_ref])
        w.writerow(["cross-attention to_k/to_v on the features, all cross blocks", "wall_isolated",
                    round(kv_all_wall["median_ms"], 6), round(kv_all_wall["p5_ms"], 6),
                    round(kv_all_wall["p95_ms"], 6), round(kv_all_wall["min_ms"], 6), kv_all_wall["n"],
                    kv_step_bytes, "step", src_ref])
        for (what, timing), s in mat.items():
            w.writerow([what, timing, round(s["median_ms"], 6), round(s["p5_ms"], 6), round(s["p95_ms"], 6),
                        round(s["min_ms"], 6), s["n"], tot_bytes if what == "clone_backbone_output" else "",
                        "inference", "harness copy" if what == "clone_backbone_output"
                        else "gr00t_n1d7.py:307-323 (vlln, vl_self_attention, state_encoder)"])
    (out / "raw").mkdir(exist_ok=True)
    write_json(out / "raw" / "hook_log.json", {k: v for k, v in log.items() if k != "ehs"})

    # --- xfer_sizes.json for bench/xfer_exact.py --sizes-json (every dimension read from the loaded modules)
    H, D, I = dims["H"], dims["D"], dims["I"]
    S, T = ps["S"], ps["P"]
    prov_commit = _gp()._provenance(args, ctx["gate"])["model_commit"]
    xs = {"scores_self_fp32": H * S * S * 4, "scores_self_fp16": H * S * S * 2,
          "scores_cross_fp32": H * S * T * 4, "scores_cross_fp16": H * S * T * 2,
          "activation_tile": S * D * 2, "ffn_tile": S * I * 2, "kv_bytes_total": tot_bytes,
          "cross_kv_per_step": kv_step_bytes}
    write_json(out / "xfer_sizes.json", {
        "sizes": xs,
        "info": {"P": T, "S": S, "D": D, "I": I, "query_heads": H, "head_dim": dims["HD"],
                 "model_commit": prov_commit, "model_repo": args.checkpoint, "checkpoint": args.checkpoint,
                 "num_denoise_steps": n, "action_horizon": ps["action_horizon"],
                 "derivation": "H, head_dim from action_head.model.config; I from the first Linear of "
                               "transformer_blocks[0].ff; S = state tokens + action_horizon; T = backbone tokens "
                               "(P); kv_bytes_total = backbone_features bytes as handed over; cross_kv_per_step = "
                               "K+V written by all cross blocks' to_k/to_v per step",
                 "source": f"{out}/handover.csv, cross_kv.csv"}})

    # --- kv_reuse.md
    n_cross = len(cross)
    (out / "kv_reuse.md").write_text(f"""# How the denoising steps use the backbone output (GR00T N1.7)

**Answer: the hand-over object is the backbone's hidden states (`backbone_features`, {feat.shape[1]} tokens x
{feat.shape[2]}, {_desc(feat)['dtype']}, {tot_bytes:,} B), not a KV cache. It is passed by reference to every
denoising step, but the {n_cross} cross-attention blocks project it through their own `to_k` / `to_v` at EVERY step:
{kv_step_bytes:,} B of K/V are re-computed per step ({kv_step_bytes * n:,} B per inference at N = {n}),
costing {kv_all['median_ms']:.4f} ms per step (CUDA events, n = {kv_all['n']}).**

## Evidence from the source
* `Gr00tN1d7.get_action` (gr00t_n1d7.py:603-612): backbone -> action_head.get_action(backbone_outputs, ...).
* `get_action_with_features` (gr00t_n1d7.py:397-435): the denoising loop passes the same `vl_embeds` as
  `encoder_hidden_states` to `self.model(...)` at every step.
* `AlternateVLDiT.forward` (dit.py:349-415): even blocks cross-attend (idx % 4 == 0 onto the non-image tokens, the
  other even blocks onto the image tokens), odd blocks self-attend.
* The attention processor projects the features inside every call: `{src_ref}`.

## Evidence from the running process (one real inference, N = {n}, `raw/hook_log.json`)
| quantity | value |
|---|---|
| DiT forward calls | {steps_seen} |
| `to_k` calls per cross block | {json.dumps(log['to_k_calls'])} |
| distinct `encoder_hidden_states` data_ptr across steps | {len(ptrs)} (the features are referenced, not copied) |
| K/V bytes written per step (all cross blocks) | {kv_step_bytes:,} B |
| to_k/to_v time per step, all cross blocks | {kv_all['median_ms']:.4f} ms (gpu_event), {kv_all_wall['median_ms']:.4f} ms (wall) |
| valid backbone tokens | {ps['P_valid']} of {ps['P']} ({ps['image_tokens']} image, {ps['text_tokens_valid']} text) |

T016's `kvonce` variant (bench/groot_savings.py) computes these K/V once per frame instead and was measured in
`reports/E14_prefix_savings.md`.
""")

    # --- model_inputs.csv
    bench_commit = sh_out(f"git -C {REPO} rev-parse HEAD")
    mi = ModelInputs(_model_name(args), prov_commit, bench_commit, PM, ps["P"], ps["S"], n)
    mi.add("kv_bytes_total", tot_bytes, "bytes", "value", 1, "handover.csv", key="padded")
    mi.add("kv_bytes_total", tot_bytes * ps["P_valid"] // ps["P"], "bytes", "value", 1, "handover.csv",
           key="valid")
    for r in kv_rows:
        mi.add("kv_bytes_per_layer", r["kv_bytes_written"], "bytes", "value", 1, "cross_kv.csv",
               key="cross_kv_projection", variant=f"block={r['block']}")
    mi.add("kv_copied_bytes_per_step", kv_step_bytes, "bytes", "value", 1, "kv_reuse.md",
           variant="basis=cross_kv_recomputed_per_step")
    mi.add_stats("t_kv_materialise", kv_all, "ms", kv_all["n"], "kv_copy.csv", key="cross_kv_projection",
                 variant="per=step", timing="gpu_event")
    mi.add_stats("t_kv_materialise", kv_all_wall, "ms", kv_all_wall["n"], "kv_copy.csv",
                 key="cross_kv_projection", variant="per=step", timing="wall_isolated")
    for (what, timing), s in mat.items():
        mi.add_stats("t_kv_materialise", s, "ms", s["n"], "kv_copy.csv", key=what, variant="per=inference",
                     timing=timing)
    for r in brows:
        if r["per"] == "step" and r["tensor"] != "dit_hidden_states":
            mi.add("boundary_bytes_per_step", r["bytes"], "bytes", "value", 1, "boundary.csv", key=r["tensor"])
    n_mi = mi.write(out / "model_inputs.csv")

    write_summary(out, "E11c - GR00T N1.7 hand-over object (protocol §13.4)", [
        ("Answer", (out / "kv_reuse.md").read_text().split("## Evidence from the source")[0].split("\n", 2)[2]),
        ("Hand-over object", table(hrows, ["stage", "tensor", "shape", "dtype", "bytes", "valid_tokens",
                                           "padded_tokens", "notes"])),
        ("Boundary tensors", table(brows, ["tensor", "direction", "shape", "dtype", "bytes", "per", "notes"])),
        ("Cross-attention K/V per step", table(kv_rows, ["block", "tokens", "kv_bytes_written", "median_ms",
                                                         "p5_ms", "p95_ms", "n"])),
        ("Conditions", f"GR00T N1.7 eager (bf16), MAXN + jetson_clocks, N = {n}, P = {ps['P']} backbone tokens, "
                       f"S = {ps['S']}. Transfers at these sizes: xfer_sizes.json -> bench/xfer_exact.py."),
        ("Reproduce", f"```\npython bench/groot_phases.py {' '.join(sys.argv[1:])}\n```"),
    ], [])
    _write_meta(out, _meta(args, ctx, "E11c (GR00T hand-over)", {
        "prefix_tokens": ps["P"], "prefix_tokens_valid": ps["P_valid"], "image_tokens": ps["image_tokens"],
        "suffix_tokens": ps["S"], "action_horizon": ps["action_horizon"], "n_warmup": 20, "n_timed": reps,
        "dit": {k: v for k, v in dims.items()}, "cross_blocks_seen": cross,
        "kv_bytes_total": tot_bytes, "cross_kv_bytes_per_step": kv_step_bytes,
        "dit_calls_observed": steps_seen, "distinct_encoder_ptrs": len(ptrs),
        "kv_projection_source": src_ref, "actions_finite": ok, "actions_nonfinite": bad,
        "model_inputs_rows": n_mi, "raw_files": [],
        "notes": ["Hooks are instance attributes installed for ONE inference through the head's class get_action "
                  "and removed; timing runs afterwards without them.",
                  "kv_copied_bytes_per_step carries the cross-attention K/V re-computed each step (variant "
                  "basis=cross_kv_recomputed_per_step): GR00T has no KV cache to copy; the per-step re-projection "
                  "is the analogous per-step re-materialisation.",
                  "kv_bytes_per_layer rows (key cross_kv_projection) are per cross-attention block, per step."]}))
    print(f"\nwrote {out} ({n_mi} model_inputs rows)", flush=True)
    return 0 if ok else 3


# ----------------------------------------------------------------------------------------------- E11d prefix sweep

# short must be SHORTER than typical (the fixture's "pick up the object", 4 tokens): the first run used
# "pick up the cup", also 4 tokens, so short and typical were the same point.
PROMPTS = {"short": "pick",
           "typical": None,
           "long": ("pick up the red cup from the left side of the table, move it carefully over the white plate "
                    "without touching the bowl, place it upright in the center of the plate, then release the "
                    "gripper, move the arm back to its home position above the table and wait there for the next "
                    "instruction from the operator before doing anything else")}


def _find_tokenizer(policy):
    seen, frontier = set(), [("policy.processor", getattr(policy, "processor", None))]
    for _ in range(5):
        nxt = []
        for path, o in frontier:
            if o is None or id(o) in seen:
                continue
            seen.add(id(o))
            if hasattr(o, "model_max_length") and callable(o):
                return o, path
            d = o if isinstance(o, dict) else getattr(o, "__dict__", {})
            for k, v in list(d.items())[:200]:
                if isinstance(v, (str, int, float, bool, bytes)) or torch.is_tensor(v):
                    continue
                nxt.append((f"{path}.{k}", v))
        frontier = nxt
    return None, "not found"


def prefixsweep(args, ctx) -> int:
    import torch._dynamo as _dyn                                       # noqa: PLC0415
    import prefix_sweep as PSW                                         # noqa: PLC0415
    GP = _gp()
    model, policy, out, n, spans = ctx["model"], ctx["policy"], ctx["out"], ctx["n"], ctx["spans"]
    base = ctx["obs"]
    lk = policy.language_key
    base_text = base["language"][lk][0]
    vkeys = list(base["video"])
    old_limit = _dyn.config.cache_size_limit
    _dyn.config.cache_size_limit = max(old_limit, 64)
    tok, tok_path = _find_tokenizer(policy)
    tok_limit = int(getattr(tok, "model_max_length", 0) or 0) if tok is not None else 0
    eager = not ctx["compiled_desc"]
    dit = model.action_head.model
    step_ev = []
    if eager:
        # per-step DiT time from events around the DiT forward (eager only: a hook cannot see into a graph)
        orig_dit = dit.forward

        def dit_timed(*a, **k):
            s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            s.record()
            r = orig_dit(*a, **k)
            e.record()
            step_ev.append((s, e))
            return r
        dit.forward = dit_timed

    def img_set(k):
        if k == 1:
            return {vkeys[0]: base["video"][vkeys[0]]}
        if k == 2:
            return dict(base["video"])
        v = dict(base["video"])
        v["extra_camera_0"] = base["video"][vkeys[0]].copy()
        return v

    rows, samples, refused, first, img_tok = [], [], [], {}, {}
    finite_bad = 0
    for n_img in (1, 2, 3):
        for plabel, ptext in PROMPTS.items():
            text = base_text if ptext is None else ptext
            label = f"{n_img}img_{plabel}"
            obs = {"video": img_set(n_img), "state": base["state"], "language": {lk: [text]}}
            try:
                inp = GP.build_inputs(policy, obs)
                t0 = time.perf_counter()          # the first call carries any (re)compile for this shape
                _step(args, model, inp)
                torch.cuda.synchronize()
                first[label] = round(time.perf_counter() - t0, 3)
                ps = _ps(model, inp)
                img_tok.setdefault(n_img, ps["image_tokens"])
                if n_img > 1 and ps["image_tokens"] <= img_tok.get(n_img - 1, -1):
                    # the processor accepted the extra key but did not use it: the "3 image" point would be a
                    # 2-image point with a misleading label (first run, 2026-09-28: 256 image tokens for both)
                    raise ValueError(f"extra image key(s) silently ignored by the processor: {ps['image_tokens']} "
                                     f"image tokens with {n_img} images vs {img_tok[n_img - 1]} with {n_img - 1}")
            except Exception as e:                                     # noqa: BLE001
                refused.append({"point": label, "n_images": n_img, "prompt_label": plabel,
                                "error": f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"})
                print(f"  {label:16s} REFUSED: {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            for _ in range(args.warmup):
                _step(args, model, inp)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            reg = {"vision": [], "vlm_prefix": [], "ae_total": [], "ae_step": [], "e2e": []}
            res = None
            for _ in range(args.timed):
                for v in spans.values():
                    v.clear()
                step_ev.clear()
                torch.cuda.synchronize()
                t = time.perf_counter()
                res = _step(args, model, inp)
                torch.cuda.synchronize()
                reg["e2e"].append((time.perf_counter() - t) * 1e3)
                reg["vision"].append(sum(s.elapsed_time(e) for s, e in spans["vision"]))
                reg["vlm_prefix"].append(sum(s.elapsed_time(e) for s, e in spans["vlm_backbone"]))
                reg["ae_total"].append(sum(s.elapsed_time(e) for s, e in spans["ae_total"]))
                if eager and step_ev:
                    reg["ae_step"].append(sum(s.elapsed_time(e) for s, e in step_ev) / len(step_ev))
            peak = torch.cuda.max_memory_allocated()
            ok, bad = _finite(res)
            finite_bad += bad
            n_tok = len(tok(text)["input_ids"]) if tok is not None else ""
            base_row = {"mode": args.rung, "n_images": n_img, "image_keys_supplied": ";".join(obs["video"]),
                        "prompt_label": plabel, "prompt_tokenizer_tokens": n_tok,
                        "prompt_tokens": ps["text_tokens_valid"], "image_tokens": ps["image_tokens"],
                        "P_processed": ps["P"], "P_valid": ps["P_valid"], "S_processed": ps["S"],
                        "peak_mem_bytes": peak, "first_call_s": first[label], "actions_finite": ok}
            for region, vals in reg.items():
                if not vals:
                    continue
                st = summarize(vals)
                rows.append({**base_row, "region": region, "median_ms": round(st["median_ms"], 5),
                             "p5_ms": round(st["p5_ms"], 5), "p95_ms": round(st["p95_ms"], 5), "n": st["n"]})
                samples += [{"mode": args.rung, "point": label, "region": region, "iter": i, "ms": v}
                            for i, v in enumerate(vals)]
            print(f"  {label:16s} P={ps['P']} valid={ps['P_valid']} img={ps['image_tokens']} "
                  f"text={ps['text_tokens_valid']}  e2e {summarize(reg['e2e'])['median_ms']:8.2f} ms  "
                  f"first {first[label]:.1f} s", flush=True)
    if eager:
        dit.forward = orig_dit
    _dyn.config.cache_size_limit = old_limit
    if not rows:
        raise SystemExit(f"every point refused: {refused}")
    _csv(out / "prefix_sweep.csv", rows)
    _csv(out / "latency_samples.csv", samples)
    if refused:
        _csv(out / "raw" / "refused_points.csv", refused)

    fits = []
    for key, xname, regions_ in (("vision_vs_images", "n_images", ("vision",)),
                                 ("vlm_prefix_vs_P", "P_processed", ("vlm_prefix",)),
                                 ("ae_step_vs_P", "P_processed", ("ae_step", "ae_total", "e2e"))):
        for region in regions_:
            agg = {}
            for r in rows:
                if r["region"] == region:
                    agg.setdefault(r[xname], []).append(r["median_ms"])
            if len(agg) < 2:
                fits.append({"fit": key, "region": region, "x": xname, "fit_a_ms": "", "fit_b": "", "unit_b": "",
                             "fit_r2": "", "n_points": len(agg), "residuals": "",
                             "note": f"x takes {len(agg)} value(s) across the accepted points; no slope"})
                continue
            xs = sorted(agg)
            ys = [sorted(agg[x])[len(agg[x]) // 2] for x in xs]
            a, b, r2, res_ = PSW.linfit(xs, ys)
            fits.append({"fit": key, "region": region, "x": xname, "fit_a_ms": round(a, 5), "fit_b": round(b, 8),
                         "unit_b": "ms_per_image" if xname == "n_images" else "ms_per_token",
                         "fit_r2": round(r2, 6), "n_points": len(xs),
                         "residuals": ";".join(f"{v:.4f}" for v in res_), "note": ""})
    _csv(out / "prefix_fit.csv", fits)

    prov_commit = _gp()._provenance(args, ctx["gate"])["model_commit"]
    style = _style(ctx)
    mi = ModelInputs(_model_name(args), prov_commit, sh_out(f"git -C {REPO} rev-parse HEAD"), PM,
                     rows[0]["P_processed"], rows[0]["S_processed"], n)
    qmap = {"vision": "t_vision", "vlm_prefix": "t_vlm_prefix", "ae_step": "t_ae_step", "ae_total": "t_ae_total",
            "e2e": "t_e2e"}
    for r in rows:
        var = f"n_images={r['n_images']};prompt={r['prompt_label']};P_valid={r['P_valid']};rung={args.rung}"
        for stat, col in (("median", "median_ms"), ("p5", "p5_ms"), ("p95", "p95_ms")):
            mi.add(qmap[r["region"]], r[col], "ms", stat, r["n"], "prefix_sweep.csv", variant=var,
                   exec_style=style, timing="wall_isolated" if r["region"] == "e2e" else "gpu_event")
    for f_ in fits:
        if f_["fit_a_ms"] == "":
            continue
        v = f"region={f_['region']};rung={args.rung}"
        mi.add("prefix_latency_fit", f_["fit_a_ms"], "ms", "fit_a", f_["n_points"], "prefix_fit.csv",
               key=f_["fit"], variant=v, exec_style=style)
        mi.add("prefix_latency_fit", f_["fit_b"], f_["unit_b"], "fit_b", f_["n_points"], "prefix_fit.csv",
               key=f_["fit"], variant=v, exec_style=style)
        mi.add("prefix_latency_fit", f_["fit_r2"], "ratio", "fit_r2", f_["n_points"], "prefix_fit.csv",
               key=f_["fit"], variant=v, exec_style=style)
    n_mi = mi.write(out / "model_inputs.csv")

    nm = [f"{r['point']}: refused by the checkpoint's processor -- {r['error']}" for r in refused]
    if not eager:
        nm.append("t_ae_step (compiled): the action head replays as one CUDA graph (reduce-overhead), so there is no "
                  "per-step boundary to time; ae_total / N is a derivation, not a measurement, and is not written")
    write_summary(out, f"E11d - GR00T N1.7 prefix vs input ({args.rung})", [
        ("Answer", table(rows, ["n_images", "prompt_label", "prompt_tokens", "image_tokens", "P_processed",
                                "P_valid", "region", "median_ms", "p5_ms", "p95_ms", "n"],
                         nd={"median_ms": 3, "p5_ms": 3, "p95_ms": 3})),
        ("Fits", table(fits, ["fit", "region", "x", "fit_a_ms", "fit_b", "unit_b", "fit_r2", "n_points", "note"])),
        ("Conditions", f"rung `{args.rung}` ({'; '.join(ctx['compiled_desc']) or 'eager'}), N = {n}, MAXN. "
                       f"The embodiment declares cameras {vkeys}; the 1-image point drops `{vkeys[-1]}`, the 3-image "
                       f"point adds `extra_camera_0` (a copy of `{vkeys[0]}`). vlm_prefix = the whole backbone "
                       f"(vision INSIDE, T012 step 4); vision = the image-encoder calls within it. Prompt limit: "
                       f"tokenizer model_max_length = {tok_limit} ({tok_path})."),
        ("Reproduce", f"```\npython bench/groot_phases.py {' '.join(sys.argv[1:])}\n```"),
    ], nm)
    _write_meta(out, _meta(args, ctx, "E11d (GR00T prefix vs input)", {
        "prefix_tokens": rows[0]["P_processed"], "suffix_tokens": rows[0]["S_processed"],
        "n_warmup": args.warmup, "n_timed": args.timed, "first_call_seconds_per_shape": first,
        "tokenizer_max_length": tok_limit, "tokenizer_found_at": tok_path,
        "camera_keys_declared": vkeys, "prompts": {k: (v or base_text) for k, v in PROMPTS.items()},
        "refused_points": refused or "none", "dynamo_cache_size_limit": max(old_limit, 64),
        "dynamo_cache_size_limit_default": old_limit,
        "actions_nonfinite_total": finite_bad, "not_measured": nm or "none",
        "model_inputs_rows": n_mi, "raw_files": [],
        "notes": ["regions from groot_phases' CUDA-event timers around backbone.forward (vlm_prefix), "
                  ".visual.forward (vision) and action_head.get_action (ae_total); e2e is wall clock",
                  "ae_step (eager only) = mean of CUDA-event times around action_head.model.forward (the DiT) per "
                  "inference",
                  "torch._dynamo cache_size_limit raised to 64 for the sweep so a new prompt / image shape "
                  "recompiles instead of silently falling back to eager after 8 shapes"]}))
    print(f"\nwrote {out} ({n_mi} model_inputs rows)", flush=True)
    return 0 if finite_bad == 0 else 3


# ----------------------------------------------------------------------------------------------- E11e mempressure

def mempressure(args, ctx) -> int:
    import mempressure as MP                                           # noqa: PLC0415
    GP = _gp()
    model, inputs, step, out, n = ctx["model"], ctx["inputs"], ctx["step"], ctx["out"], ctx["n"]
    try:
        os.sched_setaffinity(0, {0, 1, 2, 3, 4, 5})
    except OSError as e:                                               # noqa: BLE001
        print(f"  could not pin foreground: {e}")
    MP.hog_self_test(out)
    ps = _ps(model, inputs)
    ok0, bad0 = _finite(step())

    def make_fn(phase):
        return GP.build_phases(args, model, inputs, step, [phase])[phase]
    rows = MP.measure_loads(out, make_fn, ["prefix"], args.seconds)
    cpu_rows = MP.measure_cpu_util(out, make_fn)
    prows = MP.measure_power(out, make_fn)
    ok1, bad1 = _finite(step())
    prov_commit = GP._provenance(args, ctx["gate"])["model_commit"]
    mi = ModelInputs(_model_name(args), prov_commit, sh_out(f"git -C {REPO} rev-parse HEAD"), PM, ps["P"],
                     ps["S"], n)
    MP.emit_rows(mi, rows, cpu_rows, prows, _style(ctx))
    n_mi = mi.write(out / "model_inputs.csv")
    for f_ in sorted((out / "raw").glob("latency_samples_*.csv")):
        compress_and_record(f_)
    write_summary(out, f"E11e - GR00T N1.7 memory pressure and CPU ({args.rung})", [
        ("Answer", table(rd(out / "mempressure.csv"),
                         ["load_label", "duty_pct_achieved_during", "bg_GBps_alone", "bg_GBps_during",
                          "median_ms", "p5_ms", "p95_ms", "n", "throughput_inf_per_s", "slowdown"])),
        ("CPU utilisation", table([r for r in rd(out / "cpu_util.csv") if not r["scope"].startswith("cpu")],
                                  ["phase", "scope", "pct", "median_pct", "n", "cores_busy"])),
        ("Power", table(rd(out / "power.csv"), ["phase", "rail", "mean_mW", "median_mW", "p5_mW", "p95_mW", "n"])),
        ("Conditions", f"rung `{args.rung}`, N = {n}, MAXN; foreground cores 0-5, hog cores 6-11; prefix = the "
                       f"whole backbone (vision inside), the same loop as E7. PROXY, NOT PIM (protocol §13.5)."),
        ("Reproduce", f"```\npython bench/groot_phases.py {' '.join(sys.argv[1:])}\n```"),
    ], [])
    _write_meta(out, _meta(args, ctx, "E11e (GR00T memory pressure)", {
        "prefix_tokens": ps["P"], "suffix_tokens": ps["S"], "n_warmup": 5,
        "n_timed": f"{args.seconds} s loop per condition", "measured_peak_GBps": MP.PEAK_GBPS,
        "foreground_cores": MP.FG_CORES, "background_cores": MP.BG_CORES, "hog_period_ms": 20,
        "hog_buffer_MiB": 256, "actions_finite_before_after": [ok0, ok1],
        "actions_nonfinite": bad0 + bad1, "model_inputs_rows": n_mi, "raw_files": [],
        "notes": ["PROXY, NOT PIM. Protocol §13.5: the result is the sensitivity of the GPU work to lost "
                  "bandwidth, not PIM interference.",
                  "Load sweep, CPU utilisation and power are bench/mempressure.py's functions, unchanged; the "
                  "phase callables are groot_phases.build_phases (E7's loops).",
                  "hog_self_test ran after the model was loaded (the LeRobot driver runs it before)."]}))
    print(f"\nwrote {out} ({n_mi} rows)", flush=True)
    return 0 if bad0 + bad1 == 0 else 3


# ----------------------------------------------------------------------------------------------- E11f pipelining

HOST_EXCLUDE = ("attention_fused", "feedforward")   # §13.2: attention products and large linears are not host segments


def _base_argv(drop=("--e11", "--outdir", "--seconds", "--steps")) -> list[str]:
    out, skip = [], False
    for a in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if a in drop:
            skip = True
            continue
        if any(a.startswith(d + "=") for d in drop):
            continue
        out.append(a)
    return out


def _host_sequence(args, ctx, ps):
    import segments_groot as SG                                        # noqa: PLC0415
    model, n = ctx["model"], ctx["n"]
    blocks = model.action_head.model.transformer_blocks
    dims = _dit_dims(model)
    SG._API[0] = args.api
    segs, notes = {}, []
    for bi in (0, 1):
        with SG.DiTBlockCapture(model, bi) as cap, torch.no_grad():
            _call(model, _bf(ps["bo"]), ps["ai"], _eager_head(model))
        if cap.args is None:
            raise SystemExit(f"DiT block {bi} not captured through the head's class get_action")
        t = SG.intermediates(cap)
        S = t["x"].shape[1]
        ehs = t["encoder_hidden_states"]
        T = ehs.shape[1] if ehs is not None else S
        segs[cap.kind] = [s for s in SG.build_segments(t, S, dims["D"], dims["I"], dims["H"], dims["HD"], T)
                          if s["segment"] not in HOST_EXCLUDE]
    kinds = ["cross_attn" if i % 2 == 0 else "self_attn" for i in range(len(blocks))]
    n_calls = n * sum(len(segs[k]) for k in kinds)
    if ctx["compiled_desc"]:
        notes.append("host-sequence segments were captured through the action head's CLASS get_action (eager "
                     "Python), because the compiled head replays as a CUDA graph and runs no block body; the modules "
                     "and shapes are the same")
    print(f"  host sequence: {len(blocks)} blocks x N={n}; per block "
          f"{ {k: [s['segment'] for s in v] for k, v in segs.items()} }", flush=True)
    pause = args.pause_ms / 1e3

    def run():
        for _ in range(n):
            for k in kinds:
                for sd in segs[k]:
                    sd["fn"]()
                    if pause:
                        time.sleep(pause)
        torch.cuda.synchronize()
    return run, n_calls * args.pause_ms, segs, notes


def pipelining(args, ctx) -> int:
    import phase_loops as PL                                           # noqa: PLC0415
    import pipelining as PIPE                                          # noqa: PLC0415
    GP = _gp()
    model, inputs, step, out, n = ctx["model"], ctx["inputs"], ctx["step"], ctx["out"], ctx["n"]
    (out / "nsys").mkdir(parents=True, exist_ok=True)
    mode = _mode_label(args, ctx)
    ps = _ps(model, inputs)
    ok0, bad0 = _finite(step())
    trows, crows, irows = [], [], []
    for phase in ("prefix", "ae", "full"):
        fn = GP.build_phases(args, model, inputs, step, [phase])[phase]
        r = PL.run_loop(fn, args.seconds, phase)
        r.pop("samples")
        trows.append({"mode": mode, "phase": phase, "effective_exec_style": _style(ctx), **r})
        print(f"  {phase:8s} {r['throughput_inf_per_s']:7.3f} inf/s  median {r['median_ms']:8.2f} ms", flush=True)

    seq_fn, sleep_ms, segs, cap_notes = _host_sequence(args, ctx, ps)
    prefix_fn = GP.build_phases(args, model, inputs, step, ["prefix"])["prefix"]
    cdir = out / "raw" / f"contender_{args.rung}"
    ready = cdir / "READY"
    if ready.exists():
        ready.unlink()
    cmd = ([sys.executable, str(GP_FILE)] + _base_argv()
           + ["--e11", "contender", "--outdir", str(cdir), "--seconds", str(args.conc_reps * 3 + 120)])
    s_hi, least, prio_ok = PIPE.measure_concurrency(out, mode, seq_fn, prefix_fn, cmd, 1800.0, sleep_ms,
                                                    args.conc_reps, crows, ready_file=ready)

    # --- GPU idle inside the denoising loop: eager only (a compiled head has no per-step NVTX boundary)
    if not ctx["compiled_desc"]:
        unprof = {}
        ah = model.action_head
        for nn in (4, 10):
            ah.num_inference_timesteps = nn
            fn = GP.build_phases(args, model, inputs, step, ["ae"])["ae"]
            r = PL.run_loop(fn, 20.0, f"ae_unprofiled_N{nn}")
            r.pop("samples")
            unprof[nn] = r["median_ms"]
            print(f"  unprofiled ae loop N={nn}: {r['median_ms']:.2f} ms", flush=True)
        ah.num_inference_timesteps = n
        for nn in (4, 10):
            tag = f"{mode}_N{nn}"
            c = (["bash", str(REPO / "bench" / "run_nsys.sh"), str(out), tag, "--", sys.executable, str(GP_FILE)]
                 + _base_argv() + ["--steps", str(nn), "--e11", "nsys_ae", "--outdir", str(out / "raw" / f"nsys_ae_{tag}")])
            rr = subprocess.run(c, capture_output=True, text=True, timeout=3600)
            (out / "raw" / f"nsys_{tag}_driver.log").write_text(rr.stdout + rr.stderr)
            g = PIPE.gpu_idle_from_nsys(out / "nsys", tag)
            if g:
                u = unprof[nn]
                g.update({"ae_total_ms_unprofiled": round(u, 4), "profiling_inflation": round(g["ae_total_ms"] / u, 4),
                          "gpu_idle_ms_corrected": round(u - g["kernel_busy_ms"], 4),
                          "gpu_idle_frac_corrected": round((u - g["kernel_busy_ms"]) / u, 4)})
                irows.append({"mode": mode, "N": nn, **g})
                print(f"  nsys {tag}: busy {g['kernel_busy_ms']:.2f} ms, corrected idle "
                      f"{g['gpu_idle_frac_corrected']}", flush=True)
            else:
                irows.append({"mode": mode, "N": nn, "ae_total_ms_unprofiled": round(unprof[nn], 4),
                              "note": "nsys exports missing or no ae_step NVTX ranges; see raw/nsys_*_driver.log"})
    else:
        irows.append({"mode": mode, "N": n, "note": "not measured: the compiled action head replays as one CUDA "
                                                    "graph, so there are no per-step NVTX ranges (as for the "
                                                    "LeRobot compiled captures)"})
    ok1, bad1 = _finite(step())
    for name, rows_ in (("throughput.csv", trows), ("concurrency.csv", crows), ("gpu_idle.csv", irows)):
        keys = sorted({k for r in rows_ for k in r})
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows([{k: r.get(k, "") for k in keys} for r in rows_])
    prov_commit = GP._provenance(args, ctx["gate"])["model_commit"]
    n_mi = PIPE.emit_model_inputs(out, trows, crows, irows, n, sh_out(f"git -C {REPO} rev-parse HEAD"), None,
                                  {"P": ps["P"], "S": ps["S"]}, model_name=_model_name(args),
                                  model_commit=prov_commit)
    write_summary(out, f"E11f - GR00T N1.7 pipelining ({args.rung})", [
        ("Throughput", table(trows, ["phase", "mode", "median_ms", "p5_ms", "p95_ms", "n", "throughput_inf_per_s"])),
        ("Concurrency (host sequence vs the prefix loop)",
         table(crows, ["sharing", "condition", "median_ms", "p95_ms", "max_ms", "n", "slowdown",
                       "slowdown_non_sleep"])),
        ("GPU idle inside the denoising loop", table(irows, ["mode", "N", "ae_total_ms_unprofiled", "kernel_busy_ms",
                                                              "gpu_idle_frac_corrected", "note"])),
        ("Conditions", f"rung `{args.rung}`, N = {n}; host sequence = {len(model.action_head.model.transformer_blocks)}"
                       f" DiT blocks x N of the host segments (all E11a segments except {HOST_EXCLUDE}), "
                       f"{args.pause_ms} ms pause between segments; two streams from one thread and a second process "
                       f"running the prefix loop (§13.7)."),
        ("Reproduce", f"```\npython bench/groot_phases.py {' '.join(sys.argv[1:])}\n```"),
    ], [r["note"] for r in irows if r.get("note")])
    _write_meta(out, _meta(args, ctx, "E11f (GR00T pipelining)", {
        "prefix_tokens": ps["P"], "suffix_tokens": ps["S"], "n_warmup": 5,
        "n_timed": f"{args.seconds} s loops; {args.conc_reps} concurrency reps",
        "nsys_version": sh_out("nsys --version"),
        "host_segments": {k: [s["segment"] for s in v] for k, v in segs.items()},
        "host_segments_excluded": list(HOST_EXCLUDE), "sleep_ms_per_sequence": sleep_ms,
        "contender_stream_priority": s_hi.priority, "contender_stream_is_elevated": s_hi.priority != least,
        "high_priority_stream_accepted": prio_ok, "contender_command": " ".join(cmd),
        "actions_finite_before_after": [ok0, ok1], "actions_nonfinite": bad0 + bad1,
        "model_inputs_rows": n_mi, "raw_files": [],
        "notes": cap_notes + [
            "Concurrency and gpu_idle use bench/pipelining.py's functions unchanged (measure_concurrency, "
            "gpu_idle_from_nsys, emit_model_inputs).",
            "gpu_idle captures run in a child process (groot_phases.py --e11 nsys_ae) while this process holds its "
            "model idle; NVTX ae_step[i] ranges wrap each DiT forward."]}))
    print(f"\nwrote {out} ({n_mi} rows)", flush=True)
    return 0 if bad0 + bad1 == 0 else 3


def contender(args, ctx) -> int:
    """Second process for E11f: the prefix loop, started after load (+ compile) and announced through READY."""
    import phase_loops as PL                                           # noqa: PLC0415
    fn = _gp().build_phases(args, ctx["model"], ctx["inputs"], ctx["step"], ["prefix"])["prefix"]
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    (ctx["out"] / "READY").write_text(time.strftime("%H:%M:%S"))
    r = PL.run_loop(fn, args.seconds, "contender_prefix", warmup=0)
    r.pop("samples")
    print(json.dumps(r), flush=True)
    return 0


def nsys_ae(args, ctx) -> int:
    """Child of E11f: NVTX ae_step[i] around each DiT forward, 3 warm inferences, then 2 inside cudaProfilerStart/Stop."""
    if ctx["compiled_desc"]:
        raise SystemExit("nsys_ae is eager-only")
    model, step = ctx["model"], ctx["step"]
    dit = model.action_head.model
    orig = dit.forward
    k = {"i": 0}

    def f(*a, **kw):
        torch.cuda.nvtx.range_push(f"ae_step[{k['i']}]")
        k["i"] += 1
        try:
            return orig(*a, **kw)
        finally:
            torch.cuda.nvtx.range_pop()
    dit.forward = f
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    for _ in range(2):
        step()
        torch.cuda.synchronize()
    cudart.cudaProfilerStop()
    dit.forward = orig
    print(f"  nsys_ae: captured 2 inferences, {k['i']} DiT calls in total", flush=True)
    return 0


DISPATCH = {"handover": handover, "prefixsweep": prefixsweep, "mempressure": mempressure,
            "pipelining": pipelining, "contender": contender, "nsys_ae": nsys_ae}


def run(args, ctx) -> int:
    if args.api != "new":
        raise SystemExit("groot_e11 covers the N1.7 API (T012); N1.5 / N1.6 are out of T012's scope")
    return DISPATCH[args.e11](args, ctx)
