"""T010 / E11d — how the GPU prefix scales with its input (protocol §13, plan part 4).

The hybrid model keeps vision and the VLM prefix on the GPU and treats them as a fixed cost
(110 ms + 258 ms from the characterization paper, at a fixed P = 816). E11d measures how that
cost actually moves with the number of camera images and the prompt length.

What the loaded pi0 does with its input, measured not assumed:

  * `config.input_features` declares **three** image slots — `image`, `image2` and
    `empty_camera_0` — and `config.empty_cameras = 1`. A camera that is not supplied is NOT
    dropped: `PI0Policy._preprocess_images` synthesises a -1-filled tensor with a zero mask,
    and it is still encoded into 256 prefix tokens.
  * `config.tokenizer_max_length = 48`, so the prompt is padded to 48 tokens whatever it says.

Together those give a **constant processed P = 3*256 + 48 = 816** across the whole sweep, while
`P_valid` moves. Protocol §13 calls that out explicitly — "models that pad the prompt to a fixed
length will show a constant processed P — report that, it is a result" — so both are reported.

    python bench/prefix_sweep.py --outdir-prefix results/<date>_E11d_pi0
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_registry as MR  # noqa: E402
import phase_builders as PB  # noqa: E402
import pi0_common as P  # noqa: E402
import regions as R  # noqa: E402
from e11_summaries import e11d  # noqa: E402
from common import (REPO, collect_raw, sh_out, software_versions, summarize,  # noqa: E402
                    write_json)
from host_segments import ModelInputs  # noqa: E402

IMG_KEYS = ["observation.images.image", "observation.images.image2",
            "observation.images.empty_camera_0"]
PROMPTS = {
    "short": "pick up the mug",
    "typical": None,                       # the fixture's own instruction
    "long": ("put the white mug on the left plate and put the yellow and white mug on the "
             "right plate then return the gripper to its home position and wait for the "
             "next instruction from the operator"),
}


def make_fixture(base: dict, n_images: int, prompt: str | None) -> dict:
    """A fixture with `n_images` supplied camera slots and the given prompt.

    The dataset carries two cameras, so the third slot is filled with a copy of the second —
    what varies is how many slots arrive with a valid mask, which is what changes P_valid.
    """
    fx = copy.deepcopy(base)
    imgs = dict(fx["images"])
    keys = list(imgs)
    want = []
    for i in range(n_images):
        src = keys[i] if i < len(keys) else keys[-1]
        want.append((IMG_KEYS[i], dict(imgs[src])))
    fx["images"] = dict(want)
    fx["_n_images_supplied"] = n_images
    fx["_third_image_is_duplicate"] = n_images > len(keys)
    if prompt is not None:
        fx["task"] = prompt
    return fx


def run_point(policy, fx, steps, warmup, timed, instrument, spec):
    batch, _, _ = P.build_batch(policy, fx, checkpoint=spec.checkpoint,
                                revision=spec.revision)
    tok = P.token_counts(policy, batch, spec)
    rec = R.Recorder() if instrument else None
    undo = R.instrument(policy, rec, spec.container) if instrument else None
    try:
        for _ in range(warmup):
            policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        e2e, per_region = [], {}
        for _ in range(timed):
            if rec is not None:
                rec.reset()
                rec.start()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            e2e.append((time.perf_counter() - t0) * 1e3)
            if rec is not None:
                rec.stop()
                for name, v in rec.results_ms().items():
                    per_region.setdefault(name, []).append(v)
        peak = torch.cuda.max_memory_allocated()
    finally:
        if undo:
            undo()
    return tok, e2e, per_region, peak


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    ss_res = sum(r * r for r in res)
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, (1 - ss_res / ss_tot if ss_tot else 1.0), res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pi0",
                    help="pi0 | pi05 | smolvla. T011 extends E11 beyond pi0; the expert layout and the checkpoint-specific processors come from bench/model_registry.py.")
    ap.add_argument("--outdir-prefix", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timed", type=int, default=30)
    ap.add_argument("--modes", default="eager,reduce-overhead")
    args = ap.parse_args()
    spec = MR.get(args.model)
    gate = P.gate()
    print("gate:", json.dumps(gate), flush=True)
    base = P.load_fixture()

    for mode in [m for m in args.modes.split(",") if m]:
        out = Path(f"{args.outdir_prefix}_{mode}_prefixsweep")
        (out / "raw").mkdir(parents=True, exist_ok=True)
        cm = None if mode == "eager" else mode
        t_load = time.perf_counter()
        policy, cfg, shipped = MR.load_policy(spec, n=args.steps, compile_mode=cm)
        real = P.assert_real_weights(policy, spec=spec)
        print(f"\n=== {mode} (load {time.perf_counter()-t_load:.1f} s) ===", flush=True)
        rows, samples, compile_s = [], [], {}
        for n_img in (1, 2, 3):
            for plabel, ptext in PROMPTS.items():
                fx = make_fixture(base, n_img, ptext)
                label = f"{n_img}img_{plabel}"
                # warm each shape until it stops recompiling; the first call carries compile
                t0 = time.perf_counter()
                b0, _, _ = P.build_batch(policy, fx, checkpoint=spec.checkpoint,
                                revision=spec.revision)
                policy.predict_action_chunk(b0)
                torch.cuda.synchronize()
                compile_s[label] = round(time.perf_counter() - t0, 3)
                tok, e2e, reg, peak = run_point(
                    policy, fx, args.steps, args.warmup, args.timed, instrument=(cm is None))
                s = summarize(e2e)
                base_row = {
                    "mode": mode, "n_images": n_img, "prompt_label": plabel,
                    "prompt_tokens": tok["language_tokens"],
                    "prompt_tokens_valid": tok["language_tokens_valid"],
                    "image_tokens": tok["prefix_tokens_P"] - tok["language_tokens"],
                    "P_processed": tok["prefix_tokens_P"],
                    "P_valid": tok["prefix_tokens_valid"],
                    # S carried per row so the summary never has to fall back on pi0's 51
                    "S_processed": tok["suffix_tokens_S"],
                    "S_valid": tok["suffix_tokens_valid"],
                    "n_image_slots_encoded": tok["n_images"],
                    "peak_mem_bytes": peak, "first_call_s": compile_s[label]}
                for region, ms in list(reg.items()) + [("e2e", None)]:
                    if region == "e2e":
                        st = s
                        vals = e2e
                    else:
                        vals = ms if isinstance(ms, list) else [ms]
                        st = summarize(vals)
                    rows.append({**base_row, "region": region,
                                 "median_ms": round(st["median_ms"], 5),
                                 "p5_ms": round(st["p5_ms"], 5),
                                 "p95_ms": round(st["p95_ms"], 5), "n": st["n"]})
                for i, v in enumerate(e2e):
                    samples.append({"mode": mode, "point": label, "region": "e2e",
                                    "iter": i, "ms": v})
                print(f"  {label:16s} P={tok['prefix_tokens_P']} valid={tok['prefix_tokens_valid']}"
                      f" prompt={tok['language_tokens_valid']}/{tok['language_tokens']}"
                      f"  e2e {s['median_ms']:8.2f} ms  peak {peak/1e6:6.0f} MB", flush=True)

        with (out / "prefix_sweep.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with (out / "latency_samples.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["mode", "point", "region", "iter", "ms"])
            w.writeheader()
            w.writerows(samples)

        # --- fits
        fits = []
        for key, xname, xf, regions_ in (
                ("vision_vs_images", "n_images", lambda r: r["n_images"], ("vision",)),
                ("vlm_prefix_vs_P", "P_processed", lambda r: r["P_processed"], ("vlm_prefix",)),
                ("ae_step_vs_P", "P_processed", lambda r: r["P_processed"],
                 ("ae_step", "ae_total", "e2e"))):
            for region in regions_:
                pts = [(xf(r), r["median_ms"]) for r in rows if r["region"] == region]
                if len(pts) < 2:
                    continue
                agg = {}
                for x, y in pts:
                    agg.setdefault(x, []).append(y)
                xs = sorted(agg)
                ys = [sorted(agg[x])[len(agg[x]) // 2] for x in xs]
                fr = linfit(xs, ys)
                if fr is None:
                    fits.append({"fit": key, "region": region, "x": xname, "fit_a_ms": "",
                                 "fit_b": "", "unit_b": "", "fit_r2": "", "n_points": len(xs),
                                 "residuals": "",
                                 "note": f"x is constant at {xs[0]} across the sweep; no slope "
                                         f"can be fitted, which is itself the result"})
                    print(f"  fit {key:18s} {region:11s} x constant at {xs[0]} -> no slope")
                    continue
                a, b, r2, res = fr
                unit = "ms_per_image" if xname == "n_images" else "ms_per_token"
                fits.append({"fit": key, "region": region, "x": xname,
                             "fit_a_ms": round(a, 5), "fit_b": round(b, 8), "unit_b": unit,
                             "fit_r2": round(r2, 6), "n_points": len(xs),
                             "residuals": ";".join(f"{v:.4f}" for v in res), "note": ""})
                print(f"  fit {key:18s} {region:11s} a={a:8.3f} b={b:.5f} {unit} r2={r2:.4f}")
        with (out / "prefix_fit.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fits[0].keys()))
            w.writeheader()
            w.writerows(fits)

        # S from the run's own token counts, not the literal 51, which is pi0's.
        s_tokens = rows[0].get("S_processed") or rows[0].get("suffix_tokens_S") or ""
        mi = ModelInputs(spec.name, spec.revision or "", sh_out(f"git -C {REPO} rev-parse HEAD"),
                         "MAXN+jetson_clocks", rows[0]["P_processed"], s_tokens,
                         args.steps)
        qmap = {"vision": "t_vision", "vlm_prefix": "t_vlm_prefix", "ae_step": "t_ae_step",
                "ae_total": "t_ae_total", "e2e": "t_e2e", "preprocess": "t_preprocess",
                "postprocess": "t_postprocess"}
        style = "eager" if cm is None else "compiled"
        for r in rows:
            q = qmap.get(r["region"])
            if not q:
                continue
            var = (f"n_images={r['n_images']};prompt={r['prompt_label']};"
                   f"P_valid={r['P_valid']}")
            if cm:
                var += f";compile_mode={cm}"
            for stat, col in (("median", "median_ms"), ("p5", "p5_ms"), ("p95", "p95_ms")):
                mi.add(q, r[col], "ms", stat, r["n"], "prefix_sweep.csv", variant=var,
                       exec_style=style, timing="wall_isolated")
        for f_ in fits:
            if f_["fit_a_ms"] == "":
                continue
            v = f"region={f_['region']}" + (f";compile_mode={cm}" if cm else "")
            mi.add("prefix_latency_fit", f_["fit_a_ms"], "ms", "fit_a", f_["n_points"],
                   "prefix_fit.csv", key=f_["fit"], variant=v, exec_style=style)
            mi.add("prefix_latency_fit", f_["fit_b"], f_["unit_b"], "fit_b", f_["n_points"],
                   "prefix_fit.csv", key=f_["fit"], variant=v, exec_style=style)
            mi.add("prefix_latency_fit", f_["fit_r2"], "ratio", "fit_r2", f_["n_points"],
                   "prefix_fit.csv", key=f_["fit"], variant=v, exec_style=style)
        n_mi = mi.write(out / "model_inputs.csv")

        sw = software_versions()
        write_json(out / "meta.json", {
            "experiment": "E11d", "task": "T010",
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": spec.name, "model_repo": spec.checkpoint,
            "model_commit": spec.revision, "model_revision": spec.revision,
            "checkpoint": spec.checkpoint, "precision": "bfloat16",
            "compile_mode": cm, "batch": 1, "num_denoise_steps": args.steps,
            "action_horizon": getattr(cfg, "chunk_size", None),
            "prefix_tokens": rows[0]["P_processed"],
            "suffix_tokens": rows[0].get("S_processed", ""),
            "n_warmup": args.warmup, "n_timed": args.timed,
            "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
            "gpu_hz": gate["gpu_cur_freq_hz"], "gate": gate,
            "emc_hz": int(sh_out("sudo -n sh -c 'cat /sys/kernel/debug/bpmp/debug/clk/emc/rate'") or 0),
            "jetpack": "6.2.3+b81", "l4t": "R36.5.0",
            "cuda": sw.get("torch_cuda"), "torch": sw.get("torch"),
            "torchvision": sw.get("torchvision"), "triton": sw.get("triton"),
            "python": sw.get("python"), "venv": sw.get("executable"),
            "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
            "bench_files_sha256": sw.get("bench_files_sha256"),
            "weights_verified": real, "shipped_config": shipped,
            "command": f"python bench/prefix_sweep.py --outdir-prefix {args.outdir_prefix}",
            "first_call_seconds_per_shape": compile_s,
            "tokenizer_max_length": getattr(cfg, "tokenizer_max_length", None),
            "empty_cameras": getattr(cfg, "empty_cameras", None),
            "image_input_features": IMG_KEYS,
            "throttled": False,
            "throttled_evidence": "GPU clock pinned min==max, recorded in `gate`.",
            "model_inputs_rows": n_mi, "raw_files": [],
            "notes": [
                "A camera that is not supplied is NOT dropped: config.empty_cameras=1 and "
                "_preprocess_images synthesises a -1-filled tensor with a zero mask, which is "
                "still encoded into 256 prefix tokens. The prompt is padded to "
                "tokenizer_max_length=48. So P_processed is CONSTANT at 816 across all nine "
                "points while P_valid moves — protocol §13 asks for this to be reported as a "
                "result, and it is why the vlm_prefix-vs-P and ae_step-vs-P fits have no "
                "slope to report.",
                "The dataset supplies two cameras, so the third slot of the 3-image points "
                "carries a copy of the second; what varies is how many slots arrive with a "
                "valid mask. Recorded per row as _third_image_is_duplicate.",
                "Regions are instrumented in eager only: bench/regions.py's context managers "
                "force a graph break under torch.compile (see bench/run_e4.py), so the "
                "compiled sweep reports t_e2e only.",
            ],
        })
        meta = json.loads((out / "meta.json").read_text())
        e11d(out)
        meta["raw_files"] = collect_raw(out, meta["command"])
        write_json(out / "meta.json", meta)
        print(f"wrote {out} ({n_mi} rows)", flush=True)
        del policy
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
