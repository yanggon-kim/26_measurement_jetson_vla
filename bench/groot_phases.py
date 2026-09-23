"""GR00T phase split: backbone vs action head, at the model's own N.

GR00T cannot use bench/paper_repro.py. It lives in ~/venvs/groot (torch 2.10.0) while the
LeRobot harness lives in ~/venvs/pi0 (torch 2.8.0), and paper_repro imports lerobot. This
is the same measurement — same regions, same CSV schema, same warm-up and timing rules —
without the import.

## The two phases, read from the source (gr00t_n1d7.py:603)

    backbone_inputs, action_inputs = self.prepare_input(inputs)
    backbone_outputs = self.backbone(backbone_inputs)                    # <- vlm_prefix
    action_outputs  = self.action_head.get_action(backbone_outputs, ...) # <- ae_total

and the denoising loop is inside the action head,
`for t in range(self.num_inference_timesteps)` (:366+).

## Why vision is not split out here, stated rather than quietly omitted

In the LeRobot family `embed_image` is a distinct method, so vision separates cleanly from
the VLM. GR00T's backbone is a whole VLM invoked as one unit — Cosmos-Reason2-2B
(Qwen3-VL) for N1.7, Eagle for N1.6 and N1.5 — and isolating its vision encoder means
reaching into internals that differ across exactly the versions we are comparing. For the
action-head-share question, backbone vs action_head is the split that matters and it is
exact. `vision` is reported as "inside vlm_prefix" and the comparison table says so.

    ~/venvs/groot/bin/python bench/groot_phases.py --outdir results/<date>_E4d_groot_n17 \
        --checkpoint nvidia/GR00T-N1.7-3B
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch


def summarize(xs):
    s = sorted(xs)
    n = len(s)
    q = lambda p: s[0] if n == 1 else s[int(p*(n-1))] + (
        s[min(int(p*(n-1))+1, n-1)] - s[int(p*(n-1))]) * (p*(n-1) - int(p*(n-1)))
    return {"median_ms": statistics.median(s), "mean_ms": statistics.fmean(s),
            "p5_ms": q(0.05), "p95_ms": q(0.95), "min_ms": s[0], "max_ms": s[-1], "n": n}


def gate() -> dict:
    def rd(p, d="?"):
        try:
            return Path(p).read_text().strip()
        except OSError:
            return d
    return {"gpu_cur_freq_hz": rd("/sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq"),
            "gpu_min_freq_hz": rd("/sys/devices/platform/17000000.gpu/devfreq_dev/min_freq"),
            "gpu_max_freq_hz": rd("/sys/devices/platform/17000000.gpu/devfreq_dev/max_freq"),
            "cpu_online": rd("/sys/devices/system/cpu/online"),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-",
            "capability": str(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available() else "-"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--checkpoint", default="nvidia/GR00T-N1.7-3B")
    ap.add_argument("--version", default="n17")
    ap.add_argument("--api", choices=("new", "old"), default="new",
                    help="'old' is N1.5's Gr00tPolicy signature and flat obs keys")
    ap.add_argument("--data-config", default="oxe_droid",
                    help="N1.5 only: key into gr00t.experiment.data_config.DATA_CONFIG_MAP")
    ap.add_argument("--ncu-region", default=None,
                    help="C1: install NVTX ranges and profile ONE inference, then exit")
    ap.add_argument("--measure-s", action="store_true",
                    help="C3: read the action head's token count off a real inference")
    ap.add_argument("--compat-shims", action="store_true",
                    help="re-export symbols the older GR00T remote code expects")
    ap.add_argument("--gr00t-root", default=None,
                    help="Isaac-GR00T checkout to import from, for N1.6 / N1.5")
    ap.add_argument("--embodiment-tag", default="oxe_droid_relative_eef_relative_joint",
                    help="must be one the checkpoint ships; it lists them on a bad value")
    ap.add_argument("--steps", type=int, default=None, help="default: the config's own")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--tf32", choices=("on", "off"), default="on")
    ap.add_argument("--head-mode", default=None,
                    help="torch.compile mode for action_head.get_action "
                         "(max-autotune-no-cudagraphs = compile only, "
                         "reduce-overhead = compile + CUDA graphs)")
    ap.add_argument("--backbone-mode", default=None,
                    help="torch.compile mode for backbone.forward (whole VLM at once)")
    ap.add_argument("--vision-mode", default=None,
                    help="torch.compile mode for backbone.model.visual.forward")
    ap.add_argument("--llm-mode", default=None,
                    help="torch.compile mode for backbone.model.language_model.forward")
    ap.add_argument("--rung", default="eager",
                    help="name for this configuration; one row file per rung")
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    g = gate()
    print("gate:", json.dumps(g), flush=True)
    if g["gpu_cur_freq_hz"] != g["gpu_max_freq_hz"]:
        raise SystemExit(f"clocks not locked: {g}; run sudo jetson_clocks first")
    torch.set_float32_matmul_precision("high" if args.tf32 == "on" else "highest")

    compat = []
    if args.compat_shims:
        # N1.6's Eagle backbone is remote code written against an older transformers, and
        # this environment has 4.57.6 because N1.7's Qwen3-VL requires >= 4.57. The one
        # symbol it misses moved rather than disappeared:
        #     ImportError: cannot import name 'VideoInput' from 'transformers.image_utils'
        # transformers.video_utils.VideoInput is the same object, so it is re-exported
        # under the old name. This is an import-time alias for a type annotation; it does
        # not change any computation, and it is recorded in meta.json as compat_shims so a
        # reader knows the environment was adjusted.
        import transformers.image_utils as IU
        if not hasattr(IU, "VideoInput"):
            from transformers.video_utils import VideoInput
            IU.VideoInput = VideoInput
            compat.append("transformers.image_utils.VideoInput := "
                          "transformers.video_utils.VideoInput")

    if args.gr00t_root:
        # N1.6 and N1.5 live in worktrees pinned to the commits that still contain their
        # model modules (env/setup_groot_n1d5_n1d6.md). Putting the chosen checkout first
        # on sys.path is what selects the version; the measurement code is identical.
        sys.path.insert(0, args.gr00t_root)
    if args.api == "new":
        from gr00t.policy.gr00t_policy import Gr00tPolicy            # noqa: E402
    from huggingface_hub import snapshot_download                     # noqa: E402

    ckpt_dir = (args.checkpoint if Path(args.checkpoint).is_dir()
                else snapshot_download(args.checkpoint))
    print(f"loading {args.checkpoint} tag={args.embodiment_tag} ...", flush=True)
    # Gr00tPolicy is the supported inference path: it loads the model in bf16 on the
    # device AND the checkpoint's own processor, so the input pipeline measured here is
    # the shipped one rather than a reimplementation.
    # N1.7's Gr00tPolicy resolves a string tag to the enum itself; N1.6's does not and
    # dies on "'str' object has no attribute 'value'". Resolve it here so both versions
    # take the same argument.
    if args.api == "old":
        policy = _load_n15(args, ckpt_dir)
    else:
        from gr00t.data.embodiment_tags import EmbodimentTag              # noqa: E402
        tag = args.embodiment_tag
        if isinstance(tag, str):
            if hasattr(EmbodimentTag, "resolve"):
                tag = EmbodimentTag.resolve(tag)
            else:
                try:
                    tag = EmbodimentTag(tag)
                except ValueError:
                    tag = EmbodimentTag[tag.upper()]
        policy = Gr00tPolicy(embodiment_tag=tag, model_path=ckpt_dir, device="cuda")
    model = policy.model
    n = args.steps or int(getattr(model.action_head.config, "num_inference_timesteps", 4))
    model.action_head.num_inference_timesteps = n
    print(f"N (num_inference_timesteps) = {n}", flush=True)

    # timing hooks around exactly the two phases, installed on the instance
    spans: dict[str, list] = {"vision": [], "vlm_backbone": [], "ae_total": []}

    def timed(obj, attr, label):
        orig = getattr(obj, attr)

        def wrapper(*a, **k):
            s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            s.record()
            r = orig(*a, **k)
            e.record()
            spans[label].append((s, e))
            return r
        setattr(obj, attr, wrapper)
        return lambda: setattr(obj, attr, orig)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from graph_clone import clone_graph_outputs                   # noqa: E402

    def compile_and_clone(obj, attr, mode):
        """torch.compile, and clone the result when CUDA graphs are in play.

        reduce-overhead reuses one static output buffer per callable, so a callable
        invoked more than once per inference loses its earlier results. GR00T's
        vlm_graphs died with exactly that ("accessing tensor output of CUDAGraphs that
        has been overwritten by a subsequent run") -- the same defect that cost three
        separate diagnoses on the LeRobot side.
        """
        c = torch.compile(getattr(obj, attr), mode=mode)
        if "reduce-overhead" not in mode:
            setattr(obj, attr, c)
            return ""
        def wrapper(*a, _c=c, **k):
            return clone_graph_outputs(_c(*a, **k))
        setattr(obj, attr, wrapper)
        return "+clone"

    compiled_desc, compile_s = [], 0.0
    if args.head_mode:
        t0 = time.time()
        cl = compile_and_clone(model.action_head, "get_action", args.head_mode)
        compile_s += time.time() - t0
        compiled_desc.append(f"action_head.get_action@{args.head_mode}{cl}")
    if args.vision_mode:
        # Expected to fail on this checkpoint, and the reason is worth recording rather
        # than hiding: Qwen3-VL's vision encoder computes
        #     max_seqlen_q = int(grid_thw[:, 1:].max().item())
        # and hands it to flash_attn_varlen_func, whose schema declares SymInt. Dynamo
        # graph-breaks at .item() and then passes a 0-dim FakeTensor, so the custom op
        # raises "Expected a value of type 'int' for argument 'max_seqlen_q'".
        t0 = time.time()
        cl = compile_and_clone(_vision_module(model), "forward", args.vision_mode)
        compile_s += time.time() - t0
        compiled_desc.append(f"backbone.model.visual.forward@{args.vision_mode}{cl}")
    if args.llm_mode:
        t0 = time.time()
        cl = compile_and_clone(_language_module(model), "forward", args.llm_mode)
        compile_s += time.time() - t0
        compiled_desc.append(
            f"backbone.model.language_model.forward@{args.llm_mode}{cl}")
    if args.backbone_mode:
        t0 = time.time()
        cl = compile_and_clone(model.backbone, "forward", args.backbone_mode)
        compile_s += time.time() - t0
        compiled_desc.append(f"backbone.forward@{args.backbone_mode}{cl}")
    if compiled_desc:
        print(f"compiled {', '.join(compiled_desc)} in {compile_s:.1f} s", flush=True)

    # NOT timed(model, "backbone", ...): that replaces the backbone MODULE with a plain
    # function, and prepare_input calls self.backbone.prepare_input (gr00t_n1d7.py:570),
    # which would raise AttributeError. Wrapping .forward leaves the module intact.
    # Instrumenting AFTER compiling keeps both region boundaries outside the compiled
    # callables, the same order opt_ladder.py uses for the LeRobot models.
    undo = [timed(model.backbone, "forward", "vlm_backbone"),
            timed(model.action_head, "get_action", "ae_total")]
    # GR00T's backbone is Qwen3VLForConditionalGeneration, which has the SAME two parts the
    # LeRobot models expose separately: .visual (the image encoder, analogous to
    # embed_image) and .language_model (the prefix, analogous to pg.forward). Timing
    # .visual as a nested region turns GR00T's two-way split into the three-way split every
    # other model in this study gets, so vision is no longer "inside vlm_prefix" here.
    vis = _vision_module(model)
    if vis is not None:
        undo.append(timed(vis, "forward", "vision"))
    else:
        raise SystemExit(
            "the backbone's vision module was not found under any known attribute "
            "(model/eagle_model x visual/vision_model). Refusing to run: the vision region "
            "would silently report 0.00 ms and the three-way split would be wrong.")

    if args.api == "old":
        obs, shapes = _build_obs_n15(policy, ckpt_dir)
    else:
        obs, shapes = build_observation(policy, ckpt_dir)
    print(f"observation shapes: {json.dumps(shapes, default=str)}", flush=True)

    # The CPU input pipeline is a real phase of GR00T inference and NVIDIA's own
    # benchmark_inference.py reports it as one, so it is timed here rather than hidden
    # by building the inputs once. The GPU phase split below then times the model alone.
    prep = (lambda: policy.apply_transforms(
        {k: (v if isinstance(v, list) else __import__("numpy").expand_dims(v, 0))
         for k, v in obs.items()})) if args.api == "old" \
        else (lambda: build_inputs(policy, obs))
    prep()
    dp = []
    for _ in range(min(args.timed, 20)):
        t0 = time.perf_counter()
        prep()
        dp.append((time.perf_counter() - t0) * 1e3)
    print(f"data_processing (CPU): {statistics.median(dp):.2f} ms", flush=True)

    inputs = prep()

    if args.ncu_region:
        # C1 needs ncu to see ONE instance of one phase. Same shape as
        # bench/ncu_target.py does for the LeRobot models: NVTX ranges on the three wrap
        # points, a short warm-up, then exactly one inference between
        # cudaProfilerStart/Stop, so --profile-from-start off + --nvtx-include isolates
        # the requested phase.
        #
        # Range names are bracket-free on purpose: ncu's --nvtx-include expression
        # language gives [ and ] special meaning, and two 250 s runs profiled ZERO kernels
        # before that was understood.
        import torch.cuda.nvtx as nvtx

        def nvtx_wrap(obj, attr, label):
            orig = getattr(obj, attr)

            def w(*a, **k):
                nvtx.range_push(label)
                try:
                    return orig(*a, **k)
                finally:
                    nvtx.range_pop()
            setattr(obj, attr, w)

        vis_m = _vision_module(model)
        if vis_m is not None:
            nvtx_wrap(vis_m, "forward", "vision")
        nvtx_wrap(model.backbone, "forward", "vlm_prefix")
        nvtx_wrap(model.action_head, "get_action", "ae_total")

        for _ in range(max(2, args.warmup // 4)):
            step()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        step()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        print(f"profiled one inference with NVTX range {args.ncu_region!r}", flush=True)
        return 0

    if args.measure_s:
        # C3 needs S, the number of tokens the action head's GEMMs actually run on. The
        # config's action_horizon (N1.7 40, N1.6 50, N1.5 16) is a strong prior, not the
        # answer: pi0's chunk_size is 50 and its measured S is 51, because a state token is
        # prepended. So read it off the tensor the first DiT block is actually handed.
        blocks = model.action_head.model.transformer_blocks
        seen = {}

        def grab(mod, inp):
            if "S" not in seen and inp and torch.is_tensor(inp[0]):
                seen["shape"] = tuple(inp[0].shape)
                seen["S"] = int(inp[0].shape[-2]) if inp[0].dim() >= 2 else None

        h = blocks[0].register_forward_pre_hook(lambda m, i: grab(m, i))
        step()
        h.remove()
        cfg_ah = getattr(model.action_head.config, "action_horizon", None)
        rec = {"model": f"groot_{args.version}", "embodiment_tag": args.embodiment_tag,
               "suffix_tokens_S": seen.get("S"),
               "block0_input_shape": list(seen.get("shape", ())),
               "config_action_horizon": cfg_ah,
               "n_transformer_blocks": len(blocks),
               "source": "measured: forward pre-hook on action_head.model."
                         "transformer_blocks[0] during one real inference"}
        (out / "suffix_tokens.json").parent.mkdir(parents=True, exist_ok=True)
        (out / "suffix_tokens.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec, indent=1), flush=True)
        return 0

    # A rung that fails must leave a ROW saying so. GR00T's vlm_compile and vlm_graphs
    # died with no row at all, and the watchdog audit then reported the model as
    # "4 measured, 0 failed" — an audit cannot see a rung that left no trace.
    def _fail_row(exc):
        rowdir = out / "raw" / "rows"
        rowdir.mkdir(parents=True, exist_ok=True)
        (rowdir / f"{args.rung}.json").write_text(json.dumps({
            "rung": args.rung, "N": n, "status": "failed",
            "compiled": ";".join(compiled_desc), "compile_s": round(compile_s, 3),
            "e2e_median_ms": "", "vision_ms": "", "vlm_backbone_ms": "",
            "vlm_prefix_ms": "", "ae_total_ms": "",
            "data_processing_cpu_ms": round(statistics.median(dp), 4),
            "matmul_precision": "high" if args.tf32 == "on" else "highest",
            "error": f"{type(exc).__name__}: {str(exc)[:300]}", "n": 0,
        }, indent=1))
        (out / "raw" / f"traceback_{args.rung}.txt").write_text(traceback.format_exc())
        print(f"  NOT MEASURED: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
        print(f"  row + traceback written for {args.rung}", flush=True)

    def step():
        """One inference, with the CUDA-graph generation advanced first.

        With TWO cudagraph-managed callables in the same frame (action_head.get_action and
        language_model.forward) torch's cudagraph_trees keeps the previous run's outputs
        alive across the boundary and refuses the next replay:
          "accessing tensor output of CUDAGraphs that has been overwritten by a subsequent
           run ... call torch.compiler.cudagraph_mark_step_begin() before"
        Cloning the outputs was necessary but not sufficient; the generation counter has to
        advance too. This is the documented API for exactly that, and it is a no-op when
        nothing is graph-captured, so every rung runs the same code path.
        """
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        if args.api == "old":
            # N1.5's own policy runs the model under inference_mode + autocast
            # (policy.py:189-191); timing it any other way would not be its inference.
            with torch.inference_mode(), torch.autocast(device_type="cuda",
                                                        dtype=torch.bfloat16):
                return model.get_action(inputs)
        with torch.no_grad():
            return model.get_action(inputs)

    try:
        for _ in range(args.warmup):
            step()
        torch.cuda.synchronize()
    except Exception as e:                                        # noqa: BLE001
        _fail_row(e)
        return 1

    e2e, per = [], {k: [] for k in spans}
    for _ in range(args.timed):
        for k in spans:
            spans[k].clear()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        e2e.append((time.perf_counter() - t0) * 1e3)
        for k, evs in spans.items():
            per[k].append(sum(s.elapsed_time(e) for s, e in evs))
    for u in undo:
        u()

    st = summarize(e2e)
    rows = []
    # vision is nested inside vlm_backbone, so the prefix is the difference. Reporting
    # both the nested region and the derived one keeps the arithmetic checkable.
    per["vlm_prefix_excl_vision"] = [b - v for b, v in
                                     zip(per["vlm_backbone"], per["vision"])] \
        if per.get("vision") and any(per["vision"]) else []
    for k in ("vision", "vlm_backbone", "vlm_prefix_excl_vision", "ae_total"):
        if not per.get(k):
            continue
        s = summarize(per[k])
        rows.append({"region": k, **{kk: round(vv, 4) for kk, vv in s.items()},
                     "share_of_e2e": round(s["median_ms"] / st["median_ms"], 4)})
    dps = summarize(dp)
    rows.append({"region": "data_processing_cpu",
                 **{kk: round(vv, 4) for kk, vv in dps.items()},
                 "share_of_e2e": round(dps["median_ms"] / st["median_ms"], 4)})
    rows.append({"region": "e2e", **{kk: round(vv, 4) for kk, vv in st.items()},
                 "share_of_e2e": 1.0})
    rowdir = out / "raw" / "rows"
    rowdir.mkdir(parents=True, exist_ok=True)
    (rowdir / f"{args.rung}.json").write_text(json.dumps({
        "rung": args.rung, "N": n,
        "compiled": ";".join(compiled_desc), "compile_s": round(compile_s, 3),
        "e2e_median_ms": round(st["median_ms"], 4),
        "e2e_p5_ms": round(st["p5_ms"], 4), "e2e_p95_ms": round(st["p95_ms"], 4),
        "vision_ms": round(summarize(per["vision"])["median_ms"], 4)
                     if per.get("vision") and any(per["vision"]) else "",
        "vlm_backbone_ms": round(summarize(per["vlm_backbone"])["median_ms"], 4),
        "vlm_prefix_ms": round(summarize(per["vlm_prefix_excl_vision"])["median_ms"], 4)
                         if per.get("vlm_prefix_excl_vision") else "",
        "ae_total_ms": round(summarize(per["ae_total"])["median_ms"], 4),
        "data_processing_cpu_ms": round(statistics.median(dp), 4),
        "matmul_precision": "high" if args.tf32 == "on" else "highest",
        "n": st["n"],
    }, indent=1))

    with (out / "phase_split.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'region':12s} {'median ms':>10s} {'share':>7s}")
    for r in rows:
        print(f"{r['region']:12s} {r['median_ms']:>10.2f} {100*r['share_of_e2e']:>6.1f}%")

    (out / "meta.json").write_text(json.dumps({
        "experiment": "E4d (GR00T phase split)", "model": f"groot_{args.version}",
        "checkpoint": args.checkpoint, "precision": "bfloat16",
        "matmul_precision": "high" if args.tf32 == "on" else "highest",
        "compile": ";".join(compiled_desc) or "eager",
        "rung": args.rung, "num_inference_timesteps": n,
        "n_warmup": args.warmup, "n_timed": args.timed, "gate": g,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
        "embodiment_tag": args.embodiment_tag,
        "gr00t_root": args.gr00t_root,
        "compat_shims": compat,
        "observation_shapes": {k: {kk: list(vv) for kk, vv in v.items()}
                               for k, v in shapes.items()},
        "fixture_note": "keys and shapes are read from the checkpoint's own "
                        "processor_config.json and statistics.json; pixel and state "
                        "VALUES are random, which latency does not depend on",
        "data_processing_note": "data_processing_cpu is the checkpoint's own processor + "
                                "collator, timed separately; e2e below is the GPU path "
                                "only (model.get_action on already-collated inputs), the "
                                "same split NVIDIA's benchmark_inference.py reports",
        "vision_note": "SUPERSEDED 2026-09-23: vision IS separated now. GR00T's backbone "
                       "is Qwen3VLForConditionalGeneration, whose .visual and "
                       ".language_model are the same two parts the LeRobot models expose "
                       "as embed_image and pg.forward, so vision is timed as a nested "
                       "region and vlm_prefix is the backbone minus it. GR00T therefore "
                       "gets the same three-way split as every other model in the study.",
        "regions": {"vision": "backbone.model.visual (Qwen3-VL image encoder)",
                    "vlm_backbone": "Gr00tN1d7.backbone (gr00t_n1d7.py:603), vision INSIDE",
                    "vlm_prefix_excl_vision": "vlm_backbone minus vision, derived",
                    "ae_total": "action_head.get_action, containing the "
                                "num_inference_timesteps loop (:366+)"},
    }, indent=1))
    print(f"\nwrote {out}")
    return 0


def _load_n15(args, ckpt_dir):
    """N1.5 uses the older Gr00tPolicy API, so it gets its own constructor.

    N1.6/N1.7: Gr00tPolicy(embodiment_tag, model_path, device=...) and the checkpoint
    carries its own processor.
    N1.5: Gr00tPolicy(model_path, embodiment_tag, modality_config, modality_transform,
    denoising_steps, device) -- the caller supplies the data config, which lives in
    gr00t.experiment.data_config.DATA_CONFIG_MAP.

    The phase split is the same measurement in both: policy.apply_transforms is the CPU
    input pipeline and policy.model.get_action is the GPU path, which is exactly the
    decomposition N1.5's own Gr00tPolicy.get_action uses (policy.py:180-182).
    """
    from gr00t.experiment.data_config import DATA_CONFIG_MAP                  # noqa: E402
    from gr00t.model.policy import Gr00tPolicy                                # noqa: E402

    if args.data_config not in DATA_CONFIG_MAP:
        raise SystemExit(f"data config {args.data_config!r} not in DATA_CONFIG_MAP; "
                         f"it has {sorted(DATA_CONFIG_MAP)}")
    dc = DATA_CONFIG_MAP[args.data_config]
    policy = Gr00tPolicy(
        model_path=ckpt_dir,
        embodiment_tag=args.embodiment_tag,
        modality_config=dc.modality_config(),
        modality_transform=dc.transform(),
        device="cuda",
    )
    return policy


def _build_obs_n15(policy, ckpt_dir, image_hw=(256, 256), text="pick up the object"):
    """N1.5's observation keys are flat: video.<k>, state.<k>, annotation.<k>."""
    import numpy as np

    mc = policy.get_modality_config()
    obs = {}
    shapes = {"video": {}, "state": {}}
    vcfg = mc["video"]
    h, w = image_hw
    for k in vcfg.modality_keys:
        name = k if k.startswith("video.") else f"video.{k}"
        a = np.random.randint(0, 255, (len(vcfg.delta_indices), h, w, 3), dtype=np.uint8)
        obs[name] = a
        shapes["video"][name] = a.shape
    scfg = mc["state"]
    dims = _state_dims_n15(ckpt_dir, policy, scfg.modality_keys)
    for k in scfg.modality_keys:
        name = k if k.startswith("state.") else f"state.{k}"
        a = np.random.randn(len(scfg.delta_indices), dims[k.split(".")[-1]]).astype(
            np.float32)
        obs[name] = a
        shapes["state"][name] = a.shape
    for k in mc.get("language", mc.get("annotation")).modality_keys:
        name = k if k.startswith("annotation.") else f"annotation.{k}"
        obs[name] = [text]
    return obs, shapes


def _state_dims_n15(ckpt_dir, policy, state_keys):
    """Widths from the checkpoint's experiment_cfg/metadata.json (N1.5's layout)."""
    import json
    meta = json.loads((Path(ckpt_dir) / "experiment_cfg" / "metadata.json").read_text())
    tag = getattr(policy.embodiment_tag, "value", policy.embodiment_tag)
    if tag not in meta:
        raise SystemExit(f"metadata.json has no embodiment {tag!r}; it has {sorted(meta)}")
    st = meta[tag]["statistics"]["state"]
    out = {}
    for k in state_keys:
        leaf = k.split(".")[-1]
        if leaf not in st:
            raise SystemExit(f"state key {leaf!r} missing from metadata statistics")
        out[leaf] = len(st[leaf]["min"])
    return out


def _vision_module(model):
    """The backbone's image encoder, whatever this GR00T version calls it.

    N1.7's Qwen3-VL backbone names it `visual`; N1.6's Eagle backbone names it
    `vision_model` (eagle_backbone.py:72). Both sit under backbone.model, and both are the
    counterpart of the LeRobot family's embed_image, so the three-way phase split is the
    same measurement across versions.
    """
    for container in _backbone_containers(model):
        for attr in ("visual", "vision_model"):
            m = getattr(container, attr, None)
            if m is not None:
                return m
    return None


def _backbone_containers(model):
    """Where the two halves hang, across GR00T versions.

    N1.7 (Qwen3Backbone) and N1.6 (EagleBackbone) both use `backbone.model`; N1.5's
    EagleBackbone calls it `backbone.eagle_model` (eagle_backbone.py:59). Getting this
    wrong is silent -- the vision hook simply never fires and the region reports 0.00 ms,
    which is exactly what N1.5's first run did.
    """
    out = []
    for attr in ("model", "eagle_model"):
        c = getattr(model.backbone, attr, None)
        if c is not None:
            out.append(c)
    return out


def _language_module(model):
    """The backbone's LLM. Same attribute name everywhere, different container."""
    for container in _backbone_containers(model):
        m = getattr(container, "language_model", None)
        if m is not None:
            return m
    return None


def build_observation(policy, ckpt_dir, image_hw=(256, 256),
                      text="pick up the object"):
    """A fixture observation in the shape THIS checkpoint declares, not a guessed one.

    Every key and every dimension is read from the checkpoint's own files:
      * which camera views, and how many frames of each -> processor_config.json's
        modality_configs[<tag>]["video"] (modality_keys, delta_indices)
      * which state vectors, and how wide each one is -> the same file for the names and
        statistics.json for the widths (len(statistics[<tag>]["state"][key]["min"]))
    Guessing these is exactly the mistake that made SmolVLA run three inferences on zero
    images without raising anything.

    The pixel and state VALUES are random. Latency does not depend on them -- there is no
    data-dependent control flow in the backbone or the flow-matching loop -- but the shapes,
    dtypes and key names do, and those are the checkpoint's. This is stated in meta.json
    rather than left for a reader to discover.
    """
    import numpy as np

    tag = policy.embodiment_tag.value
    mc = policy.modality_configs
    video_keys = mc["video"].modality_keys
    n_frames = len(mc["video"].delta_indices)
    state_keys = mc["state"].modality_keys
    n_state_steps = len(mc["state"].delta_indices)

    h, w = image_hw
    video = {k: np.random.randint(0, 255, (1, n_frames, h, w, 3), dtype=np.uint8)
             for k in video_keys}

    dims = _state_dims(ckpt_dir, tag, state_keys)
    state = {k: np.random.randn(1, n_state_steps, dims[k]).astype(np.float32)
             for k in state_keys}

    obs = {"video": video, "state": state,
           "language": {policy.language_key: [text]}}
    shapes = {"video": {k: v.shape for k, v in video.items()},
              "state": {k: v.shape for k, v in state.items()}}
    return obs, shapes


def _state_dims(ckpt_dir, tag, state_keys):
    """Width of each state vector, from the checkpoint's statistics."""
    import json
    stats = json.loads((Path(ckpt_dir) / "statistics.json").read_text())
    if tag not in stats:
        raise SystemExit(f"statistics.json has no entry for embodiment {tag!r}; "
                         f"it has {sorted(stats)}")
    out = {}
    for k in state_keys:
        if k not in stats[tag]["state"]:
            raise SystemExit(f"state key {k!r} is in the modality config but not in "
                             f"statistics.json for {tag!r}")
        out[k] = len(stats[tag]["state"][k]["min"])
    return out


def build_inputs(policy, obs):
    """Run the checkpoint's own processor + collator, exactly as inference does.

    This mirrors prepare_model_inputs in NVIDIA's own
    scripts/deployment/benchmark_inference.py: one VLAStepData per batch element through
    policy.processor, then policy.collate_fn, then everything floating-point to bf16.

    The result already contains input_ids / pixel_values, so Gr00tN1d7.prepare_input finds
    no "vlm_content" key and skips the collator: model.get_action(inputs) is then the
    GPU-only path, which is what the phase split must time. The CPU processing this
    function does is timed separately and reported as its own phase, because it is real
    inference cost that the LeRobot models' region timing also leaves out.
    """
    import torch as _t
    from gr00t.data.types import MessageType, VLAStepData

    bs = next(iter(obs["video"].values())).shape[0]
    processed = []
    for i in range(bs):
        step = VLAStepData(
            images={k: v[i] for k, v in obs["video"].items()},
            states={k: v[i] for k, v in obs["state"].items()},
            actions={},
            text=obs["language"][policy.language_key][i],
            embodiment=policy.embodiment_tag,
        )
        processed.append(policy.processor(
            [{"type": MessageType.EPISODE_STEP.value, "content": step}]))
    collated = policy.collate_fn(processed)["inputs"]

    def to_bf16(x):
        if isinstance(x, _t.Tensor) and _t.is_floating_point(x):
            return x.to(dtype=_t.bfloat16)
        if isinstance(x, dict) or hasattr(x, "items"):
            return {k: to_bf16(v) for k, v in x.items()}
        if isinstance(x, list):
            return [to_bf16(v) for v in x]
        return x

    return to_bf16(collated)


if __name__ == "__main__":
    raise SystemExit(main())
