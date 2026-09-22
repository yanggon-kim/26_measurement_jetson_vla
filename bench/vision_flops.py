"""T004 / A1 — analytic FLOPs for the vision tower, to complete the roofline.

`floor.csv` carries `flops = 0` for the `vision` region with the note "FLOPs not counted
analytically". That leaves vision without an operational intensity, and therefore out of
the roofline comparison against arXiv:2604.24447, which reports 840 FLOP/B for the VLM
backbone and 64.5 for the Action Expert. Our own analytic figures for those two are
816.0 and 60.0; vision is the missing third.

Shapes are read from the checkpoint's safetensors header, not assumed: every count below
is derived from a tensor that exists in `model.safetensors`, and the script prints which
one. Nothing is loaded onto the GPU.

Convention matches bench/run_e3.py's floor rows: a multiply-accumulate is 2 FLOPs, and
only the matrix work is counted (norms, activations and softmax are excluded, as they are
for the other regions — their cost is bytes, not FLOPs).

    python bench/vision_flops.py --json results/<run>/vision_flops.json
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, write_json  # noqa: E402

CKPT = Path.home() / (".cache/huggingface/hub/models--lerobot--pi0_libero_finetuned_v044/"
                      "snapshots/45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f/model.safetensors")


def header(p: Path) -> dict:
    with p.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--json", default=None)
    ap.add_argument("--images", type=int, default=3,
                    help="image slots actually encoded per inference (this checkpoint "
                         "always encodes 3: two cameras plus a synthesised empty one)")
    args = ap.parse_args()
    h = header(Path(args.ckpt))
    h.pop("__metadata__", None)

    def shape(suffix: str):
        for k, v in h.items():
            if k.endswith(suffix):
                return k, v["shape"], v["dtype"]
        raise SystemExit(f"no tensor ending in {suffix!r}")

    k_pe, pe, dt_pe = shape("vision_model.embeddings.patch_embedding.weight")
    k_pos, pos, _ = shape("vision_model.embeddings.position_embedding.weight")
    k_q, q, dt_q = shape("vision_model.encoder.layers.0.self_attn.q_proj.weight")
    k_f1, f1, _ = shape("vision_model.encoder.layers.0.mlp.fc1.weight")
    k_pr, pr, _ = shape("multi_modal_projector.linear.weight")
    L = 1 + max(int(m.group(1)) for k in h
                if (m := re.search(r"vision_tower.*encoder\.layers\.(\d+)\.", k)))

    D = q[0]                      # hidden
    T = pos[0]                    # tokens per image = patches
    I = f1[0]                     # mlp intermediate
    Cin, kh, kw = pe[1], pe[2], pe[3]
    Dout = pr[0]                  # projector output

    per = {}
    per["patch_embedding"] = 2 * T * (Cin * kh * kw) * D
    per["qkv_proj"] = 3 * 2 * T * D * D
    per["attn_scores"] = 2 * T * T * D          # QK^T summed over all heads
    per["attn_apply"] = 2 * T * T * D           # scores x V
    per["o_proj"] = 2 * T * D * D
    per["mlp_fc1"] = 2 * T * D * I
    per["mlp_fc2"] = 2 * T * I * D
    per_layer = sum(v for k, v in per.items() if k != "patch_embedding")
    one_image = per["patch_embedding"] + L * per_layer + 2 * T * D * Dout
    total = one_image * args.images

    # bytes: the vision region's weights, read once per inference (floor.csv's basis)
    vis_bytes = sum(
        (v["shape"] and __import__("math").prod(v["shape"]) or 0)
        * (4 if v["dtype"] == "F32" else 2)
        for k, v in h.items() if "vision_tower" in k or "multi_modal_projector" in k)
    # as LOADED: params_to_keep_float32 upcasts the whole vision path to fp32
    vis_bytes_loaded = sum(
        (__import__("math").prod(v["shape"]) if v["shape"] else 0) * 4
        for k, v in h.items() if "vision_tower" in k or "multi_modal_projector" in k)

    out = {
        "_purpose": "analytic FLOPs for the vision region, completing floor.csv's zero",
        "_shapes_from": str(args.ckpt),
        "_tensors_used": {"patch_embedding": k_pe, "position_embedding": k_pos,
                          "q_proj": k_q, "mlp_fc1": k_f1, "projector": k_pr},
        "_convention": "one multiply-accumulate = 2 FLOPs; matrix work only, as in "
                       "bench/run_e3.py's floor rows (norms/softmax/activations excluded)",
        "encoder_layers": L, "hidden": D, "tokens_per_image": T, "mlp_intermediate": I,
        "patch": [Cin, kh, kw], "projector_out": Dout,
        "images_encoded_per_inference": args.images,
        "flops_per_component_per_image": per,
        "flops_per_layer": per_layer,
        "flops_one_image": one_image,
        "flops_vision_region": total,
        "bytes_as_stored_in_checkpoint": vis_bytes,
        "bytes_as_loaded_fp32": vis_bytes_loaded,
        "_dtype_note": (
            f"patch_embedding is {dt_pe} and the encoder linears are {dt_q} IN THE "
            f"CHECKPOINT; modeling_pi0.py:410-417 upcasts the whole vision path to fp32 at "
            f"load time via params_to_keep_float32, for a stated TRAINING reason (optimizer "
            f"dtype toggling). Running it in bf16 restores the stored precision rather than "
            f"discarding any."),
        "operational_intensity_FLOP_per_byte_fp32": round(total / vis_bytes_loaded, 2),
        "operational_intensity_FLOP_per_byte_bf16": round(total / vis_bytes, 2),
        "measured_ridge_point_FLOP_per_byte": 180.57,
    }
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("_")}, indent=2))
    print(f"\nvision I = {out['operational_intensity_FLOP_per_byte_fp32']} FLOP/B as loaded "
          f"(fp32) -> {'COMPUTE' if out['operational_intensity_FLOP_per_byte_fp32'] > 180.57 else 'MEMORY'}-bound")
    print(f"paper arXiv:2604.24447: VLM 840, Action Expert 64.5 FLOP/B; ours 816.0 and 60.0")
    if args.json:
        write_json(Path(args.json), out)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
