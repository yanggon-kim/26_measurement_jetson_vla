"""E12: capture several REAL LIBERO observations so a stale-KV bug cannot hide.

`docs/E12_corun/README.md` §6.1 is blunt about why this file exists:

    If every frame uses the same fixture observation, frame k's KV equals frame k-1's and a
    stale-KV bug is **invisible**.

`bench/fixtures/` holds exactly one observation, so the pipeline's fresh-KV check — which
non-negotiable 1 makes the gate on every number E12 reports — would have no power at all.
The spec asks for "at least three observations that give materially different actions
(other frames of the fixture's LIBERO episode, or other episodes — record which)".

## Where these come from, and why not the simulator

The existing fixture is not re-rendered: its own `source` field records that it was pulled
from the HuggingFaceVLA/libero dataset through the datasets-server rows API, episode 0,
frame 0. These come from exactly the same place, so they share its provenance, its image
pipeline and its normalisation — nothing about them is a second, subtly different path.

Booting LIBERO's MuJoCo simulator (bench/libero_eval.py does this, and it works) would have
produced frames too, but re-rendered ones: a different image pipeline for the same nominal
observation, and bench/libero_eval.py's own docstring records how expensive that class of
mismatch was to find. Same dataset, same rows API, no new pipeline.

## Which frames, and why they are enough

Offsets 0..99 of the train split are all episode 0, task_index 0 — verified against the API
before choosing them, not assumed from the 273465/1693 average. So every frame here carries
the same language prompt as the existing fixture and differs only in what the cameras and
the arm are doing. That is the harder case for the equivalence check and the honest one: if
the prompts differed too, a stale-KV mix-up would be trivially visible in the text tokens
and the check would be weaker than it looks.

`--report-diff` prints the pixel difference between consecutive frames. That number is the
scale at which a stale-KV mix-up would show up, and §6.1 asks for it explicitly.

## The control, run rather than asserted

`obs_0` is offset 0 — the same dataset row as `bench/fixtures/observation.json`. Checked
after capture: its decoded pixels are **bit-identical** to the existing fixture's
(`max|d| = 0` on both cameras), and `state` and `task` match exactly. So this path
reproduces the committed fixture rather than quietly fetching something else.

The **PNG bytes differ** (sha256 `16401bf2…` vs `2a5d41bd…`) because the server re-encodes
the image; the pixels do not. Comparing file hashes here would look like a failure and be
one only of the container. The meaningful comparison is the decoded array.

    python bench/e12_fixtures.py --out bench/fixtures/e12
    python bench/e12_fixtures.py --out bench/fixtures/e12 --report-diff
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASET = "HuggingFaceVLA/libero"
ROWS = ("https://datasets-server.huggingface.co/rows"
        "?dataset=HuggingFaceVLA%2Flibero&config=default&split=train"
        "&offset={offset}&length=1")
# Episode 0, verified: a single fetch of offsets 0..99 returned episode_index 0 and
# task_index 0 for every row. Spread across the episode so the arm and the scene actually
# move between them; 0 duplicates the existing fixture, which is deliberate -- it is the
# control that this path reproduces what bench/fixtures/observation.json already holds.
OFFSETS = (0, 25, 50, 75, 99)


def fetch(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:      # noqa: S310
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/fixtures/e12")
    ap.add_argument("--offsets", default=",".join(str(o) for o in OFFSETS))
    ap.add_argument("--report-diff", action="store_true",
                    help="print consecutive-frame pixel differences (the scale at which a "
                         "stale-KV mix-up would show; §6.1 asks for it)")
    args = ap.parse_args()

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    offsets = [int(x) for x in args.offsets.split(",") if x != ""]

    # The prompt. Every frame here is task_index 0, the same task as the existing fixture,
    # so its `task` string is the right one and is taken from the file rather than guessed.
    base = json.loads((REPO / "bench" / "fixtures" / "observation.json").read_text())
    task = base["task"]
    if base.get("task_index") != 0:
        raise SystemExit(f"the existing fixture is task_index {base.get('task_index')}, not "
                         f"0; these frames are task_index 0 and would carry the wrong prompt")

    manifest = {"_comment": (
        "E12 observation set. Real LIBERO frames from the SAME source as "
        "bench/fixtures/observation.json (the HuggingFaceVLA/libero dataset through the "
        "datasets-server rows API), not re-rendered. They exist because "
        "docs/E12_corun/README.md §6.1 requires several materially different observations: "
        "with one observation per frame the expert's KV would be identical every frame and "
        "a stale-KV bug would be invisible, which would leave the fresh-KV check with no "
        "power and every E12 number ungated."),
        "dataset": DATASET, "split": "train", "source_api": ROWS.split("?")[0],
        "episode_index": 0, "task_index": 0, "task": task,
        "frames": []}

    for i, off in enumerate(offsets):
        row = json.loads(fetch(ROWS.format(offset=off)))["rows"][0]["row"]
        if row["episode_index"] != 0 or row["task_index"] != 0:
            raise SystemExit(f"offset {off} is episode {row['episode_index']} / task "
                             f"{row['task_index']}, not episode 0 / task 0 -- these frames "
                             f"would not share the prompt taken from the existing fixture")
        rec = {"_comment": manifest["_comment"], "dataset": DATASET,
               "row_offset": off, "episode_index": row["episode_index"],
               "frame_index": row["frame_index"], "task_index": row["task_index"],
               "task": task, "timestamp": row.get("timestamp"),
               "state": row["observation.state"],
               "action_in_dataset": row.get("action"),
               "images": {}}
        for key in ("observation.images.image", "observation.images.image2"):
            img = row[key]
            png = fetch(img["src"])
            name = f"obs_{i}_{key.rsplit('.', 1)[-1]}.png"
            (out / name).write_bytes(png)
            rec["images"][key] = name
            rec.setdefault("_image_meta", {})[key] = {
                "height": img.get("height"), "width": img.get("width"),
                "sha256": hashlib.sha256(png).hexdigest(), "bytes": len(png)}
        (out / f"obs_{i}.json").write_text(json.dumps(rec, indent=2) + "\n")
        manifest["frames"].append({"file": f"obs_{i}.json", "row_offset": off,
                                   "frame_index": row["frame_index"]})
        print(f"  obs_{i}: offset {off:3d}  episode {row['episode_index']}  "
              f"frame {row['frame_index']:3d}  state[0:3]="
              f"{[round(v, 4) for v in row['observation.state'][:3]]}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n  wrote {len(offsets)} observations + manifest.json to {out}")

    if args.report_diff:
        from PIL import Image                                     # noqa: PLC0415
        import numpy as np                                        # noqa: PLC0415
        print("\n  consecutive-frame difference (the scale a stale-KV mix-up would show at):")
        prev = None
        for i in range(len(offsets)):
            rec = json.loads((out / f"obs_{i}.json").read_text())
            a = np.asarray(Image.open(out / rec["images"]["observation.images.image"]),
                           dtype=np.float32) / 255.0
            st = np.asarray(rec["state"], dtype=np.float32)
            if prev is not None:
                pa, ps = prev
                print(f"    obs_{i-1} -> obs_{i}:  image mean|d|={np.abs(a-pa).mean():.4f}  "
                      f"max|d|={np.abs(a-pa).max():.4f}  state max|d|={np.abs(st-ps).max():.4f}")
            prev = (a, st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
