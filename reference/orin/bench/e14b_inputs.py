"""T017 step 3: the fixed 64-observation input set (8 tasks x 8 frames), fetched BEFORE any T017 run.

Source: the HuggingFaceVLA/libero dataset through the same datasets-server rows API as bench/e12_fixtures.py
(same image pipeline and normalisation as bench/fixtures/observation.json), episode ranges from the dataset's
own meta/episodes parquet.

Selection rule (fixed here, before any result; not tuned):
  tasks   8 tasks with 8 DISTINCT valid-prompt lengths (PaliGemma tokenizer on "<task>\\n", as LeRobot's
          Pi0NewLineProcessor + TokenizerProcessorStep do), spanning all four LIBERO suites:
          task_index 0 (21 tokens, the fixture's task; libero_10), 16 (6; goal), 10 (8; goal), 13 (9; goal),
          14 (11; goal), 21 (12; object), 3 (14; libero_10), 36 (17; spatial).
  frames  for task t with episodes e_0 < ... < e_{n-1} (dataset order), input j = 0..7 takes episode
          e_{round(j (n-1) / 7)} at frame round(j / 8 * (length - 1)) -- episodes spread over the task and the
          frame moving through the trajectory. Task 0, j = 0 is row offset 0 = the committed fixture.
  noise   per input i a torch.Generator seed 17_000 + i (recorded; applied by the consumer).

Writes <out>/obs_<i>.json + obs_<i>_image{,2}.png (bench/fixtures/e12 schema, so pi0_common.build_batch reads
them), <out>/manifest.json and <out>/inputs.csv (i, row offset, episode, frame, task_index, prompt, valid prompt
tokens, noise seed, sha256 of the DECODED pixels of both cameras).

    python bench/e14b_inputs.py --meta <dir with episodes parquet + tasks parquet> --out bench/fixtures/e14b
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
from e12_fixtures import DATASET, ROWS, fetch  # noqa: E402

TASKS = (0, 16, 10, 13, 14, 21, 3, 36)
PER_TASK = 8
SEED0 = 17_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="dir holding chunk-000_file-000.parquet and meta_tasks.parquet")
    ap.add_argument("--out", default="bench/fixtures/e14b")
    a = ap.parse_args()
    import numpy as np                                                            # noqa: PLC0415
    import pandas as pd                                                           # noqa: PLC0415
    from PIL import Image                                                         # noqa: PLC0415
    from transformers import AutoTokenizer                                        # noqa: PLC0415

    meta = Path(a.meta)
    tok = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    tasks = pd.read_parquet(meta / "meta_tasks.parquet").reset_index().rename(columns={"index": "task"})
    tname = dict(zip(tasks["task_index"], tasks["task"]))
    ep = pd.read_parquet(meta / "chunk-000_file-000.parquet")[
        ["episode_index", "dataset_from_index", "dataset_to_index", "tasks", "length"]]
    ep["task"] = ep["tasks"].map(lambda x: x[0])
    lens = {t: len(tok(tname[t] + "\n")["input_ids"]) for t in TASKS}
    if len(set(lens.values())) != len(TASKS):
        raise SystemExit(f"prompt lengths not distinct: {lens}")
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    rows, frames = [], []
    i = 0
    for t in TASKS:
        E = ep[ep["task"] == tname[t]].sort_values("episode_index").reset_index(drop=True)
        n = len(E)
        for j in range(PER_TASK):
            e = E.iloc[round(j * (n - 1) / (PER_TASK - 1))]
            fi = round(j / PER_TASK * (int(e["length"]) - 1))
            off = int(e["dataset_from_index"]) + fi
            row = json.loads(fetch(ROWS.format(offset=off)))["rows"][0]["row"]
            if row["episode_index"] != int(e["episode_index"]) or row["task_index"] != t or row["frame_index"] != fi:
                raise SystemExit(f"offset {off}: row says episode {row['episode_index']} task {row['task_index']} "
                                 f"frame {row['frame_index']}, expected {int(e['episode_index'])}/{t}/{fi}")
            rec = {"dataset": DATASET, "row_offset": off, "episode_index": row["episode_index"],
                   "frame_index": row["frame_index"], "task_index": t, "task": tname[t],
                   "timestamp": row.get("timestamp"), "state": row["observation.state"],
                   "action_in_dataset": row.get("action"), "images": {}}
            hpix = hashlib.sha256()
            for key in ("observation.images.image", "observation.images.image2"):
                png = fetch(row[key]["src"])
                name = f"obs_{i}_{key.rsplit('.', 1)[-1]}.png"
                (out / name).write_bytes(png)
                pix = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
                hpix.update(pix.tobytes())
                rec["images"][key] = {"file": name, "width": row[key].get("width"), "height": row[key].get("height"),
                                      "mode": "RGB", "bytes": len(png), "sha256": hashlib.sha256(png).hexdigest(),
                                      "pixels_sha256": hashlib.sha256(pix.tobytes()).hexdigest()}
            (out / f"obs_{i}.json").write_text(json.dumps(rec, indent=2) + "\n")
            rows.append({"input": i, "row_offset": off, "episode_index": rec["episode_index"],
                         "frame_index": fi, "task_index": t, "prompt": tname[t], "valid_prompt_tokens": lens[t],
                         "noise_seed": SEED0 + i, "pixels_sha256": hpix.hexdigest()})
            frames.append({"file": f"obs_{i}.json", "row_offset": off})
            print(f"  obs_{i:2d}: task {t:2d} (L={lens[t]:2d}) episode {rec['episode_index']:4d} frame {fi:3d} "
                  f"offset {off}", flush=True)
            i += 1
    with (out / "inputs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (out / "manifest.json").write_text(json.dumps({
        "_comment": "T017 step 3 input set, fixed before any T017 run (selection rule in bench/e14b_inputs.py)",
        "dataset": DATASET, "split": "train", "source_api": ROWS.split("?")[0], "tasks": list(TASKS),
        "valid_prompt_tokens": {str(t): lens[t] for t in TASKS}, "noise_seed": f"{SEED0} + input",
        "frames": frames}, indent=2) + "\n")
    print(f"wrote {i} observations; distinct prompt lengths {sorted(set(lens.values()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
