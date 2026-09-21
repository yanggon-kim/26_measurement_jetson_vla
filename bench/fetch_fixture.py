"""T002 step 2 — fetch ONE real LIBERO observation into bench/fixtures/ (protocol §2).

Protocol §2 requires a real observation (camera images, robot state, language prompt), taken
from a published LIBERO dataset on Hugging Face so that nothing depends on installing the
simulator, with the dataset, episode and frame recorded.

Source: `HuggingFaceVLA/libero` — the dataset `lerobot/pi0_libero_finetuned_v044` was trained
on (its `train_config.json` names it), so the column names already match the policy's
`input_features`. One frame is pulled through the datasets-server rows API: two 256x256 JPEGs
and an 8-float state, well under 100 kB, without downloading any of the 377 parquet shards.
The task string is not in the row (only `task_index`), so it is joined from `meta/tasks.parquet`.

    python bench/fetch_fixture.py --offset 0 --outdir bench/fixtures
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import sha256, write_json  # noqa: E402

DATASET = "HuggingFaceVLA/libero"
ROWS_API = "https://datasets-server.huggingface.co/rows"
HUB = "https://huggingface.co/datasets"


def _get(url: str, timeout: int = 120) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def fetch_row(dataset: str, offset: int) -> dict:
    q = urllib.parse.urlencode(
        {"dataset": dataset, "config": "default", "split": "train",
         "offset": offset, "length": 1})
    return json.loads(_get(f"{ROWS_API}?{q}"))


def task_strings(dataset: str) -> dict[int, str]:
    """task_index -> instruction, from meta/tasks.parquet (a few kB)."""
    import pyarrow.parquet as pq
    raw = _get(f"{HUB}/{dataset}/resolve/main/meta/tasks.parquet")
    tbl = pq.read_table(io.BytesIO(raw))
    d = tbl.to_pydict()
    # In this dataset the instruction is the pandas *index*, so pyarrow surfaces it as
    # `__index_level_0__` rather than a `task` column. Accept either spelling.
    text_col = next((c for c in ("task", "__index_level_0__") if c in d), None)
    if text_col is None:
        raise KeyError(f"no task-text column in meta/tasks.parquet; columns={list(d)}")
    if "task_index" in d:
        return {int(i): str(t) for i, t in zip(d["task_index"], d[text_col])}
    return {i: str(t) for i, t in enumerate(d[text_col])}


def dataset_info(dataset: str) -> dict:
    return json.loads(_get(f"{HUB}/{dataset}/resolve/main/meta/info.json"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--offset", type=int, default=0, help="row index in the train split")
    ap.add_argument("--outdir", default="bench/fixtures")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"fetching row {args.offset} of {args.dataset} ...", flush=True)
    payload = fetch_row(args.dataset, args.offset)
    row = payload["rows"][0]["row"]
    info = dataset_info(args.dataset)
    tasks = task_strings(args.dataset)

    task_index = int(row["task_index"])
    meta = {
        "_comment": "One real LIBERO observation (protocol §2). Images are PNG next to this "
                    "file; they are the dataset's own frames, not re-rendered.",
        "dataset": args.dataset,
        "dataset_codebase_version": info.get("codebase_version"),
        "dataset_total_episodes": info.get("total_episodes"),
        "dataset_total_frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "row_offset": args.offset,
        "episode_index": int(row["episode_index"]),
        "frame_index": int(row["frame_index"]),
        "index": int(row["index"]),
        "timestamp": row.get("timestamp"),
        "task_index": task_index,
        "task": tasks[task_index],
        "source": f"{ROWS_API}?dataset={args.dataset}&config=default&split=train"
                  f"&offset={args.offset}&length=1",
        "images": {},
        "state": [float(x) for x in row["observation.state"]],
        "action_in_dataset": [float(x) for x in row["action"]],
    }

    from PIL import Image
    for key in ("observation.images.image", "observation.images.image2"):
        cell = row[key]
        img_bytes = _get(cell["src"])
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        name = key.replace("observation.images.", "") + ".png"
        im.save(out / name)
        meta["images"][key] = {
            "file": name, "width": im.width, "height": im.height,
            "mode": im.mode, "bytes": (out / name).stat().st_size,
            "sha256": sha256(out / name),
            "note": "saved as PNG (lossless) from the dataset's JPEG cell",
        }
        print(f"  {key} -> {name}  {im.width}x{im.height}")

    write_json(out / "observation.json", meta)
    print(f"  state[{len(meta['state'])}] = {[round(x, 4) for x in meta['state']]}")
    print(f"  task = {meta['task']!r}")
    print(f"  episode {meta['episode_index']} frame {meta['frame_index']}")
    print(f"wrote {out/'observation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
