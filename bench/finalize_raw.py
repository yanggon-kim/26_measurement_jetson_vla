"""Prepare a result directory's raw files for committing (protocol §3.5, §5, §6).

For every file under results/<run>/raw/ (and any extra paths given):
  1. hash and size the ORIGINAL first;
  2. scan it for secrets (§3.5) — a hit means the file is not committed;
  3. compress it with `zstd -19 -T0` if it is larger than --compress-above (default 1 MB),
     moving the uncompressed original to scratch/<run>/;
  4. if the committed file is still over --max-bytes (default 50 MB), split it into
     45 MB parts and write REASSEMBLE.txt, verifying the round trip;
  5. emit the meta.json `raw_files` list to <run>/raw_files.json.

  python bench/finalize_raw.py --run results/20260921_E0_platform \
      --kind ncu_toy.ncu-rep=ncu-rep --produced-by ncu_toy.ncu-rep="ncu --set full ..."
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, sh, sha256, write_json  # noqa: E402

SECRET_RE = re.compile(
    r"hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY")

KIND_BY_SUFFIX = {
    ".nsys-rep": "nsys-rep",
    ".sqlite": "nsys-sqlite",
    ".ncu-rep": "ncu-rep",
    ".sass": "sass",
    ".ptx": "ptx",
    ".cubin": "cubin",
    ".log": "stdout",
    ".json": "other",
    ".txt": "other",
    ".csv": "ncu-csv",
}


def scan_secrets(path: Path) -> str:
    """'clean' or a description of what matched. Uses strings(1), per protocol §3.5."""
    r = subprocess.run(f"strings -n 8 {path!s:s}", shell=True, capture_output=True,
                       text=True, errors="replace")
    hits = SECRET_RE.findall(r.stdout)
    if hits:
        return f"FAILED: {len(hits)} match(es), first pattern {hits[0][:6]}..."
    return "clean"


def compress(path: Path, level: int = 19) -> tuple[Path, dict]:
    cmd = f"zstd -{level} -T0 -q -f {path!s:s} -o {path!s:s}.zst"
    r = sh(cmd, timeout=3600)
    if r["returncode"] != 0:
        raise RuntimeError(f"zstd failed: {r['stderr'][:500]}")
    return Path(f"{path}.zst"), {"tool": f"zstd {level}", "level": level, "command": cmd}


def split_file(path: Path, part_bytes: int = 45 * 1024 * 1024) -> list[Path]:
    prefix = f"{path}.part"
    r = sh(f"split -b {part_bytes} -d -a 3 {path!s:s} {prefix}", timeout=3600)
    if r["returncode"] != 0:
        raise RuntimeError(f"split failed: {r['stderr'][:500]}")
    return sorted(path.parent.glob(f"{path.name}.part*"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="result directory")
    ap.add_argument("--compress-above", type=int, default=1 << 20)
    ap.add_argument("--max-bytes", type=int, default=50 * 1000 * 1000)
    ap.add_argument("--kind", action="append", default=[], metavar="FILE=KIND")
    ap.add_argument("--produced-by", action="append", default=[], metavar="FILE=CMD")
    ap.add_argument("--scratch", default=None,
                    help="where uncompressed originals go (default scratch/<run name>)")
    args = ap.parse_args()

    run = Path(args.run)
    raw = run / "raw"
    if not raw.is_dir():
        print(f"no {raw}", file=sys.stderr)
        return 1
    scratch = Path(args.scratch or (REPO / "scratch" / run.name))
    scratch.mkdir(parents=True, exist_ok=True)
    kinds = dict(k.split("=", 1) for k in args.kind)
    produced = dict(p.split("=", 1) for p in args.produced_by)

    entries: list[dict] = []
    for f in sorted(raw.iterdir()):
        if f.is_dir() or f.name in ("REASSEMBLE.txt",) or f.suffix == ".zst" \
                or ".part" in f.name:
            continue
        orig_sha, orig_bytes = sha256(f), f.stat().st_size
        scan = scan_secrets(f)
        e = {
            "path": None,
            "sha256": orig_sha,
            "bytes": orig_bytes,
            "kind": kinds.get(f.name, KIND_BY_SUFFIX.get(f.suffix, "other")),
            "produced_by": produced.get(f.name, ""),
            "secrets_scan": scan,
            "committed": True,
        }
        if scan != "clean":
            e.update({"committed": False,
                      "reason": "secrets scan failed; see protocol §3.5",
                      "path": f"raw/{f.name} (NOT COMMITTED)"})
            shutil.move(str(f), scratch / f.name)
            entries.append(e)
            print(f"  !! {f.name}: {scan} -> moved to scratch, not committed")
            continue

        if orig_bytes > args.compress_above:
            comp, cinfo = compress(f)
            e["compression"] = cinfo
            e["original"] = {"name": f.name, "sha256": orig_sha, "bytes": orig_bytes}
            e["sha256"], e["bytes"] = sha256(comp), comp.stat().st_size
            shutil.move(str(f), scratch / f.name)
            f = comp
            print(f"  zstd {e['original']['name']}: "
                  f"{orig_bytes/1e6:.2f} MB -> {e['bytes']/1e6:.2f} MB")
        e["path"] = f"raw/{f.name}"

        if e["bytes"] > args.max_bytes:
            parts = split_file(f)
            e["parts"] = [{"path": f"raw/{p.name}", "sha256": sha256(p),
                           "bytes": p.stat().st_size} for p in parts]
            e["reassembled"] = {"sha256": e["sha256"], "bytes": e["bytes"]}
            e["path"] = f"raw/{f.name} (split, see raw/REASSEMBLE.txt)"
            with (raw / "REASSEMBLE.txt").open("a") as rf:
                rf.write(f"cat {f.name}.part* > {f.name} && zstd -d {f.name}\n"
                         f"  reassembled {f.name}: sha256={e['sha256']} "
                         f"bytes={e['bytes']}\n"
                         f"  decompressed {e['original']['name']}: "
                         f"sha256={e['original']['sha256']} "
                         f"bytes={e['original']['bytes']}\n\n")
            f.unlink()
            print(f"  split {f.name} into {len(parts)} parts")
        entries.append(e)

    write_json(run / "raw_files.json", entries)
    total = sum(e["bytes"] for e in entries if e["committed"])
    print(f"wrote {run/'raw_files.json'}: {len(entries)} files, "
          f"{total/1e6:.2f} MB committed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
