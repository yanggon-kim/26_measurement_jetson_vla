"""Secrets scan for raw profiler files (docs/02_protocol.md §3.5; T001 step 6).

Prints / writes, per file and per class: found or not found, and byte offsets only -- never the matched bytes.
Classes:
  canary:<name>     literal value read from a file (--canary-file, one value per file; never printed)
  env:<VAR>         the value of every token-like variable of this process's environment
                    (name matches token|secret|key|session|auth, value >= 8 characters)
  file:<path>       the content of each --token-file that exists
  re:<class>        the protocol §3.5 regex patterns (hf_, gh*_, github_pat_, AKIA, sk-ant-, PRIVATE KEY)
The search runs on the raw bytes (ASCII and UTF-16LE encodings of each literal), which covers what
`strings -n 8 | grep` covers for literals of >= 8 characters.

Usage: python bench/secrets_scan.py --out scan.csv [--canary-file f ...] [--token-file f ...] FILE ...
Exit code 0 = clean, 1 = at least one hit.
"""
import argparse
import csv
import os
import re
import sys

REGEXES = {
    "hf_token": rb"hf_[A-Za-z0-9]{20,}",
    "github_token": rb"gh[pousr]_[A-Za-z0-9]{20,}",
    "github_pat": rb"github_pat_[A-Za-z0-9_]{20,}",
    "aws_key": rb"AKIA[0-9A-Z]{16}",
    "anthropic_key": rb"sk-ant-[A-Za-z0-9_-]{20,}",
    "private_key": rb"BEGIN [A-Z ]*PRIVATE KEY",
}
NAME_RE = re.compile(r"token|secret|key|session|auth", re.I)


def literal_classes(canary_files, token_files):
    out = {}
    for f in canary_files:
        v = open(f).read().strip()
        if v:
            out[f"canary:{os.path.basename(f)}"] = v
    for k, v in os.environ.items():
        if NAME_RE.search(k) and len(v) >= 8:
            out[f"env:{k}"] = v
    for f in token_files:
        p = os.path.expanduser(f)
        if os.path.exists(p):
            v = open(p, errors="ignore").read().strip()
            if v:
                out[f"file:{f}"] = v
    return out


def offsets(data, needle, limit=20):
    res, i = [], data.find(needle)
    while i >= 0 and len(res) < limit:
        res.append(i)
        i = data.find(needle, i + 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--canary-file", action="append", default=[])
    ap.add_argument("--token-file", action="append", default=["~/.cache/huggingface/token"])
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    lits = literal_classes(a.canary_file, a.token_file)
    rows, hit = [], False
    for path in a.files:
        data = open(path, "rb").read()
        for cls, val in lits.items():
            offs = []
            for enc in (val.encode(), val.encode("utf-16-le")):
                offs += offsets(data, enc)
            rows.append({"file": path, "bytes": len(data), "class": cls, "found": bool(offs),
                         "n_hits_capped20": len(offs), "offsets": " ".join(map(str, sorted(offs)))})
            hit |= bool(offs)
        for cls, rx in REGEXES.items():
            offs = [m.start() for m in re.finditer(rx, data)][:20]
            rows.append({"file": path, "bytes": len(data), "class": f"re:{cls}", "found": bool(offs),
                         "n_hits_capped20": len(offs), "offsets": " ".join(map(str, offs))})
            hit |= bool(offs)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["found"]:
            print(f"HIT {r['file']} {r['class']} offsets={r['offsets']}")
    print(f"scanned {len(a.files)} files x {len(lits) + len(REGEXES)} classes: {'NOT CLEAN' if hit else 'clean'}")
    sys.exit(1 if hit else 0)


if __name__ == "__main__":
    main()
