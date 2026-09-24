"""Export per-kernel SASS from an ncu report, in the format build_inst_inventory expects.

E5b needs the SASS of the action expert's dominant kernels. The obvious route is
`cuobjdump -sass <binary>`, and it is the wrong one here:

  * the hot kernels are `ampere_sgemm_*`, `ampere_bf16_s16816gemm_*`, `cutlass::Kernel2`
    and cuBLAS's `internal::kernel` — they live inside libcublas / libcublasLt, not in
    anything this project builds, so the binary has to be located and the mangled name
    guessed;
  * the Triton kernels do not exist on disk until they are compiled, and which `.cubin`
    in the cache corresponds to which launch is not recorded anywhere;
  * and neither tells you which of several architectures' cubins actually ran.

`ncu --import <rep> --page source --print-source sass` sidesteps all three: it prints the
SASS **that actually executed**, per kernel, from the report itself. Checked on this
device against a committed C1 report.

The one mismatch is the line format. `bench/build_inst_inventory.py` is the research
side's reference adapted only for sm_87's tensor-core opcodes, and its `INST_RE` expects
cuobjdump's `/*0040*/  OPCODE ...`. ncu prints `0x42dd29700   OPCODE ...`. Rather than
widen the reference parser — which would make it harder to compare against the desktop
study it came from — this rewrites ncu's lines into cuobjdump's shape. The PC is carried
through, masked to its low 32 bits so `--inner-loop-range` still works on it.

One cosmetic consequence: the inventory prints "file has N lines; cuobjdump emits ~2
lines per instruction". This writer emits ONE line per instruction — the control-code
lines cuobjdump interleaves are not instructions and are not reproduced — so that
parenthetical does not apply here. The COUNT is unaffected: the parser counts `INST_RE`
matches, not lines.

    python bench/ncu_sass_dump.py results/<run>/ncu/<tag>.ncu-rep.zst \\
        --outdir results/<run>/raw/sass
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crosscheck_sass import _short_name  # noqa: E402

# The plain-text source page TRUNCATES the kernel name to its 60-character column, which
# turned `void at::native::vectorized_elementwise_kernel<(int)4, at::n` into a "kernel"
# called `n`. --csv does not truncate, so the export is parsed as CSV instead.
ADDR_RE = re.compile(r"^0x[0-9a-fA-F]+$")


def export(rep: Path, outdir: Path, ncu: str) -> dict[str, Path]:
    """Run ncu --page source --csv and split its output into one file per kernel."""
    txt = subprocess.run(
        ["sudo", "-n", ncu, "--import", str(rep), "--page", "source",
         "--print-source", "sass", "--csv"],
        capture_output=True, text=True, timeout=900).stdout
    outdir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    state = {"cur": None, "lines": []}

    def flush():
        if not state["cur"] or not state["lines"]:
            return
        short = _short_name(state["cur"]) or "kernel"
        # Several launches of one kernel give several blocks with identical SASS, so the
        # first is kept and the rest dropped. Appending would double every opcode count.
        f = outdir / f"{short}.sass"
        if f.exists():
            return
        f.write_text("".join(state["lines"]))
        written[short] = f

    for row in csv.reader(io.StringIO(txt)):
        if len(row) < 2:
            continue
        a, b = row[0].strip(), row[1]
        if a == "Kernel Name":
            flush()
            state["cur"], state["lines"] = b.strip(), []
        elif state["cur"] and ADDR_RE.match(a):
            # cuobjdump's shape, which build_inst_inventory.py's INST_RE expects. The PC
            # is masked to 32 bits so --inner-loop-range still works on it.
            pc = int(a, 16) & 0xFFFFFFFF
            state["lines"].append(f"        /*{pc:04x}*/{' ' * 8}{b.strip()} ;\n")
    flush()
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path, help=".ncu-rep or .ncu-rep.zst")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--ncu", default="/usr/local/cuda/bin/ncu")
    args = ap.parse_args()

    rep = args.report
    tmp = None
    if rep.suffix == ".zst":
        # The .ncu-rep itself is not committed (protocol §6 keeps only the compressed
        # one), so decompress to a scratch file rather than beside the committed archive.
        tmp = tempfile.NamedTemporaryFile(suffix=".ncu-rep", delete=False)
        tmp.close()
        subprocess.run(["zstd", "-dqf", str(rep), "-o", tmp.name], check=True, timeout=600)
        rep = Path(tmp.name)

    try:
        written = export(rep, args.outdir, args.ncu)
    finally:
        if tmp:
            Path(tmp.name).unlink(missing_ok=True)

    if not written:
        print(f"no SASS extracted from {args.report}. The report may have been captured "
              f"without source correlation, or ncu's source page may be empty for these "
              f"kernels — say so rather than reporting an empty inventory.",
              file=sys.stderr)
        return 1
    for short, p in sorted(written.items()):
        n = sum(1 for line in p.open() if "/*" in line)
        print(f"  {short:52s} {n:6d} instructions -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
