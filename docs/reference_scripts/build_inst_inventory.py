#!/usr/bin/env python3
"""
Build a per-opcode instruction inventory from a cuobjdump SASS dump.

cuobjdump emits two physical lines per SASS instruction (the disassembled
mnemonic line, plus a control-code/encoding hex line that begins with
whitespace and `/*` followed by a hex pair). We extract opcodes from the
mnemonic lines only, classify each opcode into an execution-pipe family,
and emit a categorised CSV/markdown table.

Pipe categories follow the cuSPARSE-on-RTX-50 mapping discussed in
references/kernel-sass-forensics.md (NVIDIA forum threads + nsight-compute
documentation):

  LSU      : LDG / LDS / LDC / LDCU / STG / STS / REDG / RED / ATOM /
             LDGSTS / LDSM / STSM / TLD4 / LD.E / ST.E (Blackwell)
  FMA      : FFMA / FMUL / FADD / FSEL / IMAD / IMAD.WIDE / FCHK / HFMA2
  ALU      : IADD3 / IADD / ISETP / LOP3 / SHF / MOV / SEL / PRMT / FLO /
             POPC / I2I / I2F / F2I / F2F / IMNMX / VIMNMX / BREV / BMSK
  Shuffle  : SHFL / SHFL.SYNC
  XU       : MUFU / DMUL / DADD / DFMA (FP64 / transcendentals)
  Reduce   : REDUX (Blackwell warp-collective reduction)
  Uniform  : U-prefixed datapath (UMOV, ULEA, UISETP, ULOP3, USHF, UIADD3, ...)
  Branch   : BRA / BSSY / BSYNC / EXIT / NOP / BAR / DEPBAR / WARPSYNC /
             ENDCOLLECTIVE / S2R / S2UR / VOTE / etc.

The LSU category is further split into Load / Store / Atomic sub-classes
by opcode mnemonic (LDx → Load, STx → Store, ATOMx / RED / REDG → Atomic).

Usage:
    python build_inst_inventory.py <sass_file>                     # markdown
    python build_inst_inventory.py <sass_file> --csv               # CSV
    python build_inst_inventory.py <sass_file> --inner-loop-range 0xPC1-0xPC2
        Filters to instructions whose PC is in [PC1, PC2] (inclusive).
        Useful for emitting a separate inner-loop / per-element-block
        inventory after backward-BRA enumeration identifies the hot range.
"""

from __future__ import annotations
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path


# Match a SASS instruction line. cuobjdump format:
#   /*0040*/  PRED?  OPCODE.MOD.MOD  ARGS ;     /* 0xXXXXXX...  */
# We strip the leading `/*PC*/` cookie and the optional `@!P0` predicate.
INST_RE = re.compile(
    r"^\s*/\*([0-9a-fA-F]+)\*/\s+"           # /*0040*/  (group 1 = PC)
    r"(?:@!?P[0-9TF]\s+)?"                   # optional predicate
    r"(?:@U?P[0-9TF]\s+)?"                   # optional uniform predicate
    r"([A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*)"    # opcode + dot-modifiers (group 2)
)


def opcode_family(opcode: str) -> str:
    """Map an opcode (with modifiers) to one of the high-level categories."""
    base = opcode.split(".", 1)[0]

    # Uniform datapath — everything that starts with a U and isn't a stand-alone op
    # like ULDC/ULDCU. Keep these collapsed so we can subtotal "uniform issue".
    UNIFORM_OPS = {
        "ULDC", "ULDCU", "UMOV", "UR2UR", "USEL", "USHF", "UIADD3",
        "UFLO", "UISETP", "ULOP3", "UPLOP3", "UIMAD", "UPSETP", "UPSET",
        "USGXT", "ULEA", "UPLOP", "UNUM", "URET", "UF2F", "UF2I",
        "UI2F", "UPRMT", "UBMSK", "UIMNMX", "UBREV", "UBSYS", "UIADDPSY",
        "USCSY", "UPLOP3", "UPSETP", "UCGAJOINT", "UR2P",
    }
    if base in UNIFORM_OPS or (base.startswith("U") and base not in {"UNDEFINED"} and len(base) >= 3):
        return "Uniform"

    LSU_OPS = {
        "LDG", "LDS", "LDC", "LDCU", "LDL", "LDSM",
        "STG", "STS", "STL", "STSM",
        "REDG", "RED", "ATOMG", "ATOM", "ATOMS",
        "LDGSTS", "LDGDEPBAR", "TLD4", "TLDS", "SUST", "SULD",
        "MEMBAR",
        # Blackwell descriptor-based variants of LDG/STG
        "LD", "ST",
    }
    if base in LSU_OPS:
        return "LSU"

    FMA_OPS = {
        "FFMA", "FMUL", "FADD", "FSEL", "FSET", "FSETP", "FCMP",
        "FMNMX", "FCHK",
        "IMAD", "IMUL",   # IMAD/IMUL go through pipe_fma per NVIDIA forum
        "HFMA2", "HMUL2", "HADD2", "HFMA2_32",
        "BFFMA", "BFMUL", "BFADD",
    }
    if base in FMA_OPS:
        return "FMA"

    ALU_OPS = {
        "IADD3", "IADD", "ISETP", "ISET", "LOP3", "LOP",
        "SHF", "SHFL_INT", "SHL", "SHR",
        "MOV", "SEL", "PRMT", "FLO", "POPC", "BREV", "BMSK",
        "I2I", "I2F", "F2I", "F2F", "I2IP", "F2FP",
        "IMNMX", "IABS", "IDP", "IDP4A",
        "PSETP", "PLOP3", "P2R", "R2P",
        "SGXT", "LEA", "VABSDIFF4", "VABSDIFF",
        "CSGEN", "VIMNMX", "VIADD",
    }
    if base in ALU_OPS:
        return "ALU"

    SHUFFLE_OPS = {"SHFL"}
    if base in SHUFFLE_OPS:
        return "Shuffle"

    XU_OPS = {"MUFU", "DMUL", "DADD", "DFMA", "DSETP", "DSET", "DMNMX"}
    if base in XU_OPS:
        return "XU"

    REDUCE_OPS = {"REDUX"}
    if base in REDUCE_OPS:
        return "Reduce"

    BRANCH_OPS = {
        "BRA", "BSSY", "BSYNC", "BREAK", "CALL", "RET", "JMX", "JMP",
        "BRX", "SSY", "EXIT", "NOP", "BAR", "DEPBAR", "ERRBAR",
        "WARPSYNC", "ENDCOLLECTIVE",
        "S2R", "S2UR", "R2UR", "B2R", "CS2R",
        "VOTE", "VOTEU", "MATCH", "MATCHALL",
        "SETLMEMBASE", "NANOSLEEP", "YIELD",
        "BMSK", "PSETP",
        "QSPC",  # query special / address-space membership query
    }
    if base in BRANCH_OPS:
        return "Branch"

    return "Other"


def lsu_subclass(opcode: str) -> str | None:
    """For LSU opcodes, classify into Load / Store / Atomic.

    Returns None for non-LSU opcodes.
    """
    if opcode_family(opcode) != "LSU":
        return None
    base = opcode.split(".", 1)[0]
    # Atomic / read-modify-write
    if base in {"ATOM", "ATOMG", "ATOMS", "RED", "REDG"}:
        return "Atomic"
    # Stores
    if base in {"STG", "ST", "STS", "STL", "STSM", "SUST"}:
        return "Store"
    # Loads (and load-store fused, treated as a load for accounting)
    if base in {"LDG", "LD", "LDS", "LDC", "LDCU", "LDL", "LDSM",
                "LDGSTS", "TLD4", "TLDS", "SULD", "LDGDEPBAR",
                "MEMBAR"}:  # MEMBAR is a fence; lump with "Load" pipe accounting
        return "Load"
    return "Other"


CATEGORY_ORDER = [
    "LSU", "FMA", "ALU", "Shuffle", "XU", "Reduce", "Uniform", "Branch", "Other",
]


def parse_pc_range(arg: str) -> tuple[int, int]:
    """Parse '0xPC_lo-0xPC_hi' or 'PC_lo-PC_hi' into (lo, hi)."""
    s = arg.strip()
    if "-" not in s:
        raise ValueError(f"--inner-loop-range must be lo-hi, got {arg}")
    lo, hi = s.split("-", 1)
    return int(lo, 16), int(hi, 16)


def categorise(sass_path: Path,
               pc_range: tuple[int, int] | None = None) -> tuple[Counter, Counter, int]:
    """Return (per-opcode counts, per-category counts, total instruction count).

    If pc_range = (lo, hi) is given, only count instructions whose PC is in [lo, hi].
    """
    opcodes: Counter[str] = Counter()
    with open(sass_path) as f:
        for line in f:
            m = INST_RE.match(line)
            if not m:
                continue
            pc = int(m.group(1), 16)
            if pc_range is not None and not (pc_range[0] <= pc <= pc_range[1]):
                continue
            opcodes[m.group(2)] += 1

    cats: Counter[str] = Counter()
    for opcode, count in opcodes.items():
        cats[opcode_family(opcode)] += count

    total = sum(opcodes.values())
    return opcodes, cats, total


def lsu_subtotals(opcodes: Counter) -> Counter:
    """Compute Load / Store / Atomic subtotals within the LSU category."""
    sub: Counter[str] = Counter()
    for op, count in opcodes.items():
        sc = lsu_subclass(op)
        if sc is not None:
            sub[sc] += count
    return sub


def emit_csv(opcodes: Counter, cats: Counter, total: int,
             lsu_sub: Counter, label: str = "") -> None:
    writer = csv.writer(sys.stdout)
    if label:
        writer.writerow([f"# {label}"])
    writer.writerow(["opcode", "count", "category", "lsu_subclass"])
    for opcode, count in sorted(opcodes.items(),
                                key=lambda kv: (opcode_family(kv[0]), -kv[1])):
        sc = lsu_subclass(opcode) or ""
        writer.writerow([opcode, count, opcode_family(opcode), sc])
    writer.writerow([])
    writer.writerow(["category", "count", "pct_of_total"])
    for cat in CATEGORY_ORDER:
        cnt = cats.get(cat, 0)
        if cnt:
            writer.writerow([cat, cnt, f"{cnt/total*100:.1f}%"])
    writer.writerow(["TOTAL", total, "100.0%"])
    if cats.get("LSU", 0) > 0:
        writer.writerow([])
        writer.writerow(["lsu_subclass", "count", "pct_of_lsu", "pct_of_total"])
        lsu_total = cats["LSU"]
        for sub in ("Load", "Store", "Atomic", "Other"):
            cnt = lsu_sub.get(sub, 0)
            if cnt:
                writer.writerow([sub, cnt,
                                 f"{cnt/lsu_total*100:.1f}%",
                                 f"{cnt/total*100:.1f}%"])


def emit_markdown(sass_path: Path, opcodes: Counter, cats: Counter,
                  total: int, lsu_sub: Counter,
                  pc_range: tuple[int, int] | None = None) -> None:
    title = f"# SASS instruction inventory — `{sass_path.name}`"
    if pc_range:
        title += f" (PC range 0x{pc_range[0]:x}-0x{pc_range[1]:x})"
    print(title)
    print()
    file_lines = sum(1 for _ in open(sass_path)) if pc_range is None else None
    if file_lines is not None:
        print(f"Total instructions: **{total}** "
              f"(file has {file_lines} lines; "
              f"cuobjdump emits ~2 lines per instruction).")
    else:
        print(f"Total instructions in PC range: **{total}**.")
    print()
    print("## Per-category subtotal")
    print()
    print("| Category | Count | % of total |")
    print("|---|---:|---:|")
    for cat in CATEGORY_ORDER:
        cnt = cats.get(cat, 0)
        if cnt:
            print(f"| {cat} | {cnt} | {cnt/total*100:.1f}% |")
    print(f"| **Total** | **{total}** | **100.0%** |")
    print()
    if cats.get("LSU", 0) > 0:
        print("## LSU split (Load / Store / Atomic)")
        print()
        print("| Sub-class | Count | % of LSU | % of total |")
        print("|---|---:|---:|---:|")
        lsu_total = cats["LSU"]
        for sub in ("Load", "Store", "Atomic", "Other"):
            cnt = lsu_sub.get(sub, 0)
            if cnt:
                print(f"| {sub} | {cnt} | "
                      f"{cnt/lsu_total*100:.1f}% | {cnt/total*100:.1f}% |")
        print(f"| **LSU total** | **{lsu_total}** | **100.0%** | "
              f"**{lsu_total/total*100:.1f}%** |")
        print()
    print("## Per-opcode breakdown")
    print()
    print("| Category | Sub | Opcode | Count |")
    print("|---|---|---|---:|")
    for cat in CATEGORY_ORDER:
        rows = sorted(((op, c) for op, c in opcodes.items()
                       if opcode_family(op) == cat),
                      key=lambda kv: (-kv[1], kv[0]))
        for op, c in rows:
            sub = lsu_subclass(op) or ""
            print(f"| {cat} | {sub} | `{op}` | {c} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sass", type=Path,
                    help="cuobjdump --dump-sass output file")
    ap.add_argument("--csv", action="store_true",
                    help="emit machine-readable CSV (default: pretty markdown)")
    ap.add_argument("--inner-loop-range", type=str, default=None,
                    help="filter to PC range (e.g. 0x0240-0x1300). "
                         "Use after identifying the inner loop via "
                         "backward-BRA enumeration.")
    ap.add_argument("--label", type=str, default="",
                    help="prefix label for CSV output (e.g. 'inner-loop')")
    args = ap.parse_args()

    pc_range = None
    if args.inner_loop_range:
        pc_range = parse_pc_range(args.inner_loop_range)

    opcodes, cats, total = categorise(args.sass, pc_range)
    lsu_sub = lsu_subtotals(opcodes)

    if total == 0:
        if pc_range:
            print(f"No instructions found in PC range "
                  f"0x{pc_range[0]:x}-0x{pc_range[1]:x}",
                  file=sys.stderr)
        else:
            print(f"No instructions found in {args.sass}", file=sys.stderr)
        return 1

    if args.csv:
        emit_csv(opcodes, cats, total, lsu_sub, label=args.label)
    else:
        emit_markdown(args.sass, opcodes, cats, total, lsu_sub, pc_range)

    return 0


if __name__ == "__main__":
    sys.exit(main())
