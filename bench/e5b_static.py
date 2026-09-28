"""E5b docs/03 §0 file set, completed from what each E5b directory already committed (2026-09-28 audit).

No GPU and no new measurement. Every number below is read from a committed file of the directory (the
`ncu/*.ncu-rep.zst` reports, their `*_details.csv`, the C1 capture the targets were picked from) or from a binary
on this device (`cuobjdump` on the CUDA libraries, the Triton cache), and the path it came from is recorded.

    python bench/e5b_static.py results/20260924_E5b_pi0_best [--venv ~/venvs/pi0]

Why a new script rather than re-running the capture-time tools:

  * **One file per SYMBOL, not per short name.** `bench/ncu_sass_dump.py` keys its dumps by the short kernel
    name and keeps the first launch it sees. pi0/best's `Kernel2` capture holds FOUR different
    `cutlass::Kernel2<...>` instantiations; the committed `raw/sass/Kernel2.sass` is the 256x64 TF32 one (1 of
    20 launches) while the two wmma bf16 ones that carry most of the time (9 + 9 launches) were never dumped.
    docs/03 §1: "distinct symbols are distinct targets". Here every distinct demangled name gets its own id.
  * **Backward branches.** ncu_sass_dump masks each PC to 32 bits but leaves BRA targets 64-bit, so
    `bench/sass_backward_bra.py` saw every target above its source and reported 0 backward branches for every
    dump (fixed there, 2026-09-28). The inner-loop ranges below use the fixed script.
  * **Per-PC execution counts.** ncu's source page carries `Instructions Executed` per SASS address. That is the
    "which PCs are hot" confirmation docs/03 §3 asks for; it picks the inner loop from measured counts.

Outputs per directory (docs/03 §0 names; `<k>` = kernel id, arch `sm87`):
  ncu/<tag>_raw.csv                 ncu --import --page raw --csv of each committed report
  ncu/parsed.json                   §11 item 2 schema, joined details + raw pages per symbol
  sass/targets.csv                  every symbol of every capture, §1 verdict, resources, library / cubin
  sass/<k>_sm87_total_inventory.*   §4 (moved from inventory/ when the committed dump was this symbol's)
  sass/<k>_sm87_inner_inventory.*   §4 over the inner-loop PC range (+ compute-to-memory ratio in the .md)
  sass/<k>_sm87_backward_bra.csv    §3
  sass/<k>_sm87_crosscheck.md       §8 (bench/crosscheck_sass.py --ncu-json ncu/parsed.json)
  sass/<k>_sm87_hotloop.sass.md     §5, deep targets only
  sass/<k>_sm87_pseudocode.md       §5, deep targets only
  sass/<k>_sm87_latency_budget.csv  §9, deep targets only
  raw/<k>_sm87.sass                 full SASS (under 1 MB each: protocol §6b keeps it uncompressed)
  raw/<k>_sm87.pc_exec.csv          per-PC Instructions Executed, mean per launch
  raw/<k>_sm87.res-usage.txt        cuobjdump -res-usage entries for the symbol
  raw/<k>_sm87.ptx.zst              Triton kernels only (vendor libraries ship no sm_8x PTX -- see not_produced)
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_inst_inventory import INST_RE, lsu_subclass, opcode_family  # noqa: E402
from crosscheck_sass import _short_name  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
NCU = "/usr/local/cuda/bin/ncu"
SCRATCH = REPO / "scratch" / "e5b_static"
ARCH = "sm87"
LAUNCH_GAP_US = 62.0          # bench/pick_e5b_targets.py (E5, pi0 eager median inter-kernel gap)
OWN_S = {"pi0": (51, "T007 / docs/01 E2: 50 actions + 1 state token"),
         "pi05": (50, "T008: S = 50 (results/20260925_E5b_pi05_ssweep/meta.json S_note)"),
         "smolvla": (50, "T008: S = 50 (results/20260925_E5b_smolvla_ssweep)"),
         "groot_n17": (41, "T009: S = 41 (results/20260925_E5b_groot_n17_ssweep)")}
CUDA_LIBS = ["/usr/local/cuda/lib64/libcublas.so.12", "/usr/local/cuda/lib64/libcublasLt.so.12"]
TRITON_CACHE = [Path("/tmp/triton_yanggon"), Path.home() / ".triton" / "cache"]

# docs/03 §9 table, verbatim. (lo, hi) cycles; None = the table gives no number.
LAT = {"LDG_DRAM": (400, 500), "LDG_L2": (150, 200), "LDG_L1": (30, 40), "LDS": (30, 30), "LDC": (1, 4),
       "STG": (0, 0), "FFMA": (4, 4), "FMUL": (4, 4), "FADD": (4, 4), "IMAD": (4, 6), "IADD3": (2, 2),
       "LOP3": (2, 2), "SHF": (2, 2), "ISETP": (2, 2), "SHFL": (5, 5), "BAR": (10, 20), "BRA": (2, 4),
       "MUFU": (16, 16)}
LAT_NOTE = {"STG": "fire-and-forget (no stall)", "BAR": "~10-20 + wait for other warps",
            "RED": "contention-dependent (no number)", "ATOM": "contention-dependent (no number)"}
LAT_SRC = "docs/03 §9 table ([ASSUMPTION]: documented for Ada / Blackwell with GDDR, order of magnitude only on Orin)"


def rnd(v, k=3):
    return round(v, k) if isinstance(v, float) else (v if v is not None else "not exported")


def sh(cmd, **kw) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, **kw).stdout


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def num(v):
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def args_stripped(name: str) -> str:
    """Join key: demangled name without namespace qualifiers and without its argument list.

    The raw page prints `void Kernel2<cfg>(Params)`, the source page and C1's details page
    `void cutlass::Kernel2<cfg>(T1::Params)` -- same symbol, different qualification."""
    n = name.strip().replace("(anonymous namespace)::", "").replace("::<unnamed>::", "::")
    n = n.replace(" <unnamed>::", " ").replace("<<unnamed>::", "<").replace("unnamed>::", "")  # raw page: `unnamed>::`
    n = re.sub(r"\b[A-Za-z_]\w*::", "", n)
    n = re.sub(r"\((?:unsigned |signed )?(?:int|long|long long|short|char|bool)\)(?=-?\d)", "", n)  # source page: (int)4
    depth = 0
    for i, ch in enumerate(n):
        depth += ch == "<"
        depth -= ch == ">"
        if ch == "(" and depth == 0:
            return n[:i].strip()
    return n


# ----------------------------------------------------------------------------------------- ncu exports

def rep_file(zst: Path, d: Path) -> Path:
    out = SCRATCH / "reps" / f"{d.name}__{zst.name[:-len('.ncu-rep.zst')]}.ncu-rep"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        subprocess.run(["zstd", "-dqf", str(zst), "-o", str(out)], check=True)
    return out


def raw_page(rep: Path, base: str | None = None, metrics: str | None = None) -> str:
    cmd = [NCU, "--import", str(rep), "--csv", "--page", "raw"]
    if base:
        cmd += ["--print-kernel-base", base]
    if metrics:
        cmd += ["--metrics", metrics]
    return sh(cmd, timeout=600)


def source_blocks(rep: Path) -> list[tuple[str, list[tuple[int, str, float]]]]:
    """[(kernel name, [(pc32, sass text, instructions executed)])] per printed block.

    ncu prints every launch twice on this build (checked: identical name and counts); consecutive
    duplicates are collapsed, and the caller checks the block count against the launch count."""
    txt = sh([NCU, "--import", str(rep), "--page", "source", "--print-source", "sass", "--csv"], timeout=900)
    blocks, cur, col = [], None, None
    for row in csv.reader(io.StringIO(txt)):
        if not row:
            continue
        if row[0] == "Kernel Name":
            cur = (row[1].strip(), [])
            blocks.append(cur)
        elif row[0] == "Address":
            col = row.index("Instructions Executed") if "Instructions Executed" in row else None
        elif cur is not None and re.fullmatch(r"0x[0-9a-fA-F]+", row[0]):
            ex = num(row[col]) if col is not None and col < len(row) else None
            cur[1].append((int(row[0], 16) & 0xFFFFFFFF, row[1].strip(), ex or 0.0))
    out = []
    for b in blocks:
        if out and out[-1][0] == b[0] and out[-1][1] == b[1] and out[-1][2]:
            out[-1] = (b[0], b[1], False)          # the duplicate print of the same launch
            continue
        out.append((b[0], b[1], True))
    return [(n, rows) for n, rows, _ in out]


# ----------------------------------------------------------------------------------------- libraries

def res_index(libs: list[str]) -> dict:
    """{mangled: [(lib, arch, 'REG:.. SHARED:..')]} over every ELF of each library (cuobjdump, CPU only)."""
    cache = SCRATCH / "res_index.json"
    idx = json.loads(cache.read_text()) if cache.exists() else {}
    changed = False
    for lib in libs:
        if lib in idx.get("_libs", []):
            continue
        arch = None
        lines = sh(["cuobjdump", "-res-usage", lib], timeout=900).splitlines()
        for i, ln in enumerate(lines):
            if ln.startswith("arch = "):
                arch = ln.split("=", 1)[1].strip()
            m = re.match(r"\s*Function (\S+):", ln)
            if m and i + 1 < len(lines):
                idx.setdefault(m.group(1), []).append([lib, arch, lines[i + 1].strip()])
        idx.setdefault("_libs", []).append(lib)
        changed = True
    if changed:
        cache.write_text(json.dumps(idx))
    return idx


def lib_info(lib: str) -> dict:
    p = Path(lib).resolve()
    ptx = sh(["cuobjdump", "--list-ptx", lib], timeout=600)
    arches = sorted(set(re.findall(r"\.(sm_\d+[a-z]?)\.ptx", ptx)))
    return {"path": lib, "resolved": str(p), "bytes": p.stat().st_size, "sha256": sha(p),
            "ptx_arches_shipped": arches or ["none"]}


def triton_match(name: str, sass_ops: list[str]) -> Path | None:
    """The Triton cache cubin whose SASS opcode sequence equals the one ncu recorded."""
    for root in TRITON_CACHE:
        if not root.exists():
            continue
        for cub in root.glob(f"*/{name}.cubin"):
            ops = [m.group(2) for m in (INST_RE.match(l) for l in sh(["cuobjdump", "-sass", str(cub)]).splitlines())
                   if m]
            if ops == sass_ops:
                return cub
    return None


# ----------------------------------------------------------------------------------------- SASS analysis

def ops_of(rows):
    return [m.group(2) for m in (INST_RE.match(f"        /*{pc:04x}*/        {t} ;") for pc, t, _ in rows) if m]


def write_sass(rows, path: Path):
    path.write_text("".join(f"        /*{pc:04x}*/{' ' * 8}{t} ;\n" for pc, t, _ in rows))


REG = re.compile(r"\b(U?R\d+|U?P\d)\b")
NO_DEST = {"ST", "STG", "STS", "STL", "RED", "REDG", "BRA", "BRX", "BAR", "EXIT", "NOP", "BSSY", "BSYNC", "WARPSYNC",
           "DEPBAR", "LDGSTS", "CALL", "RET", "JMP", "MEMBAR", "ERRBAR", "CCTL", "YIELD", "BMOV", "LDGDEPBAR", "ARRIVES"}
TWO_PRED = {"ISETP", "FSETP", "HSETP2", "DSETP", "PSETP", "PLOP3", "UISETP", "ISETP64"}


def split_ops(args: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in args:
        depth += ch == "["
        depth -= ch == "]"
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def regs(tok: str, width: int) -> list[str]:
    out = []
    for r in REG.findall(tok):
        if r.startswith(("R", "UR")) and width > 1 and not r.endswith("Z"):
            pre = "UR" if r.startswith("UR") else "R"
            n = int(r[len(pre):])
            out += [f"{pre}{n + i}" for i in range(width)]
        else:
            out.append(r)
    return out


def width_of(op: str) -> int:
    if ".128" in op or re.search(r"LDSM.*\.4$", op):
        return 4
    if ".64" in op or ".WIDE" in op or re.search(r"LDSM.*\.2$", op) or op.startswith(("DADD", "DMUL", "DFMA")):
        return 2
    if op.startswith("HMMA"):
        return 4 if ".F32" in op else 2
    return 1


def defuse(text: str):
    """(opcode, dest regs, source regs) -- a static approximation, labelled as such where it is used."""
    t = text.strip()
    guard = []
    m = re.match(r"@(!?)(U?P\w+)\s+", t)
    if m:
        guard = [m.group(2)]
        t = t[m.end():]
    parts = t.split(None, 1)
    op = parts[0]
    base = op.split(".")[0]
    operands = split_ops(parts[1]) if len(parts) > 1 else []
    w = width_of(op)
    if base in NO_DEST:
        dests, srcs = [], operands
    elif base in TWO_PRED:
        dests, srcs = operands[:2], operands[2:]
    else:
        dests, srcs = operands[:1], operands[1:]
    d = [r for tok in dests for r in regs(tok, w if tok.startswith(("R", "UR")) else 1)]
    s = list(guard)
    for i, tok in enumerate(srcs):
        if base == "HMMA":
            s += regs(tok, [4, 2, w][min(i, 2)])
        elif tok.startswith("["):
            s += regs(tok, 2 if ".64" in tok else 1)
        elif base in ("STG", "STS", "ST", "STL"):
            s += regs(tok, w)
        else:
            s += regs(tok, 1)
    return op, [r for r in d if r not in ("RZ", "PT", "URZ", "UPT")], [r for r in s if r not in ("RZ", "PT", "URZ", "UPT")]


def latency(op: str, variant: str):
    base = op.split(".")[0]
    if base in ("LDG", "LD"):
        return LAT[f"LDG_{variant}"], f"{LAT_SRC}; LDG {variant} variant"
    key = {"BAR": "BAR", "SHFL": "SHFL"}.get(base, base)
    if key in LAT:
        return LAT[key], LAT_SRC + (f"; {LAT_NOTE[key]}" if key in LAT_NOTE else "")
    if base in LAT_NOTE:
        return None, f"not documented: docs/03 §9 says {LAT_NOTE[base]}"
    return None, "not documented in docs/03 §9 (no latency given for this opcode)"


CORE = ("LDG", "LD", "LDS", "LDSM", "LDGSTS", "HMMA", "FFMA", "FMUL", "FADD", "HFMA2", "MUFU", "IMAD", "SHFL")


def longest_chain(rows, variant: str, by: str = "depth"):
    """Longest register def-use chain in one pass over the body (loop-carried edges ignored).

    by="depth": most dependent instructions; ties go to the chain with more loads / math (CORE), then more
    documented cycles -- §5's "longest chain in which each instruction waits for the previous one".
    by="cycles": most §9-documented cycles. An opcode §9 gives no latency for adds 0 cycles, so the two can differ;
    both are reported."""
    last, fin, pred = {}, [], []
    rank = (lambda f: (f[1], f[2], f[0])) if by == "depth" else (lambda f: (f[0], f[1], f[2]))
    for i, (pc, t, _) in enumerate(rows):
        op, d, s = defuse(t)
        lat, _ = latency(op, variant)
        ready, p = (0.0, 0, 0), None
        for r in s:
            if r in last and rank(fin[last[r]]) > rank(ready):
                ready, p = fin[last[r]], last[r]
        fin.append((ready[0] + (lat[0] if lat else 0.0), ready[1] + 1, ready[2] + (op.split(".")[0] in CORE)))
        pred.append(p)
        for r in d:
            last[r] = i
    if not fin:
        return [], 0.0
    end = max(range(len(fin)), key=lambda i: rank(fin[i]))
    chain = []
    while end is not None:
        chain.append(end)
        end = pred[end]
    return chain[::-1], fin[chain[-1]][0]


def algorithm(opcounts: collections.Counter, has_loop: bool) -> str:
    bases = collections.Counter()
    for op, c in opcounts.items():
        bases[op.split(".")[0]] += c
    if bases["HMMA"] and (bases["LDGSTS"] or bases["LDSM"] or bases["BAR"]):
        return "tensor-core GEMM (HMMA fed from shared-memory tiles staged per K-tile; §5 fingerprint)"
    if bases["FFMA"] >= 64 and bases["LDS"] and has_loop:
        return "tile-based SIMT GEMM (long backward BRA over K-tiles, LDS A/B tiles, unrolled FFMA; §5 fingerprint)"
    if bases["MUFU"] and bases["SHFL"] and bases["HMMA"]:
        return "FlashAttention-style (HMMA + MUFU + SHFL row reduce; §5 fingerprint)"
    if bases["SHFL"] or bases["REDUX"]:
        return "reduction kernel (SHFL / REDUX ladder)"
    return "elementwise / copy kernel (global load -> arithmetic -> global store, no K-loop)"


def annotate(t: str) -> str:
    op = t.split()[1] if t.startswith("@") and len(t.split()) > 1 else t.split()[0]
    base = op.split(".")[0]
    fam = opcode_family(op)
    sub = lsu_subclass(op)
    note = {"LDG": "global load (DRAM/L2: weights or activations)", "LDGSTS": "async global->shared copy (tile staging)",
            "LDS": "shared-memory load", "LDSM": "shared-memory matrix load (tensor-core operand)",
            "STG": "global store", "STS": "shared-memory store", "LDC": "constant-memory load",
            "HMMA": "tensor-core MMA", "BAR": "block barrier", "DEPBAR": "wait on async-copy group",
            "BRA": "branch", "SHFL": "warp shuffle", "MUFU": "special-function unit"}.get(base)
    return note or (f"{fam}" + (f" / {sub}" if sub else ""))


# ----------------------------------------------------------------------------------------- per directory

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir", type=Path)
    ap.add_argument("--venv", default=None, help="the venv whose libtorch_cuda.so ran the capture")
    a = ap.parse_args()
    d = a.rundir
    m = re.search(r"E5b_(\w+?)_(eager|best)$", d.name)
    model, mode = m.group(1), m.group(2)
    region = "ae_total" if model.startswith("groot") else "ae_step_0"
    venv = Path(a.venv or (Path.home() / "venvs" / ("groot" if model.startswith("groot") else "pi0")))
    torch_cuda = str(venv / "lib/python3.10/site-packages/torch/lib/libtorch_cuda.so")
    libs = CUDA_LIBS + [torch_cuda]
    fa = list((venv / "lib/python3.10/site-packages").glob("flash_attn_2_cuda*.so"))
    libs += [str(p) for p in fa]
    (d / "sass").mkdir(exist_ok=True)
    not_produced: dict[str, str] = {}
    notes: list[str] = []

    # ---- 1. exports and per-symbol join
    syms: dict[str, dict] = {}            # demangled (no args) -> info
    reps = sorted((d / "ncu").glob("*.ncu-rep.zst"))
    for z in reps:
        tag = z.name[:-len(".ncu-rep.zst")]
        rep = rep_file(z, d)
        raw_txt = raw_page(rep)
        out_raw = d / "ncu" / f"{tag}_raw.csv"
        out_raw.write_text(raw_txt)
        if out_raw.stat().st_size > 1_000_000:          # protocol §6b
            subprocess.run(["zstd", "-19", "-qf", str(out_raw), "-o", str(out_raw) + ".zst"], check=True)
            shutil.move(str(out_raw), SCRATCH / f"{d.name}__{out_raw.name}")
        dem = {r["ID"]: r["Kernel Name"] for r in csv.DictReader(io.StringIO(raw_page(rep, "demangled", "launch__grid_size")))
               if r.get("ID")}
        man = {r["ID"]: r["Kernel Name"] for r in csv.DictReader(io.StringIO(raw_page(rep, "mangled", "launch__grid_size")))
               if r.get("ID")}
        rows = [r for r in csv.DictReader(io.StringIO(raw_txt)) if r.get("ID")]
        det = collections.defaultdict(dict)
        detf = d / "ncu" / f"{tag}_details.csv"
        if detf.exists():
            for r in csv.DictReader(detf.open()):
                v = num(r.get("Metric Value"))
                if r.get("ID") and v is not None:
                    u = r.get("Metric Unit", "")
                    if r["Metric Name"] == "Duration":        # always us below
                        v, u = v * {"ms": 1e3, "msecond": 1e3, "ns": 1e-3, "nsecond": 1e-3}.get(u, 1.0), "us"
                    det[r["ID"]][r["Metric Name"]] = (v, u)
        for r in rows:
            full = dem.get(r["ID"], r["Kernel Name"])
            key = args_stripped(full)
            s = syms.setdefault(key, {"demangled": full, "mangled": man.get(r["ID"], ""), "short": _short_name(full)
                                      or _short_name(r["Kernel Name"]) or "kernel", "reports": set(), "launches": [],
                                      "sass": None})
            s["reports"].add(f"ncu/{z.name}")
            s["launches"].append({"raw": r, "det": det.get(r["ID"], {}), "report": tag})
        for name, blk in source_blocks(rep):
            key = args_stripped(name)
            if key not in syms:
                cand = [k for k in syms if k.startswith(key) or key.startswith(k)]
                key = cand[0] if len(cand) == 1 else key
            if key in syms:
                s = syms[key]
                s.setdefault("blocks", []).append(blk)
                # the source page prints fully qualified names (the raw page mangles `<unnamed>::`), so the short
                # name -- the one bench/ncu_sass_dump.py used for the committed files -- comes from here
                s["short"] = _short_name(name) or s["short"]
    # kernel ids
    by_short = collections.defaultdict(list)
    for k, s in syms.items():
        by_short[s["short"]].append(k)
    for short, keys in by_short.items():
        for k in keys:
            s = syms[k]
            if len(keys) == 1:
                s["kid"] = short
            else:
                cm = re.search(r"Kernel2<(\w+)>", s["demangled"])
                s["kid"] = f"{short}__{cm.group(1)}" if cm else f"{short}__{hashlib.sha1(s['mangled'].encode()).hexdigest()[:8]}"
            s["kid"] = re.sub(r"[^A-Za-z0-9_]", "_", s["kid"])

    # ---- 2. C1 shares (the capture the targets were picked from)
    c1 = REPO / f"results/20260923_E4g_c1_{model}_{mode}/ncu/{model}_{mode}_{region}_details.csv"
    c1dur, c1n, c1tot = collections.defaultdict(float), collections.Counter(), 0.0
    if c1.exists():
        seen = {}
        for r in csv.DictReader(c1.open()):
            if r.get("Metric Name") == "Duration":
                v = num(r["Metric Value"]) * {"ms": 1e3, "ns": 1e-3}.get(r.get("Metric Unit", "us"), 1.0)
                seen[r["ID"]] = (args_stripped(r["Kernel Name"]), v)
        for k, v in seen.values():
            c1dur[k] += v
            c1n[k] += 1
            c1tot += v

    # ---- 3. libraries
    idx = res_index(libs)
    libinfo = {}

    # ---- 4. per symbol
    parsed_run = {}
    trows = []
    moved = []
    for key, s in sorted(syms.items(), key=lambda kv: kv[1]["kid"]):
        kid = s["kid"]
        L = s["launches"]
        n = len(L)

        def mean_raw(col):
            vals = [num(x["raw"].get(col)) for x in L]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        def mean_det(metric):
            vals = [x["det"][metric][0] for x in L if metric in x["det"]]
            return sum(vals) / len(vals) if vals else None
        dur = mean_det("Duration")
        smt = mean_det("Compute (SM) Throughput")
        launch_bound = smt is not None and dur is not None and smt < 5.0 and dur < LAUNCH_GAP_US
        deep = not launch_bound
        pipes = {f"{p}_active_pct": mean_raw(f"sm__inst_executed_pipe_{p}.avg.pct_of_peak_sustained_active")
                 for p in ("lsu", "fma", "alu", "xu", "adu", "cbu")}
        pipes["fp16_active_pct"] = "not exported on this Orin (docs/03 §11: no such column)"
        tens = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
        pipes["tensor_hmma"] = {tens: mean_raw(tens)}
        pipes["lsu_elapsed_pct"] = mean_raw("sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed")
        pipes["xu_elapsed_pct"] = mean_raw("sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed")
        g = lambda c: mean_raw(f"l1tex__t_requests_pipe_lsu_mem_{c}.sum") or 0.0  # noqa: E731
        gld, gst, lld, lst = g("global_op_ld"), g("global_op_st"), g("local_op_ld"), g("local_op_st")
        sld = mean_raw("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum") or 0.0
        sst = mean_raw("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum") or 0.0
        loads, stores = gld + lld + sld, gst + lst + sst
        tot = loads + stores
        lsu = pipes["lsu_active_pct"] or 0.0
        split = {"note": "atomics/reductions: l1tex__t_requests_pipe_lsu_mem_global_op_{red,atom} and shared_op_atom "
                         "are not in these captures' raw page, so the split is over loads + stores only; shared counts "
                         "are wavefronts (the protocol §3.3 counter), global/local are requests",
                 "op_request_counts": {"global_op_ld": gld, "global_op_st": gst, "local_op_ld": lld,
                                       "local_op_st": lst, "shared_op_ld_wavefronts": sld,
                                       "shared_op_st_wavefronts": sst, "atomics": "not exported"}}
        if tot > 0:
            split.update({"load_pct": lsu * loads / tot, "store_pct": lsu * stores / tot, "atomic_pct": "not exported",
                          "write_pct": lsu * stores / tot,
                          "load_global_pct": lsu * gld / tot, "load_local_pct": lsu * lld / tot,
                          "load_shared_pct": lsu * sld / tot, "store_global_pct": lsu * gst / tot,
                          "store_shared_pct": lsu * sst / tot, "atomic_global_pct": "not exported",
                          "atomic_shared_pct": "not exported", "load_to_write_ratio": loads / max(1.0, stores)})
        stalls = {}
        for c in L[0]["raw"]:
            mm = re.fullmatch(r"smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio", c)
            if mm:
                stalls[mm.group(1)] = mean_raw(c)
        ssum = sum(v for v in stalls.values() if v)
        waiting = ssum - (stalls.get("selected") or 0)
        ls_share = 100 * (stalls.get("long_scoreboard") or 0) / waiting if waiting else None
        mp_share = 100 * (stalls.get("math_pipe_throttle") or 0) / waiting if waiting else None
        cls = []
        if ls_share is not None and ls_share >= 60:
            cls.append("memory-latency-bound (long_scoreboard >= 60 % of waiting, §7)")
        if mp_share is not None and mp_share > 10:
            cls.append("compute-bound region (math_pipe_throttle > 10 %, §7)")
        if launch_bound:
            cls.append("launch-bound (§1: SM throughput < 5 % and duration < the 62 us launch gap)")
        occ = {"theoretical_occupancy_pct": mean_det("Theoretical Occupancy"),
               "achieved_occupancy_pct": mean_det("Achieved Occupancy"),
               "achieved_active_warps_per_sm": mean_det("Achieved Active Warps Per SM"),
               "registers_per_thread": mean_raw("launch__registers_per_thread"),
               "shared_mem_per_block_kB": mean_raw("launch__shared_mem_per_block"),
               **{f"occupancy_limit_{x}": mean_raw(f"launch__occupancy_limit_{x}")
                  for x in ("blocks", "registers", "shared_mem", "warps")}}
        parsed_run[kid] = {
            "demangled": s["demangled"], "mangled": s["mangled"], "reports": sorted(s["reports"]),
            "duration_us": dur, "n_launches": n, "pipe_utilization": pipes, "pipe_lsu_split": split,
            "stalls": stalls, "stall_sum": ssum, "long_scoreboard_pct_of_waiting": ls_share,
            "occupancy": occ, "compute_sm_throughput_pct": smt,
            "memory_l2_throughput_pct": mean_det("Memory Throughput"), "l2_hit_rate_pct": mean_det("L2 Hit Rate"),
            "executed_ipc_active": mean_det("Executed Ipc Active"),
            "gpc_cycles_active_per_warp": "not measurable: gpc__cycles_active.avg and sm__warps_launched.sum are "
                                          "not exported (protocol §3.3)",
            "classification": "; ".join(cls) or "none of the §7 single-cause signatures"}

        # SASS for this symbol
        blocks = s.get("blocks") or []
        sass_path = d / "raw" / f"{kid}_{ARCH}.sass"
        pcx_path = d / "raw" / f"{kid}_{ARCH}.pc_exec.csv"
        rows = blocks[0] if blocks else None
        if rows:
            nb = len(blocks)
            write_sass(rows, sass_path)
            with pcx_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["pc", "sass", "inst_executed_mean_per_launch", "n_launch_blocks"])
                for i, (pc, t, _) in enumerate(rows):
                    w.writerow([f"0x{pc:x}", t, round(sum(b[i][2] for b in blocks if i < len(b)) / nb, 3), nb])
            if nb != n:
                notes.append(f"{kid}: {nb} source-page blocks after collapsing duplicate prints vs {n} launches on the "
                             f"raw page; per-PC means use the {nb} blocks")
            # the committed short-name dump, if it is THIS symbol's code
            old = d / "raw" / "sass" / f"{s['short']}.sass"
            if old.exists() and old.read_text() == sass_path.read_text():
                moved.append((old, sass_path))
                for suf, new in (("_inventory.csv", "_total_inventory.csv"), ("_inventory.md", "_total_inventory.md"),
                                 ("_crosscheck.md", "_crosscheck.md")):
                    oi = d / "inventory" / f"{s['short']}{suf}"
                    if oi.exists():
                        moved.append((oi, d / "sass" / f"{kid}_{ARCH}{new}"))
        else:
            not_produced[f"raw/{kid}_{ARCH}.sass"] = "ncu's source page printed no SASS for this symbol"
        # library / cubin
        cands = [c for c in idx.get(s["mangled"], []) if c[0] in libs]    # only the libraries this run loaded
        regs_run = occ["registers_per_thread"]
        match = [c for c in cands if f"REG:{int(regs_run)} " in c[2] + " "] if regs_run else []
        lib = ""
        if s["short"].startswith("triton_") and rows:
            cub = triton_match(s["short"], ops_of(rows))
            if cub:
                dst = d / "raw" / "cubin" / f"{kid}.cubin"
                dst.parent.mkdir(exist_ok=True)
                if not dst.exists():
                    shutil.copy(cub, dst)
                lib = f"raw/cubin/{kid}.cubin (Triton cache {cub.parent.name}, SASS opcode sequence identical to ncu's)"
                (d / "raw" / f"{kid}_{ARCH}.res-usage.txt").write_text(sh(["cuobjdump", "-res-usage", str(dst)]))
                for extra in (".ptx", ".source", ".ttgir", ".json"):
                    src = cub.with_suffix(extra)
                    if src.exists() and extra == ".ptx":
                        subprocess.run(["zstd", "-19", "-qf", str(src), "-o", str(d / "raw" / f"{kid}_{ARCH}.ptx.zst")],
                                       check=True)
                    elif src.exists():
                        shutil.copy(src, d / "raw" / "cubin" / f"{kid}{extra}")
            else:
                lib = "Triton kernel; no cache cubin with an identical SASS sequence found"
                not_produced[f"raw/{kid}_{ARCH}.ptx.zst"] = lib
                not_produced[f"raw/{kid}_{ARCH}.res-usage.txt"] = lib
        elif cands:
            libs_hit = sorted({c[0] for c in cands})
            lib = "; ".join(libs_hit)
            for l_ in libs_hit:
                libinfo.setdefault(l_, lib_info(l_))
            with (d / "raw" / f"{kid}_{ARCH}.res-usage.txt").open("w") as f:
                f.write(f"# cuobjdump -res-usage <lib>, every ELF entry of {s['mangled']}\n")
                f.write(f"# ncu launch__registers_per_thread for the executed code: {regs_run}\n")
                for c in cands:
                    f.write(f"{c[0]}  arch={c[1]}  {c[2]}" + ("   <- REG matches the executed code" if c in match else "")
                            + "\n")
            arches = sorted({c[1] for c in cands})
            ptx = sorted({a_ for l_ in libs_hit for a_ in libinfo[l_]["ptx_arches_shipped"]})
            ptx_txt = "no PTX at all" if ptx == ["none"] else f"PTX only for {[x for x in ptx if x != 'none']}"
            not_produced[f"raw/{kid}_{ARCH}.ptx.zst"] = (
                f"vendor library ({lib}) ships SASS for {arches} and {ptx_txt}; no PTX exists for the arch that ran "
                f"(sm_87 / sm_80-compatible), and dumping a different arch's PTX would not be this kernel")
            not_produced[f"raw/cubin/{kid}.cubin"] = "vendor library kernel: the .so is recorded (path, sha256) instead, per §0"
            if not match:
                notes.append(f"{kid}: no library ELF entry has REG = {regs_run} (the executed code's register count); "
                             f"entries listed in raw/{kid}_{ARCH}.res-usage.txt")
        else:
            lib = "not found in " + ", ".join(Path(x).name for x in libs)
            not_produced[f"raw/{kid}_{ARCH}.res-usage.txt"] = f"symbol {s['mangled'][:80]} not in any searched library"
            not_produced[f"raw/{kid}_{ARCH}.ptx.zst"] = "binary not located"

        # C1 share
        c1k = args_stripped(s["demangled"])
        share = 100 * c1dur[c1k] / c1tot if c1tot and c1k in c1dur else None
        g0 = L[0]["raw"]
        trows.append({
            "kernel_id": kid, "kernel_short": s["short"], "mangled_symbol": s["mangled"], "demangled": s["demangled"],
            "library_or_cubin": lib, "arch": "sm_87 (CC 8.7, ncu raw page)", "region": region,
            "role": ("§1 deep target" if deep else "launch-bound (§1): §2-§4 only"),
            "S": OWN_S[model][0],
            "share_of_ae_step_kernel_time": round(share, 2) if share is not None else
            f"absent from the C1 capture ({c1.name})",
            "launches_per_step": c1n[c1k] if c1k in c1n else f"absent from the C1 capture",
            "grid": f"{g0.get('launch__grid_dim_x')}x{g0.get('launch__grid_dim_y')}x{g0.get('launch__grid_dim_z')}",
            "block": f"{g0.get('launch__block_dim_x')}x{g0.get('launch__block_dim_y')}x{g0.get('launch__block_dim_z')}",
            "regs_per_thread": regs_run, "shared_mem_per_block_kB": rnd(occ["shared_mem_per_block_kB"]),
            "theoretical_occupancy_pct": rnd(occ["theoretical_occupancy_pct"]),
            "achieved_occupancy_pct": rnd(occ["achieved_occupancy_pct"]), "e5b_launches": n,
            "e5b_mean_duration_us": round(dur, 3) if dur is not None else "not in details page",
            "e5b_sm_throughput_pct": round(smt, 2) if smt is not None else "not in details page"})
        s["deep"] = deep
        s["rows"] = rows

    # moves of the committed short-name files that turned out to be one symbol's
    # The committed short-name files are superseded, not deleted blindly: the .sass is byte-identical to
    # raw/<k>_sm87.sass; the inventories are regenerated at the §0 path from that same dump (checked equal below,
    # only the title's file name differs); the cross-check is regenerated per symbol (the old one mixed symbols).
    (SCRATCH / "moved").mkdir(parents=True, exist_ok=True)
    for old, _new in moved:
        shutil.copy(old, SCRATCH / "moved" / f"{d.name}__{old.parent.name}__{old.name}")

    json.dump({d.name: parsed_run}, (d / "ncu" / "parsed.json").open("w"), indent=1, default=str)
    with (d / "sass" / "targets.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trows[0]))
        w.writeheader()
        w.writerows(trows)

    # ---- 5. per-symbol static files
    for key, s in syms.items():
        kid, rows = s["kid"], s["rows"]
        if not rows:
            continue
        sp = d / "raw" / f"{kid}_{ARCH}.sass"
        pre = d / "sass" / f"{kid}_{ARCH}"
        py = sys.executable
        for fmt, suf in (("--csv", "_total_inventory.csv"), (None, "_total_inventory.md")):
            if not Path(f"{pre}{suf}").exists():
                out = sh([py, str(REPO / "bench/build_inst_inventory.py"), str(sp)] + ([fmt, "--label", kid] if fmt else []))
                Path(f"{pre}{suf}").write_text(out)
        subprocess.run([py, str(REPO / "bench/sass_backward_bra.py"), str(sp), "--out", f"{pre}_backward_bra.csv"],
                       capture_output=True, check=True)
        Path(f"{pre}_crosscheck.md").write_text(sh(
            [py, str(REPO / "bench/crosscheck_sass.py"), "--inventory", f"{pre}_total_inventory.csv", "--ncu-json",
             str(d / "ncu" / "parsed.json"), "--target-kernel", kid, "--gpu-arch", "sm_87"]))
        # inner loop: the backward branch whose range executes the most instructions (per-PC counts, §3)
        bras = list(csv.DictReader(open(f"{pre}_backward_bra.csv")))
        pcs = [pc for pc, _, _ in rows]
        exe = {pc: e for pc, _, e in rows}
        best, how = None, ""
        for b in bras:
            lo, hi = int(b["target_pc"], 16), int(b["source_pc"], 16)
            ex = sum(exe[pc] for pc in pcs if lo <= pc <= hi)
            if best is None or ex > best[2]:
                best = (lo, hi, ex)
        if best and best[2] > 0:
            lo, hi = best[0], best[1]
            how = (f"backward BRA 0x{hi:x} -> 0x{lo:x}: of the {len(bras)} backward branches, the range whose "
                   f"instructions executed most (ncu per-PC Instructions Executed, §3)")
        elif best:
            lo, hi = best[0], best[1]
            how = "backward BRA with the largest range (no per-PC counts available)"
        else:
            lo, hi = pcs[0], pcs[-1]
            how = "no backward BRA: fully unrolled, the inner range is the whole kernel (§3)"
        rng = f"0x{lo:x}-0x{hi:x}"
        inv_csv = sh([py, str(REPO / "bench/build_inst_inventory.py"), str(sp), "--csv", "--inner-loop-range", rng,
                      "--label", f"{kid} inner loop {rng}"])
        Path(f"{pre}_inner_inventory.csv").write_text(f"# inner loop PC range {rng}: {how}\n" + inv_csv)
        inv_md = sh([py, str(REPO / "bench/build_inst_inventory.py"), str(sp), "--inner-loop-range", rng])
        body = [(pc, t, e) for pc, t, e in rows if lo <= pc <= hi]
        oc = collections.Counter(m_.group(2) for m_ in (INST_RE.match(f"        /*{pc:04x}*/        {t} ;")
                                                         for pc, t, _ in body) if m_)
        fam = collections.Counter()
        for op, c in oc.items():
            fam[opcode_family(op)] += c
        gmem = sum(c for op, c in oc.items() if op.split(".")[0] in ("LDG", "STG", "LDGSTS", "LD", "ST"))
        fp = sum(c for op, c in oc.items() if op.split(".")[0] in (
            "FFMA", "FMUL", "FADD", "HFMA2", "HMUL2", "HADD2", "HMMA", "DFMA", "DMUL", "DADD", "MUFU"))
        ldn = sum(c for op, c in oc.items() if lsu_subclass(op) == "Load")
        stn = sum(c for op, c in oc.items() if lsu_subclass(op) == "Store")
        Path(f"{pre}_inner_inventory.md").write_text(
            f"<!-- inner loop: {how} -->\n" + inv_md +
            f"\n## Ratios (§4)\n\n* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = {fp} ÷ {gmem} = "
            f"{(round(fp / gmem, 3)) if gmem else 'n/a (no global memory op in range)'}. Integer arithmetic is left out: statically it cannot be told apart from "
            f"address arithmetic, so the ratio is a lower bound on §4's definition.\n"
            f"* static load : store = {ldn} : {stn}\n")
        if not s["deep"]:
            for suf in ("_hotloop.sass.md", "_pseudocode.md", "_latency_budget.csv"):
                not_produced[f"sass/{kid}_{ARCH}{suf}"] = ("launch-bound under docs/03 §1 (SM throughput < 5 % and "
                                                         "duration < 62 us): §5-§9 are not run on it")
            continue
        # hot loop excerpt
        hl = [f"# Hot loop — `{kid}` ({ARCH})", "",
              f"PC range {rng}; {how}. Full dump: `raw/{kid}_{ARCH}.sass`; per-PC counts: `raw/{kid}_{ARCH}.pc_exec.csv`.",
              "", "**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, "
              "for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs "
              "activations) is not in the SASS; see the phase notes in the pseudocode file.", "", "```"]
        shown = body[:200]
        for pc, t, e in shown:
            hl.append(f"/*{pc:08x}*/  {t:<58s} ; // x{e:>10.0f}  {annotate(t)}")
        hl.append("```")
        if len(body) > 200:
            hl.append(f"\n(first 200 of {len(body)} instructions shown; the rest are in the full dump)")
        Path(f"{pre}_hotloop.sass.md").write_text("\n".join(hl) + "\n")
        # dependency chains and the budget
        brows = []
        chains = {}
        for var in ("DRAM", "L2"):
            ch, _ = longest_chain(body, var, "depth")
            chains[var] = set(ch)
        cyc_chain = set(longest_chain(body, "DRAM", "cycles")[0])
        tots = {}
        for var in ("DRAM", "L2"):
            lo_, hi_, und = 0, 0, 0
            for i in sorted(chains[var]):
                op = defuse(body[i][1])[0]
                lat, _ = latency(op, var)
                if lat:
                    lo_, hi_ = lo_ + lat[0], hi_ + lat[1]
                else:
                    und += 1
            tots[var] = (lo_, hi_, und, len(chains[var]))
        for i, (pc, t, e) in enumerate(body):
            op = defuse(t)[0]
            lat, src = latency(op, "DRAM")
            lat2, _ = latency(op, "L2")
            brows.append({"instruction": f"0x{pc:x}  {t}", "role": annotate(t), "on_chain": i in chains["DRAM"],
                          "on_max_documented_cycles_chain": i in cyc_chain,
                          "cycles_assumed": f"{lat[0]}-{lat[1]}" if lat else "not documented",
                          "cycles_assumed_l2_variant": f"{lat2[0]}-{lat2[1]}" if lat2 else "not documented",
                          "source_of_cycles": src, "inst_executed_mean_per_launch": e})
        g0 = s["launches"][0]["raw"]
        warps = (num(g0.get("launch__grid_size")) or 0) * -(-(num(g0.get("launch__block_size")) or 0) // 32)
        back_ex = exe.get(hi, 0.0) if bras else None
        iters = (back_ex / warps) if (back_ex and warps) else None
        with open(f"{pre}_latency_budget.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(brows[0]))
            w.writeheader()
            w.writerows(brows)
            f.write("\n# summary (docs/03 §9)\n")
            for var in ("DRAM", "L2"):
                lo_, hi_, und, nch = tots[var]
                f.write(f"serial_cycles_per_iteration_{var}_variant,{lo_}-{hi_},deepest def-use chain ({nch} "
                        f"instructions, column on_chain); {und} of them have no documented latency in §9 and add an "
                        f"unknown amount, so this is a lower bound\n")
            clo = sum(latency(defuse(body[i][1])[0], "DRAM")[0][0] for i in cyc_chain
                      if latency(defuse(body[i][1])[0], "DRAM")[0])
            chi = sum(latency(defuse(body[i][1])[0], "DRAM")[0][1] for i in cyc_chain
                      if latency(defuse(body[i][1])[0], "DRAM")[0])
            f.write(f"max_documented_cycles_chain_DRAM_variant,{clo}-{chi},{len(cyc_chain)} instructions "
                    f"(column on_max_documented_cycles_chain)\n")
            f.write("measured_cycles_per_iteration,not measurable: gpc__cycles_active.avg and sm__warps_launched.sum "
                    "are not exported on this Orin (protocol §3.3)\n")
            f.write("overlap_ratio_serial_over_measured,not computable without the measured cycles\n")
            f.write(f"achieved_active_warps_per_sm,{parsed_run[kid]['occupancy']['achieved_active_warps_per_sm']}\n")
            f.write(f"inner_loop_iterations_per_warp_derived,{iters if iters is not None else 'no back-edge'}"
                    f"{',back-edge BRA executions per launch / warps per launch (grid x ceil(block/32))' if iters else ''}\n")
            f.write("chain_method,static register def-use over one pass of the body (loop-carried edges ignored; "
                    "vector widths from opcode modifiers) -- an approximation, bench/e5b_static.py longest_chain\n")
        # pseudocode
        pro = [(pc, t, e) for pc, t, e in rows if pc < lo]
        epi = [(pc, t, e) for pc, t, e in rows if pc > hi]

        def famsum(part):
            c = collections.Counter(opcode_family(defuse(t)[0]) for _, t, _ in part)
            return ", ".join(f"{k} {v}" for k, v in c.most_common())

        def bcount(part, *names):
            return sum(1 for _, t, _ in part if defuse(t)[0].split(".")[0] in names)
        algo = algorithm(collections.Counter(defuse(t)[0] for _, t, _ in rows), bool(bras))
        chain_ops = " -> ".join(defuse(body[i][1])[0] for i in sorted(chains["DRAM"]))[:600]
        pcode = [
            f"# Pseudocode — `{kid}` ({ARCH})", "",
            f"**Mechanical decompilation of the committed SASS** (`raw/{kid}_{ARCH}.sass`, bench/e5b_static.py): phases "
            f"are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each "
            f"range. It is a reading aid; it does not compile and names no source variable the SASS does not show.",
            "", f"Algorithm (§5 fingerprints): **{algo}**.", "",
            f"Demangled: `{s['demangled'][:300]}`", "", "```c",
            f"// === Phase: setup + pre-loop (PC 0x{pcs[0]:x}-0x{(pro[-1][0] if pro else pcs[0]):x}), "
            f"{len(pro)} instructions: {famsum(pro)}",
            f"//     thread/block ids (S2R {bcount(pro, 'S2R', 'S2UR', 'CS2R')}), constant-memory params "
            f"(LDC/ULDC {bcount(pro, 'LDC', 'ULDC', 'LDCU')}), bounds checks (ISETP {bcount(pro, 'ISETP')}), "
            f"early EXIT {bcount(pro, 'EXIT')}, first loads (LDG {bcount(pro, 'LDG')}, LDGSTS {bcount(pro, 'LDGSTS')})",
            "setup_indices_from_blockIdx_threadIdx();",
            f"for (iter = 0; iter < n_iter; ++iter) {{   // === Phase: inner loop (PC {rng}), {len(body)} instructions",
            f"    // global loads LDG {bcount(body, 'LDG')}, async tile copies LDGSTS {bcount(body, 'LDGSTS')}, "
            f"shared loads LDS {bcount(body, 'LDS')} / LDSM {bcount(body, 'LDSM')}, shared stores STS {bcount(body, 'STS')}",
            f"    // math: HMMA {bcount(body, 'HMMA')}, FFMA {bcount(body, 'FFMA')}, HFMA2 {bcount(body, 'HFMA2')}, "
            f"MUFU {bcount(body, 'MUFU')}; sync: BAR {bcount(body, 'BAR')}, DEPBAR {bcount(body, 'DEPBAR')}",
            "    stage_next_tile_and_accumulate();",
            f"}}   // back-edge at 0x{hi:x}" + (f"; {iters:.2f} iterations per warp per launch (derived)" if iters else ""),
            f"// === Phase: epilogue (PC 0x{(epi[0][0] if epi else hi):x}-0x{pcs[-1]:x}), {len(epi)} instructions: "
            f"{famsum(epi)}",
            f"//     reduction SHFL {bcount(epi, 'SHFL')}, BAR {bcount(epi, 'BAR')}; output STG {bcount(epi, 'STG')}, "
            f"EXIT {bcount(epi, 'EXIT')}",
            "write_output_tile();", "```", "",
            "## Longest dependency chain in one iteration (static register def-use, deepest)", "",
            f"`{chain_ops}`", "",
            f"{len(chains['DRAM'])} dependent instructions; cycles in `sass/{kid}_{ARCH}_latency_budget.csv` (which also "
            f"gives the chain with the most §9-documented cycles). "
            "Loads not on the chain can be in flight together (memory-level parallelism).", "",
            "```", "one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]",
            "```"]
        Path(f"{pre}_pseudocode.md").write_text("\n".join(pcode) + "\n")

    # superseded committed files: verify, then remove from the tree (copies kept in scratch/e5b_static/moved)
    for old, new in moved:
        if old.name.endswith("_inventory.csv"):
            body = lambda p: [l for l in p.read_text().splitlines() if not l.startswith("#")]  # noqa: E731
            if body(old) != body(new):
                notes.append(f"{old.relative_to(d)} differs from regenerated {new.relative_to(d)}; kept both")
                continue
        old.unlink()
    for p in (d / "raw" / "sass", d / "inventory"):
        if p.exists() and not any(p.iterdir()):
            p.rmdir()
        elif p.exists():
            notes.append(f"{p.relative_to(d)}/ still holds {sorted(x.name for x in p.iterdir())}: not matched to a symbol")

    # ---- 6. meta.json
    mp = d / "meta.json"
    meta = json.loads(mp.read_text())
    meta["gpu_arch"] = "sm_87 (compute capability 8.7 from the ncu raw page's CC column of every committed report)"
    meta["cuobjdump_version"] = sh(["cuobjdump", "--version"]).strip().splitlines()[-1]
    meta["libraries"] = list(libinfo.values()) or ["no vendor-library symbol in this directory's captures"]
    meta["e5b_static"] = {
        "script": "bench/e5b_static.py", "command": f"python bench/e5b_static.py {d}",
        "what": "docs/03 §0 file set completed on 2026-09-28 from this directory's committed ncu reports (CPU only)",
        "symbols": len(syms), "deep_targets": sum(1 for s in syms.values() if s["deep"]),
        "moved": [f"{o.relative_to(d)} -> {n.relative_to(d)}" for o, n in moved],
        "notes": notes or ["none"],
        "corrections": [
            "sass_backward_bra.py compared 32-bit-masked PCs with 64-bit BRA targets and so reported 0 backward branches "
            "for every ncu-exported dump; fixed 2026-09-28 and the backward_bra.csv files here use the fix",
            "ncu_sass_dump.py kept one dump per SHORT name; distinct instantiations (e.g. several cutlass::Kernel2<...>) "
            "now each have raw/<k>_sm87.sass; the former inventory/<short>_crosscheck.md averaged the dynamic metrics of "
            "every instantiation sharing the short name and is replaced by per-symbol sass/<k>_sm87_crosscheck.md"]}
    meta["not_produced"] = not_produced or {"none": "every §0 file was produced"}
    files = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.name != "meta.json":
            files.append({"path": str(p.relative_to(d)), "bytes": p.stat().st_size, "sha256": sha(p)})
    meta["raw_files"] = files
    meta["raw_files_note"] = "2026-09-28: re-listed after the §0 completion (bench/e5b_static.py)"
    mp.write_text(json.dumps(meta, indent=1, default=str))
    print(f"{d.name}: {len(syms)} symbols ({sum(1 for s in syms.values() if s['deep'])} deep), "
          f"{len(not_produced)} not_produced entries, {len(moved)} files moved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
