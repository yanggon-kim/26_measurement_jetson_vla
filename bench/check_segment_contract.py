"""Every key a segment builder reads must be one its intermediates() returns.

This contract broke twice in one morning and cost two preflight runs, ~20 minutes each:

    KeyError: 'scores'   host_segments.py:681   (T read from a score matrix SmolVLA
                                                 no longer reconstructs)
    KeyError: 'mask'     segments_smolvla.py:238 (left over from the removed softmax
                                                  segment)

Both were the same mistake: dropping a tensor from `intermediates()` without following
every consumer. Neither needs a GPU to detect — the reads and the returns are both
visible in the source — so this runs in the preflight before anything loads a model.

It is deliberately a SOURCE check rather than a runtime one. A runtime check would only
fire on the code path that happened to execute, and the whole problem is that these
builders have per-model branches which do not all run every time.

Known false positive, asserted here rather than special-cased silently: `_partials` is
injected by the CALLER (`t["_partials"] = torch.randn(...)` in host_segments.main), not
returned by intermediates. Anything else missing is a real break.

    python bench/check_segment_contract.py
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# keys supplied by the caller after intermediates() returns
INJECTED = {"_partials"}


def keys_read(fn) -> set[str]:
    src = inspect.getsource(fn)
    return (set(re.findall(r't\["(\w+)"\]', src))
            | set(re.findall(r't\.get\("(\w+)"', src)))


def keys_returned(fn) -> set[str]:
    src = inspect.getsource(fn)
    if "return {" not in src:
        return set()
    return set(re.findall(r'"(\w+)":', src.split("return {")[-1]))


def dead_params(mod, names) -> list[str]:
    """Parameters a function declares and never reads, found with AST rather than regex.

    The `last_hidden` argument was threaded through build_extras and measure_extras after
    the final-norm fix stopped using it, and the only thing still producing it -- a
    wrapper on layers[-1].forward -- is never called for SmolVLA, whose layer body is
    inlined. The result was a KeyError AFTER the whole measurement had succeeded. A
    parameter nothing reads is the signature of exactly that: a producer kept alive for a
    consumer that no longer exists.

    The first version of this check used a regex on the source text and reported `policy`
    and `batch` as unused because its way of skipping the signature threw away the first
    line of the body, where they are used. A check that cries wolf is worse than none, so
    it parses now.
    """
    import ast
    import inspect
    import textwrap
    out = []
    for name in names:
        fn = getattr(mod, name, None)
        if fn is None:
            continue
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        fdef = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == name)
        params = {a.arg for a in fdef.args.args} - {"self"}
        # every Name loaded anywhere in the body, nested functions included
        loaded = {n.id for n in ast.walk(fdef)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        for p_ in sorted(params - loaded):
            out.append(f"{name}({p_})")
    return out


def main() -> int:
    import host_segments as H
    import segments_groot as G
    import segments_smolvla as S

    cases = [
        ("host_segments (pi0 / pi0.5)", H.intermediates, H.build_segments),
        ("segments_smolvla", S.intermediates, S.build_segments),
        ("segments_groot", G.intermediates, G.build_segments),
    ]
    bad = 0
    for name, ifn, bfn in cases:
        reads, ret = keys_read(bfn), keys_returned(ifn)
        missing = sorted(reads - ret - INJECTED)
        status = "OK" if not missing else f"MISSING {missing}"
        print(f"  {name:30s} reads={len(reads):3d} returns={len(ret):3d}  {status}")
        if missing:
            bad += 1
    dead = dead_params(H, ("build_extras", "measure_extras", "finish"))
    if dead:
        print(f"  UNUSED PARAMETERS (a producer kept alive for a dead consumer): {dead}")
        bad += 1
    else:
        print(f"  {'host_segments extras':30s} no unused parameters")

    if bad:
        print(f"\n{bad} segment module(s) read a key their intermediates() does not "
              f"return. That is a KeyError waiting to happen on the first model whose "
              f"branch reaches it — fix before running the matrix.")
        return 1
    print("\nall segment builders read only keys their intermediates() returns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
