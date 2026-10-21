"""E12 Step 3: can CUDA green contexts partition Orin's 16 SMs, in one process?

Why this is the most valuable remaining Step 3 mechanism. E12's MAXN result is that the two
stages do not overlap because the HOST issues for 100 % of the wall clock (issue/wall =
1.0000), so GPU contention -- KerColle's actual mechanism -- was never exercised. The routes
that would exercise it have each been closed:

  * two CUDA streams in one process: one Python thread cannot feed both (§13.7 forbids a
    second, T010's numbers were GIL artefacts), and the compiled rung corrupts across streams;
  * two processes: OOM-killed. Orin's 62 GB is shared CPU+GPU and one pi0 at bf16_both is
    ~15 GB, so two do not fit (results/20260925_E12_pi0_2proc_N10_MAXN, EXIT=137).

Green contexts partition SMs WITHIN one process. That sidesteps the memory limit entirely --
one model, one interpreter -- and it lets the question be asked from the other side: instead
of "do two stages slow each other down", ask "how does ONE stage's time change when it is
given 8 SMs instead of 16". If a stage is insensitive to losing half the SMs, it was not
using them, and a co-runner could have had them. That is a contention answer obtained without
ever running two things at once.

This file only PROBES. It resolves the symbols, performs a real split on this device, and
reports the partition sizes the driver actually grants -- because "the symbol is present in
libcuda.so.1" (already established) is not the same as "sm_87 supports the split", and the
difference must be measured, not assumed.

Driver API used (CUDA 12.4+), all via ctypes against libcuda.so.1:
    cuDeviceGetDevResource(dev, &res, CU_DEV_RESOURCE_TYPE_SM)
    cuDevSmResourceSplitByCount(&groups, &nbGroups, &input, &remaining, flags, minCount)
    cuDevResourceGenerateDesc(&desc, resources, nbResources)
    cuGreenCtxCreate(&gctx, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM)
"""
from __future__ import annotations

import ctypes
import json
from pathlib import Path

CU_DEV_RESOURCE_TYPE_SM = 1
CU_GREEN_CTX_DEFAULT_STREAM = 1


class CUdevSmResource(ctypes.Structure):
    _fields_ = [("smCount", ctypes.c_uint)]


class CUdevResource(ctypes.Structure):
    """cuda.h: type, 92 bytes internal padding, then a 48-byte union. 144 bytes total.

    The padding is part of the ABI, not slack: getting it wrong makes smCount read garbage,
    which would look like a successful probe returning a wrong SM count.
    """
    _fields_ = [("type", ctypes.c_int),
                ("_internal_padding", ctypes.c_ubyte * 92),
                ("sm", CUdevSmResource),
                ("_union_pad", ctypes.c_ubyte * 44)]


def main() -> int:
    lib = ctypes.CDLL("libcuda.so.1")
    out: dict = {"device": None, "sm_total": None, "splits": [], "notes": []}

    def chk(rc: int, what: str) -> bool:
        if rc == 0:
            return True
        name = ctypes.c_char_p()
        lib.cuGetErrorName(rc, ctypes.byref(name))
        s = name.value.decode() if name.value else f"rc={rc}"
        out["notes"].append(f"{what}: {s}")
        print(f"  {what}: FAILED {s}")
        return False

    if not chk(lib.cuInit(0), "cuInit"):
        return 1
    dev = ctypes.c_int()
    if not chk(lib.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet"):
        return 1
    name = ctypes.create_string_buffer(256)
    lib.cuDeviceGetName(name, 256, dev)
    out["device"] = name.value.decode()
    print(f"  device: {out['device']}")

    for sym in ("cuDeviceGetDevResource", "cuDevSmResourceSplitByCount",
                "cuDevResourceGenerateDesc", "cuGreenCtxCreate"):
        try:
            getattr(lib, sym)
        except AttributeError:
            print(f"  {sym}: ABSENT from libcuda.so.1 -- green contexts unavailable")
            out["notes"].append(f"{sym} absent")
            Path("scratch/green_ctx_probe.json").write_text(json.dumps(out, indent=2) + "\n")
            return 2
    print("  all four green-context symbols resolve")

    whole = CUdevResource()
    if not chk(lib.cuDeviceGetDevResource(dev, ctypes.byref(whole),
                                          CU_DEV_RESOURCE_TYPE_SM),
               "cuDeviceGetDevResource(SM)"):
        return 1
    out["sm_total"] = whole.sm.smCount
    print(f"  whole-device SM resource: smCount = {whole.sm.smCount}")

    # Try the partitions E12's spec asks about. minCount is a MINIMUM: the driver rounds to
    # what the hardware can actually give, which on Orin may be a multiple of the TPC size.
    for min_count in (1, 2, 4, 6, 8, 10, 12):
        groups = (CUdevResource * 16)()
        nb = ctypes.c_uint(16)
        remaining = CUdevResource()
        inp = CUdevResource()
        lib.cuDeviceGetDevResource(dev, ctypes.byref(inp), CU_DEV_RESOURCE_TYPE_SM)
        rc = lib.cuDevSmResourceSplitByCount(
            groups, ctypes.byref(nb), ctypes.byref(inp), ctypes.byref(remaining),
            ctypes.c_uint(0), ctypes.c_uint(min_count))
        if rc != 0:
            err = ctypes.c_char_p()
            lib.cuGetErrorName(rc, ctypes.byref(err))
            e = err.value.decode() if err.value else f"rc={rc}"
            print(f"  minCount={min_count:2d}: split FAILED {e}")
            out["splits"].append({"min_count": min_count, "ok": False, "error": e})
            continue
        sizes = [groups[i].sm.smCount for i in range(nb.value)]
        rem = remaining.sm.smCount
        print(f"  minCount={min_count:2d}: {nb.value} group(s) of {sizes}, "
              f"remaining {rem}")
        rec = {"min_count": min_count, "ok": True, "n_groups": nb.value,
               "group_sm_counts": sizes, "remaining_sm": rem, "green_ctx_created": False}

        # Creating a context is the real test: a split that describes a partition the driver
        # will not instantiate is not a usable mechanism.
        desc = ctypes.c_void_p()
        rc = lib.cuDevResourceGenerateDesc(ctypes.byref(desc), groups, ctypes.c_uint(1))
        if rc == 0:
            gctx = ctypes.c_void_p()
            rc = lib.cuGreenCtxCreate(ctypes.byref(gctx), desc, dev,
                                      ctypes.c_uint(CU_GREEN_CTX_DEFAULT_STREAM))
            if rc == 0:
                rec["green_ctx_created"] = True
                print(f"               cuGreenCtxCreate OK on the {sizes[0]}-SM group")
                if hasattr(lib, "cuGreenCtxDestroy"):
                    lib.cuGreenCtxDestroy(gctx)
            else:
                err = ctypes.c_char_p()
                lib.cuGetErrorName(rc, ctypes.byref(err))
                rec["create_error"] = err.value.decode() if err.value else f"rc={rc}"
                print(f"               cuGreenCtxCreate FAILED {rec['create_error']}")
        else:
            err = ctypes.c_char_p()
            lib.cuGetErrorName(rc, ctypes.byref(err))
            rec["desc_error"] = err.value.decode() if err.value else f"rc={rc}"
            print(f"               cuDevResourceGenerateDesc FAILED {rec['desc_error']}")
        out["splits"].append(rec)

    Path("scratch").mkdir(exist_ok=True)
    Path("scratch/green_ctx_probe.json").write_text(json.dumps(out, indent=2) + "\n")
    usable = [s for s in out["splits"] if s.get("green_ctx_created")]
    print(f"\n  VERDICT: {len(usable)} of {len(out['splits'])} splits produced a usable green "
          f"context")
    if usable:
        print(f"  partitions available: "
              f"{sorted({tuple(s['group_sm_counts']) for s in usable})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
