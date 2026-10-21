"""CUDA green contexts on Orin: partition the 16 SMs inside ONE process, for E12 Step 3.

WHY THIS IS THE ROUTE THAT IS LEFT
E12's MAXN result is that pi0's two stages do not overlap because the HOST issues for the
whole wall clock (issue/wall = 1.0000), so GPU contention -- KerColle's mechanism -- was never
exercised. Every other route to exercising it is closed: two streams in one process cannot be
fed by one Python thread (§13.7 forbids a second; T010's numbers were GIL artefacts) and the
compiled rung corrupts across streams; two processes are OOM-killed, because Orin's 62 GB is
shared CPU+GPU and one pi0 at bf16_both is ~15 GB.

Green contexts partition SMs within one process, so they dodge the memory limit and let the
question be asked from the other side. Instead of "do two stages slow each other down", ask
**"how does ONE stage's time change when it is given 8 SMs instead of 16?"** A stage that
barely slows when half the SMs are taken away was not using them, and a co-runner could have
had them -- a contention answer that never needs two things running at once.

PROBED ON THIS DEVICE (bench/probe_green_ctx.py, driver 12.6, Orin, 16 SMs): all four
green-context entry points resolve and 7 of 7 splits produced a usable context. The driver
grants (4,4,4,4), (6,6), (8,8), (10,) and (12,). minCount is a MINIMUM and the driver rounds
to what the hardware gives, so this module always reports the size it actually got and never
silently accepts a different one.

THE CONTROL THIS MODULE SHIPS WITH, AND WHY IT IS NOT OPTIONAL
Creating a green context proves nothing about whether work launched on its stream is actually
confined to the partition. If it is not, every per-SM-count number would be a measurement of
nothing, arriving as plausible milliseconds -- the exact failure mode that made E12's compiled
pipeline produce NaN behind sensible timings. `verify_partition_is_real()` therefore times a
compute-bound GEMM at each SM count and requires it to slow down roughly in proportion. A flat
curve means the mechanism is inert and the caller must stop.
"""
from __future__ import annotations

import contextlib
import ctypes

CU_DEV_RESOURCE_TYPE_SM = 1
CU_GREEN_CTX_DEFAULT_STREAM = 1

_lib = ctypes.CDLL("libcuda.so.1")


class CUdevSmResource(ctypes.Structure):
    _fields_ = [("smCount", ctypes.c_uint)]


class CUdevResource(ctypes.Structure):
    """cuda.h layout: type, 92 bytes of ABI padding, then a 48-byte union. 144 total.

    The padding is ABI, not slack. Getting it wrong makes smCount read adjacent bytes, which
    would present as a successful probe reporting a wrong SM count.
    """
    _fields_ = [("type", ctypes.c_int),
                ("_internal_padding", ctypes.c_ubyte * 92),
                ("sm", CUdevSmResource),
                ("_union_pad", ctypes.c_ubyte * 44)]


def _err(rc: int) -> str:
    p = ctypes.c_char_p()
    _lib.cuGetErrorName(rc, ctypes.byref(p))
    return p.value.decode() if p.value else f"rc={rc}"


def _chk(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what}: {_err(rc)}")


def device_sm_count(dev_ordinal: int = 0) -> int:
    _chk(_lib.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    _chk(_lib.cuDeviceGet(ctypes.byref(dev), dev_ordinal), "cuDeviceGet")
    res = CUdevResource()
    _chk(_lib.cuDeviceGetDevResource(dev, ctypes.byref(res), CU_DEV_RESOURCE_TYPE_SM),
         "cuDeviceGetDevResource")
    return res.sm.smCount


class GreenCtxStream:
    """A green context of (about) `want_sm` SMs, and a torch stream that runs inside it.

    `granted_sm` is what the driver actually gave. It is never assumed equal to `want_sm`;
    callers label their results with `granted_sm`.
    """

    def __init__(self, want_sm: int, dev_ordinal: int = 0, _resource=None,
                 _meta: dict | None = None):
        import torch                                                   # noqa: PLC0415
        _chk(_lib.cuInit(0), "cuInit")
        self._dev = ctypes.c_int()
        _chk(_lib.cuDeviceGet(ctypes.byref(self._dev), dev_ordinal), "cuDeviceGet")

        inp = CUdevResource()
        _chk(_lib.cuDeviceGetDevResource(self._dev, ctypes.byref(inp),
                                         CU_DEV_RESOURCE_TYPE_SM),
             "cuDeviceGetDevResource")
        self.device_sm = inp.sm.smCount

        if _resource is not None:
            # Built by split_pair() from a group or from a split's REMAINDER, so the two
            # partitions of one split are disjoint by construction.
            res = _resource
            self.want_sm = want_sm
            self.granted_sm = res.sm.smCount
            self.n_groups = (_meta or {}).get("n_groups", 1)
            self.remaining_sm = (_meta or {}).get("remaining_sm", 0)
        else:
            groups = (CUdevResource * 32)()
            nb = ctypes.c_uint(32)
            remaining = CUdevResource()
            _chk(_lib.cuDevSmResourceSplitByCount(
                groups, ctypes.byref(nb), ctypes.byref(inp), ctypes.byref(remaining),
                ctypes.c_uint(0), ctypes.c_uint(want_sm)), "cuDevSmResourceSplitByCount")
            if nb.value < 1:
                raise RuntimeError(f"split for {want_sm} SMs produced no groups")
            self.want_sm = want_sm
            self.granted_sm = groups[0].sm.smCount
            self.n_groups = nb.value
            self.remaining_sm = remaining.sm.smCount
            res = groups[0]
        self._res = res            # keep the struct alive for the descriptor

        self._desc = ctypes.c_void_p()
        _chk(_lib.cuDevResourceGenerateDesc(ctypes.byref(self._desc), ctypes.byref(res),
                                            ctypes.c_uint(1)),
             "cuDevResourceGenerateDesc")
        self._gctx = ctypes.c_void_p()
        _chk(_lib.cuGreenCtxCreate(ctypes.byref(self._gctx), self._desc, self._dev,
                                   ctypes.c_uint(CU_GREEN_CTX_DEFAULT_STREAM)),
             "cuGreenCtxCreate")

        raw = ctypes.c_void_p()
        _chk(_lib.cuGreenCtxStreamCreate(ctypes.byref(raw), self._gctx,
                                         ctypes.c_uint(1), ctypes.c_int(0)),
             "cuGreenCtxStreamCreate")
        self._raw_stream = raw

        # The primary context, saved so activate() can put it back. Measured on this device:
        # wrapping the green stream is NOT sufficient. A plain `a @ b` runs on it, but
        # nn.Linear does not -- it takes cuBLASLt's fused gemm_and_bias path, which returns
        # CUBLAS_STATUS_EXECUTION_FAILED, and torch's "recover by calling unfused cublas
        # path" then raises `CUDA error: invalid device context`. Setting DISABLE_ADDMM_CUDA_LT=1
        # does NOT fix it. What does is making the green context CURRENT on the thread while
        # launching, via cuCtxFromGreenCtx + cuCtxSetCurrent, so cuBLAS binds its handle to
        # the context that owns the stream. Hence activate().
        self._primary = ctypes.c_void_p()
        _chk(_lib.cuCtxGetCurrent(ctypes.byref(self._primary)), "cuCtxGetCurrent")
        self._ctx = ctypes.c_void_p()
        _chk(_lib.cuCtxFromGreenCtx(ctypes.byref(self._ctx), self._gctx),
             "cuCtxFromGreenCtx")
        # torch.cuda must be up before an external stream can be wrapped, else the wrap
        # succeeds against an uninitialised context and fails later inside a timed region.
        torch.cuda.init()
        self.stream = torch.cuda.ExternalStream(raw.value)

    @contextlib.contextmanager
    def switch(self):
        """Make this green context current and select its stream, WITHOUT any synchronisation.

        For issuing work inside a pipeline (E12b). activate() below drains the device on exit,
        which is right for a one-stage measurement but, called around every stage of a co-run,
        would serialise the two streams and manufacture exactly the no-overlap result the
        co-run exists to test. The primary context is restored on exit; queued work keeps
        running in this green context.
        """
        import torch                                                   # noqa: PLC0415
        _chk(_lib.cuCtxSetCurrent(self._ctx), "cuCtxSetCurrent(green)")
        try:
            with torch.cuda.stream(self.stream):
                yield self.stream
        finally:
            _chk(_lib.cuCtxSetCurrent(self._primary), "cuCtxSetCurrent(primary)")

    @contextlib.contextmanager
    def activate(self):
        """Make this green context current for the duration, then restore the primary one.

        Required, not cosmetic: without it cuBLASLt's fused addmm fails on the green stream
        and torch's fallback leaves the thread with an invalid context. Always paired with
        `torch.cuda.stream(self.stream)` by the caller -- the context selects WHICH SMs, the
        stream selects the ordering, and both are needed.
        """
        import torch                                                   # noqa: PLC0415
        _chk(_lib.cuCtxSetCurrent(self._ctx), "cuCtxSetCurrent(green)")
        try:
            with torch.cuda.stream(self.stream):
                yield self.stream
                # Drain inside the green context: a sync after the primary context is
                # restored would wait on the wrong context's work.
                torch.cuda.synchronize()
        finally:
            _chk(_lib.cuCtxSetCurrent(self._primary), "cuCtxSetCurrent(primary)")

    def __repr__(self) -> str:
        return (f"<GreenCtxStream want={self.want_sm} granted={self.granted_sm} "
                f"of {self.device_sm} groups={self.n_groups} rem={self.remaining_sm}>")

    def close(self) -> None:
        if getattr(self, "_gctx", None) is not None:
            _lib.cuGreenCtxDestroy(self._gctx)
            self._gctx = None


def split_pair(prefix_sms: int, expert_sms: int):
    """Two DISJOINT green contexts for one split (E12b M3), prefix first, expert second.

    Equal halves come from one split's two groups; an unequal split takes the larger side
    as group 0 of a split at that size and the smaller side as that split's REMAINDER, so the
    two never share an SM. What the driver granted is returned on each object
    (`granted_sm`) and must be what the caller reports -- a requested 14/2 may come back as
    something else, and the granularity is itself part of the result.
    """
    _chk(_lib.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    _chk(_lib.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    inp = CUdevResource()
    _chk(_lib.cuDeviceGetDevResource(dev, ctypes.byref(inp), CU_DEV_RESOURCE_TYPE_SM),
         "cuDeviceGetDevResource")
    big, small = max(prefix_sms, expert_sms), min(prefix_sms, expert_sms)
    groups = (CUdevResource * 32)()
    nb = ctypes.c_uint(32)
    rem = CUdevResource()
    _chk(_lib.cuDevSmResourceSplitByCount(groups, ctypes.byref(nb), ctypes.byref(inp),
                                          ctypes.byref(rem), ctypes.c_uint(0),
                                          ctypes.c_uint(big)), "split")
    meta = {"n_groups": nb.value, "remaining_sm": rem.sm.smCount}
    if prefix_sms == expert_sms:
        if nb.value < 2:
            raise RuntimeError(f"{big}/{big}: driver gave {nb.value} group(s), need 2")
        a, b = groups[0], groups[1]
    else:
        if nb.value < 1 or rem.sm.smCount == 0:
            raise RuntimeError(f"{prefix_sms}/{expert_sms}: no remainder to use "
                               f"(groups {nb.value}, remaining {rem.sm.smCount})")
        big_res, small_res = groups[0], rem
        a, b = (big_res, small_res) if prefix_sms >= expert_sms else (small_res, big_res)
    # copy the structs: the arrays above go out of scope
    ra, rb = CUdevResource(), CUdevResource()
    ctypes.memmove(ctypes.byref(ra), ctypes.byref(a), ctypes.sizeof(CUdevResource))
    ctypes.memmove(ctypes.byref(rb), ctypes.byref(b), ctypes.sizeof(CUdevResource))
    return (GreenCtxStream(prefix_sms, _resource=ra, _meta=meta),
            GreenCtxStream(expert_sms, _resource=rb, _meta=meta))


def verify_partition_is_real(sm_counts=(4, 8, 16), iters: int = 30,
                            n: int = 2048) -> dict:
    """Does work on a green stream actually get confined to the partition?

    Times a compute-bound bf16 GEMM at each SM count. A real partition slows it roughly in
    proportion to the SMs removed. A FLAT curve means the green context is decorative and no
    per-SM number from it means anything -- so this returns a verdict the caller must honour
    rather than a number the caller can quietly use.
    """
    import torch                                                       # noqa: PLC0415
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    out = {"n": n, "iters": iters, "points": []}

    def timed(g) -> float:
        """g is a GreenCtxStream or None for the whole device. Uses activate(), so the
        control exercises exactly the mechanism the measurement will use."""
        cm = g.activate() if g is not None else contextlib.nullcontext()
        with cm:
            for _ in range(5):
                a @ b
            torch.cuda.synchronize()
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            ev0.record()
            for _ in range(iters):
                a @ b
            ev1.record()
            torch.cuda.synchronize()
        return ev0.elapsed_time(ev1) / iters

    full = device_sm_count()
    base = None
    for sm in sm_counts:
        if sm >= full:
            ms, granted, g = timed(None), full, None
        else:
            g = GreenCtxStream(sm)
            granted = g.granted_sm
            ms = timed(g)
        if granted == full:
            base = ms
        out["points"].append({"want_sm": sm, "granted_sm": granted,
                              "ms_per_gemm": round(ms, 4)})
        if g is not None:
            g.close()
    if base:
        for p in out["points"]:
            p["slowdown_vs_full"] = round(p["ms_per_gemm"] / base, 4)
            p["ideal_slowdown"] = round(full / p["granted_sm"], 4)
    small = [p for p in out["points"] if p["granted_sm"] < full]
    worst = max((p.get("slowdown_vs_full", 1.0) for p in small), default=1.0)
    out["partition_is_real"] = bool(worst > 1.15)
    out["verdict"] = (
        f"largest slowdown from a reduced partition is {worst:.3f}x. "
        + ("Work IS confined to the partition, so per-SM-count timings are meaningful."
           if out["partition_is_real"] else
           "FLAT: the green context did NOT restrict execution. Every per-SM number from "
           "this mechanism would be a measurement of nothing -- do not report any."))
    return out


if __name__ == "__main__":
    import json
    print(f"  device SMs: {device_sm_count()}")
    r = verify_partition_is_real()
    for p in r["points"]:
        print(f"  want {p['want_sm']:2d} -> granted {p['granted_sm']:2d}  "
              f"{p['ms_per_gemm']:8.4f} ms/gemm  "
              f"slowdown {p.get('slowdown_vs_full', float('nan')):.3f}x  "
              f"(ideal {p.get('ideal_slowdown', float('nan')):.3f}x)")
    print(f"\n  {r['verdict']}")
    print(json.dumps({"partition_is_real": r["partition_is_real"]}))
