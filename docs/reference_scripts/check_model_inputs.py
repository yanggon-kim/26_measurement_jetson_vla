#!/usr/bin/env python3
"""Validate an E11 `model_inputs.csv` against docs/02_protocol.md §13.8.

Standard library only; runs on the Jetson and on the research server (no GPU needed).

    python docs/reference_scripts/check_model_inputs.py results/<run>/model_inputs.csv [more.csv ...]

Exit status 0 = every file conforms; 1 = at least one error (each printed as file:line: message).
Unlike the other files in this directory, this one is run in place: do not copy or adapt it. If the
schema needs a new quantity, propose it in STATUS.md; the research side extends this file.
"""

import csv
import math
import os
import re
import sys

HEADER = ["model", "model_commit", "bench_commit", "quantity", "key", "variant", "value", "unit",
          "stat", "n", "exec_style", "timing", "power_mode", "precision", "P", "S", "N", "source_file"]

MODELS = {"pi0", "pi05", "smolvla", "groot_n17", "groot_n16", "groot_n15", "none"}  # thor branch: all six models
STATS = {"median", "mean", "p5", "p95", "min", "max", "value", "fit_a", "fit_b", "fit_r2"}
EXEC_STYLES = {"", "eager", "compiled", "cudagraph"}
TIMINGS = {"", "wall_isolated", "gpu_event", "wall_launch", "wall_loop", "slope", "analytic"}
PRECISIONS = {"", "bf16", "fp16", "fp32"}

# quantity -> allowed units
QUANTITIES = {
    "t_preprocess": {"ms"}, "t_vision": {"ms"}, "t_vlm_prefix": {"ms"}, "t_postprocess": {"ms"},
    "t_ae_step": {"ms"}, "t_ae_total": {"ms"}, "t_e2e": {"ms"},
    "t_host_segment": {"ms"},
    "n_kernels_host_segment": {"count"}, "bytes_host_segment": {"bytes"}, "flops_host_segment": {"flops"},
    "t_host_layer": {"ms"}, "t_step_extra": {"ms"}, "t_launch_per_kernel": {"ms"}, "t_sync": {"ms"},
    "xfer_latency": {"ms"}, "xfer_latency_fit": {"ms", "ms_per_MB", "ratio"},
    "kv_bytes_per_layer": {"bytes"}, "kv_bytes_total": {"bytes"}, "t_kv_materialise": {"ms"},
    "kv_copied_bytes_per_step": {"bytes"}, "boundary_bytes_per_step": {"bytes"},
    "prefix_latency_fit": {"ms", "ms_per_image", "ms_per_token", "ratio"},
    "prefix_throughput": {"inf_per_s"}, "e2e_throughput": {"inf_per_s"}, "ae_throughput": {"inf_per_s"},
    "prefix_slowdown": {"ratio"}, "host_layer_slowdown": {"ratio"}, "bg_bw_GBps": {"GBps"},
    "cpu_util": {"pct"}, "p_mean_W": {"W"},
    "gpu_idle_ae_total": {"ms"}, "gpu_idle_frac_ae_total": {"ratio"},
    "achieved_bw_GBps": {"GBps"}, "achieved_tflops": {"TFLOPps"},
}
KEY_REQUIRED = {
    "t_host_segment", "n_kernels_host_segment", "bytes_host_segment", "flops_host_segment", "t_host_layer",
    "t_step_extra", "t_sync", "xfer_latency", "xfer_latency_fit", "kv_bytes_per_layer", "kv_bytes_total",
    "t_kv_materialise", "boundary_bytes_per_step", "prefix_latency_fit", "prefix_slowdown",
    "host_layer_slowdown", "bg_bw_GBps", "cpu_util", "p_mean_W", "achieved_bw_GBps", "achieved_tflops",
}
FIT_UNITS = {"fit_a": {"ms"}, "fit_r2": {"ratio"},
             "fit_b": {"ms_per_MB", "ms_per_image", "ms_per_token"}}
VARIANT_RE = re.compile(r"^[A-Za-z0-9_]+=[^;=]+(;[A-Za-z0-9_]+=[^;=]+)*$")


def check(path):
    errors = []
    base = os.path.dirname(os.path.abspath(path))

    def err(line, msg):
        errors.append(f"{path}:{line}: {msg}")

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return [f"{path}:1: empty file"]
        if header != HEADER:
            return [f"{path}:1: header must be exactly {','.join(HEADER)}"]
        n_rows = 0
        for line, row in enumerate(reader, start=2):
            if not row:
                continue
            n_rows += 1
            if len(row) != len(HEADER):
                err(line, f"{len(row)} fields, expected {len(HEADER)}")
                continue
            r = dict(zip(HEADER, row))
            q = r["quantity"]
            if r["model"] not in MODELS:
                err(line, f"model '{r['model']}' not in {sorted(MODELS)}")
            if r["model"] != "none" and not r["model_commit"]:
                err(line, "model_commit is empty")
            if not r["bench_commit"]:
                err(line, "bench_commit is empty")
            if q not in QUANTITIES:
                err(line, f"unknown quantity '{q}'")
            else:
                if r["unit"] not in QUANTITIES[q]:
                    err(line, f"unit '{r['unit']}' not allowed for {q} ({sorted(QUANTITIES[q])})")
                if q in KEY_REQUIRED and not r["key"]:
                    err(line, f"{q} needs a key")
            if r["stat"] not in STATS:
                err(line, f"stat '{r['stat']}' not in {sorted(STATS)}")
            elif r["stat"] in FIT_UNITS and r["unit"] not in FIT_UNITS[r["stat"]]:
                err(line, f"stat {r['stat']} needs unit in {sorted(FIT_UNITS[r['stat']])}")
            if r["variant"] and not VARIANT_RE.match(r["variant"]):
                err(line, f"variant '{r['variant']}' is not name=value[;name=value...]")
            try:
                v = float(r["value"])
                if not math.isfinite(v):
                    err(line, "value is not finite (leave unmeasured quantities out of the file)")
            except ValueError:
                err(line, f"value '{r['value']}' is not a number")
            try:
                if int(r["n"]) < 1:
                    err(line, "n must be >= 1")
            except ValueError:
                err(line, f"n '{r['n']}' is not an integer")
            if r["exec_style"] not in EXEC_STYLES:
                err(line, f"exec_style '{r['exec_style']}' not in {sorted(EXEC_STYLES)}")
            if r["timing"] not in TIMINGS:
                err(line, f"timing '{r['timing']}' not in {sorted(TIMINGS)}")
            if r["precision"] not in PRECISIONS:
                err(line, f"precision '{r['precision']}' not in {sorted(PRECISIONS)}")
            if not r["power_mode"]:
                err(line, "power_mode is empty")
            for col in ("P", "S", "N"):
                if r[col]:
                    try:
                        int(r[col])
                    except ValueError:
                        err(line, f"{col} '{r[col]}' is not an integer")
            src = r["source_file"]
            if not src:
                err(line, "source_file is empty")
            elif os.path.isabs(src) or ".." in src.split("/"):
                err(line, f"source_file '{src}' must be relative to, and inside, the result directory")
            elif not os.path.exists(os.path.join(base, src)):
                err(line, f"source_file '{src}' does not exist in {base}")
        if n_rows == 0:
            err(1, "no data rows")
    return errors


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    all_errors = []
    for p in argv[1:]:
        e = check(p)
        all_errors += e
        print(f"{p}: {'OK' if not e else str(len(e)) + ' error(s)'}")
    for e in all_errors:
        print(e)
    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
