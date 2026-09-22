# E11b — transfer, synchronisation and zero-copy costs (protocol §13.3)

Run directory: `20260922b_E11b_xfer`  
Written by `bench/xfer_latency.py` at 2026-09-22T13:20:58Z.

## Answer — the fixed cost of a transfer is not zero

A host<->device transfer of B bytes costs `t = a + b*B`. Measured on this device, with pinned memory, **a = 38.2-39.7 us** and b corresponds to 35.7-36.0 GB/s. The hybrid model assumes a = 0 and prices every transfer as bytes / 143 GB/s.

| method | dir | a (ms) | b (ms/MB) | GB/s from b | r2 | points |
|---|---|---|---|---|---|---|
| pageable | d2h | 0.0567 | 0.1300 | 7.69 | 0.9999 | 13 |
| pageable | h2d | 0.0739 | 0.0689 | 14.52 | 0.9984 | 13 |
| pinned | d2h | 0.0382 | 0.0280 | 35.69 | 0.9990 | 13 |
| pinned | h2d | 0.0397 | 0.0278 | 36.00 | 0.9995 | 13 |

## Per-size medians (every size, both directions, three methods)

| method | dir | size | bytes | median ms | p5 | p95 | n | GB/s |
|---|---|---|---|---|---|---|---|---|
| pageable | h2d | 4KiB | 4096 | 0.0571 | 0.0540 | 0.0640 | 1000 | 0.07 |
| pageable | d2h | 4KiB | 4096 | 0.0602 | 0.0569 | 0.0642 | 1000 | 0.07 |
| pinned | h2d | 4KiB | 4096 | 0.0425 | 0.0370 | 0.0456 | 1000 | 0.10 |
| pinned | d2h | 4KiB | 4096 | 0.0441 | 0.0345 | 0.0473 | 1000 | 0.09 |
| pageable | h2d | 16KiB | 16384 | 0.0575 | 0.0543 | 0.0612 | 1000 | 0.28 |
| pageable | d2h | 16KiB | 16384 | 0.0617 | 0.0580 | 0.0648 | 1000 | 0.27 |
| pinned | h2d | 16KiB | 16384 | 0.0437 | 0.0374 | 0.0472 | 1000 | 0.38 |
| pinned | d2h | 16KiB | 16384 | 0.0431 | 0.0349 | 0.0469 | 1000 | 0.38 |
| pageable | h2d | 64KiB | 65536 | 0.0620 | 0.0592 | 0.0658 | 1000 | 1.06 |
| pageable | d2h | 64KiB | 65536 | 0.0678 | 0.0645 | 0.0707 | 1000 | 0.97 |
| pinned | h2d | 64KiB | 65536 | 0.0431 | 0.0392 | 0.0471 | 1000 | 1.52 |
| pinned | d2h | 64KiB | 65536 | 0.0427 | 0.0337 | 0.0469 | 1000 | 1.53 |
| pageable | h2d | 256KiB | 262144 | 0.0852 | 0.0823 | 0.0902 | 200 | 3.08 |
| pageable | d2h | 256KiB | 262144 | 0.0943 | 0.0903 | 0.0982 | 200 | 2.78 |
| pinned | h2d | 256KiB | 262144 | 0.0431 | 0.0400 | 0.0460 | 200 | 6.08 |
| pinned | d2h | 256KiB | 262144 | 0.0429 | 0.0358 | 0.0467 | 200 | 6.11 |
| pageable | h2d | 1MiB | 1048576 | 0.1680 | 0.1618 | 0.1718 | 200 | 6.24 |
| pageable | d2h | 1MiB | 1048576 | 0.1898 | 0.1868 | 0.1950 | 200 | 5.53 |
| pinned | h2d | 1MiB | 1048576 | 0.0707 | 0.0618 | 0.0747 | 200 | 14.83 |
| pinned | d2h | 1MiB | 1048576 | 0.0635 | 0.0588 | 0.0726 | 200 | 16.52 |
| pageable | h2d | 2MiB | 2097152 | 0.2420 | 0.2378 | 0.2542 | 200 | 8.67 |
| pageable | d2h | 2MiB | 2097152 | 0.3284 | 0.3244 | 0.3413 | 200 | 6.39 |
| pinned | h2d | 2MiB | 2097152 | 0.0986 | 0.0952 | 0.1026 | 200 | 21.27 |
| pinned | d2h | 2MiB | 2097152 | 0.0991 | 0.0883 | 0.1024 | 200 | 21.16 |
| pageable | h2d | 4MiB | 4194304 | 0.3878 | 0.3831 | 0.4142 | 200 | 10.81 |
| pageable | d2h | 4MiB | 4194304 | 0.5853 | 0.5807 | 0.6099 | 200 | 7.17 |
| pinned | h2d | 4MiB | 4194304 | 0.1627 | 0.1595 | 0.1673 | 200 | 25.78 |
| pinned | d2h | 4MiB | 4194304 | 0.1630 | 0.1602 | 0.1664 | 200 | 25.73 |
| pageable | h2d | 16MiB | 16777216 | 1.2270 | 1.2191 | 1.2605 | 200 | 13.67 |
| pageable | d2h | 16MiB | 16777216 | 2.2283 | 2.1950 | 2.2470 | 200 | 7.53 |
| pinned | h2d | 16MiB | 16777216 | 0.5040 | 0.5008 | 0.5088 | 200 | 33.29 |
| pinned | d2h | 16MiB | 16777216 | 0.5063 | 0.5024 | 0.5098 | 200 | 33.14 |
| pageable | h2d | scores_fp32 | 1414944 | 0.1783 | 0.1747 | 0.1883 | 200 | 7.93 |
| pageable | d2h | scores_fp32 | 1414944 | 0.2374 | 0.2330 | 0.2454 | 200 | 5.96 |
| pinned | h2d | scores_fp32 | 1414944 | 0.0783 | 0.0753 | 0.0831 | 200 | 18.08 |
| pinned | d2h | scores_fp32 | 1414944 | 0.0796 | 0.0746 | 0.0816 | 200 | 17.78 |
| pageable | h2d | scores_fp16 | 707472 | 0.1312 | 0.1258 | 0.1472 | 200 | 5.39 |
| pageable | d2h | scores_fp16 | 707472 | 0.1494 | 0.1460 | 0.1535 | 200 | 4.73 |
| pinned | h2d | scores_fp16 | 707472 | 0.0559 | 0.0464 | 0.0586 | 200 | 12.66 |
| pinned | d2h | scores_fp16 | 707472 | 0.0468 | 0.0441 | 0.0596 | 200 | 15.13 |
| pageable | h2d | activation_tile | 104448 | 0.0664 | 0.0633 | 0.0703 | 200 | 1.57 |
| pageable | d2h | activation_tile | 104448 | 0.0721 | 0.0689 | 0.0758 | 200 | 1.45 |
| pinned | h2d | activation_tile | 104448 | 0.0432 | 0.0382 | 0.0472 | 200 | 2.42 |
| pinned | d2h | activation_tile | 104448 | 0.0425 | 0.0350 | 0.0471 | 200 | 2.46 |
| pageable | h2d | ffn_tile | 417792 | 0.1012 | 0.0982 | 0.1060 | 200 | 4.13 |
| pageable | d2h | ffn_tile | 417792 | 0.1151 | 0.1113 | 0.1194 | 200 | 3.63 |
| pinned | h2d | ffn_tile | 417792 | 0.0437 | 0.0417 | 0.0476 | 200 | 9.55 |
| pinned | d2h | ffn_tile | 417792 | 0.0436 | 0.0345 | 0.0511 | 200 | 9.57 |
| pageable | h2d | kv_bytes_total | 15040512 | 1.1006 | 1.0924 | 1.1327 | 200 | 13.66 |
| pageable | d2h | kv_bytes_total | 15040512 | 2.0286 | 2.0039 | 2.0585 | 200 | 7.41 |
| pinned | h2d | kv_bytes_total | 15040512 | 0.4580 | 0.4549 | 0.4625 | 200 | 32.84 |
| pinned | d2h | kv_bytes_total | 15040512 | 0.4604 | 0.4557 | 0.4632 | 200 | 32.67 |
| zero_copy | cpu_write | scores_fp32 | 1414944 | 0.0950 | 0.0924 | 0.1041 | 200 | 14.89 |
| zero_copy | gpu_read | scores_fp32 | 1414944 | 0.8008 | 0.7850 | 0.8383 | 200 | 1.77 |
| zero_copy | roundtrip | scores_fp32 | 1414944 | 1.0740 | 1.0564 | 1.1240 | 200 | 1.32 |
| zero_copy | cpu_write | scores_fp16 | 707472 | 0.0491 | 0.0489 | 0.0503 | 200 | 14.40 |
| zero_copy | gpu_read | scores_fp16 | 707472 | 0.7759 | 0.4887 | 0.8110 | 200 | 0.91 |
| zero_copy | roundtrip | scores_fp16 | 707472 | 0.6947 | 0.6760 | 0.7430 | 200 | 1.02 |
| zero_copy | cpu_write | activation_tile | 104448 | 0.0094 | 0.0089 | 0.0106 | 200 | 11.10 |
| zero_copy | gpu_read | activation_tile | 104448 | 0.2407 | 0.2348 | 0.2533 | 200 | 0.43 |
| zero_copy | roundtrip | activation_tile | 104448 | 0.3343 | 0.3260 | 0.3509 | 200 | 0.31 |
| zero_copy | cpu_write | ffn_tile | 417792 | 0.0324 | 0.0322 | 0.0332 | 200 | 12.88 |
| zero_copy | gpu_read | ffn_tile | 417792 | 0.2771 | 0.2710 | 0.2862 | 200 | 1.51 |
| zero_copy | roundtrip | ffn_tile | 417792 | 0.4094 | 0.4018 | 0.4297 | 200 | 1.02 |
| zero_copy | cpu_write | kv_bytes_total | 15040512 | 1.4783 | 1.4639 | 1.5039 | 200 | 10.17 |
| zero_copy | gpu_read | kv_bytes_total | 15040512 | 0.7149 | 0.6968 | 0.7591 | 200 | 21.04 |
| zero_copy | roundtrip | kv_bytes_total | 15040512 | 2.5478 | 2.5104 | 2.6302 | 200 | 5.90 |
| device_control | gpu_read | scores_fp32 | 1414944 | 0.0723 | 0.0684 | 0.0774 | 200 | 19.58 |
| device_control | gpu_read | scores_fp16 | 707472 | 0.0716 | 0.0687 | 0.0759 | 200 | 9.88 |
| device_control | gpu_read | activation_tile | 104448 | 0.0641 | 0.0615 | 0.0666 | 200 | 1.63 |
| device_control | gpu_read | ffn_tile | 417792 | 0.0634 | 0.0612 | 0.0660 | 200 | 6.59 |
| device_control | gpu_read | kv_bytes_total | 15040512 | 0.1549 | 0.1414 | 0.1587 | 200 | 97.11 |

## Synchronisation costs

| sync | median ms | p5 | p95 | n | net of control | note |
|---|---|---|---|---|---|---|
| synchronize_idle | 0.0104 | 0.0102 | 0.0109 | 1000 |  |  |
| event_synchronize_done | 0.0012 | 0.0011 | 0.0012 | 1000 |  |  |
| event_query | 0.0008 | 0.0008 | 0.0009 | 1000 |  |  |
| tiny_kernel_roundtrip | 0.0607 | 0.0588 | 0.0625 | 1000 |  |  |
| _two_kernels_one_stream | 0.0921 | 0.0892 | 0.0954 | 1000 |  | control, subtracted from stream_handoff |
| stream_handoff | 0.1608 | 0.1568 | 0.1696 | 1000 | 0.0687 | reported net of _two_kernels_one_stream, which is subtracted from the median, p5 and p95 alike |

## Conditions

model = none (no policy is loaded; the sizes are π0's, taken from E11c). MAXN + jetson_clocks, GPU 1300.5 MHz. MB = 10^6 B in the fits; KiB/MiB labels are binary. Zero-copy runs in a separate process because `torch.cuda.memory.change_current_allocator` is global and irreversible, and its control — the same reduction over an ordinary device tensor — runs in that same process so the two are comparable.

## Reproduce

```
nvcc -O3 -arch=sm_87 --shared -Xcompiler -fPIC \
     -o bench/managed_alloc.so bench/managed_alloc.cu
python bench/xfer_latency.py --outdir results/20260922b_E11b_xfer
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
