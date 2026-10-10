#!/usr/bin/env python3
"""Root-cause analysis of the VLM prefix (vision + VLM prefill) on Jetson AGX Orin, MAXN.

CPU-only, reads committed results only; no number is typed in except model dimensions (documented in SHAPES)
and the analytic per-stage floors from microarch (embedded below with their commit, re-checked when the file is
reachable).

Models / configurations (the best rung of each, as in reports/E4i_best_combinations.md):
  pi0        bf16_both  (LeRobot f0d2b37, N = 10)   kernels: results/20260924_E5_pi0_best_N10 (nsys)
  groot_n17  vlm_compile (N = 4)                    kernels: results/20260924_E5_groot_n17_best (nsys)
  smolvla    vlm_graphs  (N = 10)                   kernels: results/20260924_E5_smolvla_best_N10 (nsys)

Method
  * Stage wall time: the untraced timing rows (E4c / E4d ladders; E14b `pad` for pi0 graphs_both).
  * Kernel time per class: nsys CUDA GPU trace, kernels whose start lies in the stage window, averaged over the
    captured inferences. Kernel durations are GPU times and are not inflated by tracing; idle = wall - kernel union
    (derived), because nsys inflates traced wall time (x1.53 on this device, E5 meta notes).
  * GEMM shape per (kernel, grid): pi0 from E13d's per-kernel shapes; GR00T and SmolVLA from tile x grid and the
    model dimensions (SHAPES), validated by the per-stage linear FLOPs of stage_flops_bytes.csv (printed check).
  * Floor per class = max(FLOPs / 33.85 TFLOP/s, unique bytes / 187.47 GB/s) (E1 measured peaks); a pass that can be
    fused away or removed (SIMT glue, copies, idle, dead work) has floor 0.
  * Counters: E4g C1 ncu (first 40 kernels per phase, SpeedOfLight / Memory / WarpState / Occupancy + stall
    ratios), joined on (kernel name, grid); pi0 DRAM bytes: E13d (L2 fills + write sectors, full coverage).
  * recoverable_ms_upper = measured - floor: the method-independent upper bound; rows of kind `dead_work` overlap
    the `efficiency` rows of the same stage (removing work shrinks every class), so only rows of one kind add.

Outputs: analysis/prefix_root_cause.csv (one row per model, stage, cause)
         analysis/prefix_root_cause/classes.csv (kernel-class breakdown with counters)
         analysis/prefix_root_cause/tables.md (the tables quoted in analysis/prefix_root_cause.md)
"""
import csv, glob, json, math, os, re, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results')
OUT_DIR = os.path.join(ROOT, 'analysis', 'prefix_root_cause')
OUT_CSV = os.path.join(ROOT, 'analysis', 'prefix_root_cause.csv')

# ---------------------------------------------------------------- peaks (E1, measured, MAXN, 1300.5 MHz)
def peaks():
    tf = gb = None
    for d in csv.DictReader(open(os.path.join(R, '20260922_E1_peak/latency_summary.csv'))):
        if d['kernel'] == 'gemm_square' and d['dtype'] == 'bfloat16' and d['shape'] == '2048x2048x2048':
            tf = float(d['TFLOPps'])
        if d['kernel'] == 'axpy_inplace' and d['dtype'] == 'bfloat16':
            gb = float(d['GBps'])
    assert abs(tf - 33.85) < 0.01 and abs(gb - 187.47) < 0.01, (tf, gb)
    return tf, gb
PEAK_TF, PEAK_GB = peaks()

# ---------------------------------------------------------------- analytic stage floors (microarch, read-only)
# microarch/04_data/stage_flops_bytes.csv @ workspace commit f751ff4 (2026-10-08); (flops, flops_linear, flops_attn, bytes)
FLOORS = {
    ('pi0', 'vision'): (660529741824, 636068560896, 24461180928, 2488822176),
    ('pi0', 'prefill'): (3332496162816, 3234311700480, 98184462336, 3963965440),
    ('groot_n17', 'vision'): (699005927424, 673236123648, 25769803776, 813914112),
    ('groot_n17', 'prefill'): (449465942016, 439697276928, 9768665088, 1611874304),
    ('groot_n17', 'lm_head'): (169896050688, 169896050688, 0, 622329856),
    ('smolvla', 'vision'): (430637580288, 353328168960, 77309411328, 392918016),
    ('smolvla', 'prefill'): (57604300800, 55679447040, 1924853760, 314855040),
}
def check_floors():
    p = '/home/yanggon/05_VLA_LPDDR/microarch/04_data/stage_flops_bytes.csv'
    if not os.path.exists(p):
        return 'stage_flops_bytes.csv not reachable: embedded values used'
    st = {'vision': 'vision', 'vlm_prefix': 'prefill', 'vlm_lm_head': 'lm_head'}
    for r in csv.DictReader(open(p)):
        k = (r['model'], st.get(r['stage']))
        if k in FLOORS:
            v = (int(r['flops']), int(r['flops_linear']), int(r['flops_attn']), int(r['bytes']))
            assert v == FLOORS[k], (k, v, FLOORS[k])
    return 'stage_flops_bytes.csv re-read: embedded values match'

def stage_floor(model, stage):
    f, fl, fa, b = FLOORS[(model, stage)]
    return max(f / PEAK_TF / 1e9, b / PEAK_GB / 1e6), f / PEAK_TF / 1e9, b / PEAK_GB / 1e6

# ---------------------------------------------------------------- stage wall times (untraced)
def walls():
    w = {}
    for r in csv.DictReader(open(os.path.join(R, '20260922_E4c_pi0_opt_ladder/opt_ladder.csv'))):
        if r['N'] == '10' and r['rung'] == 'bf16_both':
            w[('pi0', 'vision')] = float(r['vision_ms']); w[('pi0', 'prefill')] = float(r['vlm_prefix_ms'])
    for r in csv.DictReader(open(os.path.join(R, '20260922_E4c_smolvla_opt_ladder/opt_ladder.csv'))):
        if r['N'] == '10' and r['rung'] == 'vlm_graphs':
            w[('smolvla', 'vision')] = float(r['vision_ms']); w[('smolvla', 'prefill')] = float(r['vlm_prefix_ms'])
    g = json.load(open(os.path.join(R, '20260923_E4d_groot_n17_ladder/raw/rows/vlm_compile.json')))
    w[('groot_n17', 'vision')] = g['vision_ms']; w[('groot_n17', 'prefill')] = g['vlm_prefix_ms']
    gb = {}
    for r in csv.DictReader(open(os.path.join(R, '20260927_E14b_pi0_padtrim_stages_MAXN/stages.csv'))):
        gb[(r['variant'], r['stage'])] = float(r['median_ms'])
    return w, gb
WALL, E14B = walls()

# ---------------------------------------------------------------- nsys: kernels per stage window
RUNS = {
    'pi0': ('20260924_E5_pi0_best_N10', 'best_N10', 2),
    'groot_n17': ('20260924_E5_groot_n17_best', 'best', 2),
    'smolvla': ('20260924_E5_smolvla_best_N10', 'best_N10', 2),
}
def stage_windows(model):
    d, tag, ninf = RUNS[model]
    rng = [(r['Name'].lstrip(':'), int(r['Projected Start (ns)']), int(r['Projected Duration (ns)']))
           for r in csv.DictReader(open(os.path.join(R, d, 'nsys', f'{tag}_nvtx_gpu_proj_trace.csv')))]
    top = [x for x in rng if x[0] in ('vision', 'vlm_prefix', 'ae_total') or x[0].startswith('ae_step[')]
    wins = []
    i = 0
    while i < len(top):
        if top[i][0] == 'vision':
            v0 = top[i][1]
            j = i
            while j < len(top) and top[j][0] == 'vision':
                j += 1
            v_end = top[j - 1][1] + top[j - 1][2]
            assert top[j][0] == 'vlm_prefix'
            p0 = top[j][1]
            ae0 = top[j + 1][1]
            # pi0 / SmolVLA: vision = [first vision, vlm_prefix); GR00T: vision = its NVTX range, the glue
            # before vlm_prefix (rope index, masked scatter) belongs to the prefill timing (backbone - vision)
            if model == 'groot_n17':
                wins.append({'vision': (v0, v_end), 'prefill': (v_end, ae0)})
            else:
                wins.append({'vision': (v0, p0), 'prefill': (p0, ae0)})
            i = j + 1
        else:
            i += 1
    assert len(wins) == ninf, (model, len(wins))
    return wins

def kernels(model):
    d, tag, ninf = RUNS[model]
    wins = stage_windows(model)
    out = {'vision': [], 'prefill': []}
    for r in csv.DictReader(open(os.path.join(R, d, 'nsys', f'{tag}_cuda_gpu_trace.csv'))):
        s = int(r['Start (ns)']); du = int(r['Duration (ns)'])
        for w in wins:
            for st, (a, b) in w.items():
                if a <= s < b:
                    g = (r['GrdX'], r['GrdY'], r['GrdZ'])
                    out[st].append({'name': r['Name'], 'grid': tuple(int(x) for x in g) if g[0] else None,
                                    'block': r['BlkX'], 'start': s, 'dur': du})
    return out, ninf

def union_ms(ks):
    iv = sorted((k['start'], k['start'] + k['dur']) for k in ks)
    tot, cur = 0, None
    for a, b in iv:
        if cur is None or a > cur[1]:
            if cur: tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur: tot += cur[1] - cur[0]
    return tot / 1e6

# ---------------------------------------------------------------- GEMM shapes and kernel classes
# (model, stage, kernel-name regex, grid) -> (class, [(M, N, K, batch, share of launches)])
# pi0 shapes are E13d's measured (M, N, K) for the same kernel and grid; GR00T / SmolVLA from tile x grid and
# config: GR00T Qwen3-VL vision 24 x 1024 (MLP 4096), 1024 patches (4 images); LLM 16 x 2048 (MLP 6144, q 16 x 128,
# kv 8 x 128), P = 273, vocab 151936. SmolVLA SigLIP 12 x 768 (MLP 3072), 2 x 1024 patches; LLM 16 x 960 (MLP 2560,
# q 15 x 64, kv 5 x 64), P = 177.
G = 'gemm'
SHAPES = [
    # pi0 vision (SigLIP So400m, 27 layers x 3 images, 256 patches, hidden 1152, MLP 4304, 16 heads x 72)
    ('pi0', 'vision', r'^ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn', (34, 4, 1), 'GEMM fc1 256x4304x1152', [(256, 4304, 1152, 1, 1)]),
    ('pi0', 'vision', r'^ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_32x6_tn', (9, 4, 1), 'GEMM fc2 256x1152x4304', [(256, 1152, 4304, 1, 1)]),
    ('pi0', 'vision', r'^ampere_bf16_s16816gemm_bf16_128x64_ldg8_(relu_)?f2f_stages_64x3_tn', (9, 4, 1), 'GEMM q/k/v/o 256x1152x1152', [(256, 1152, 1152, 1, 1)]),
    ('pi0', 'vision', r'^ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn', (16, 4, 1), 'GEMM projector 256x2048x1152', [(256, 2048, 1152, 1, 1)]),
    # pi0 prefill (Gemma 2B, 18 layers, P = 816, hidden 2048, MLP 16384, 8 q heads / 1 kv head x 256)
    ('pi0', 'prefill', r'^ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn', (16, 7, 9), 'GEMM down 816x2048x16384 (K = 16384)', [(816, 2048, 16384, 1, 1)]),
    ('pi0', 'prefill', r'^ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn', (128, 7, 1), 'GEMM gate/up 816x16384x2048', [(816, 16384, 2048, 1, 1)]),
    ('pi0', 'prefill', r'^ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn', (16, 7, 2), 'GEMM q/o 816x2048x2048', [(816, 2048, 2048, 1, 1)]),
    ('pi0', 'prefill', r'^ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn', (2, 13, 1), 'GEMM k/v 816x256x2048', [(816, 256, 2048, 1, 1)]),
    ('pi0', 'prefill', r'^ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn', (7, 13, 8), 'attention QK^T (8 heads, 816x816x256)', [(816, 816, 256, 8, 1)]),
    ('pi0', 'prefill', r'^ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nn', (2, 7, 8), 'attention PV (8 heads, 816x256x816)', [(816, 256, 816, 8, 1)]),
    # GR00T N1.7 vision
    ('groot_n17', 'vision', r'gemm', (24, 8, 1), 'GEMM qkv 1024x3072x1024', [(1024, 3072, 1024, 1, 1)]),
    ('groot_n17', 'vision', r'gemm', (8, 8, 1), 'GEMM o 1024x1024x1024 + fc2 1024x1024x4096', [(1024, 1024, 1024, 1, .5), (1024, 1024, 4096, 1, .5)]),
    ('groot_n17', 'vision', r'gemm', (32, 8, 1), 'GEMM fc1 1024x4096x1024', [(1024, 4096, 1024, 1, 1)]),
    ('groot_n17', 'vision', r'gemm', (32, 2, 1), 'GEMM mergers fc1 256x4096x4096', [(256, 4096, 4096, 1, 1)]),
    ('groot_n17', 'vision', r'gemm', (16, 2, 2), 'GEMM mergers fc2 256x2048x4096', [(256, 2048, 4096, 1, 1)]),
    ('groot_n17', 'vision', r'fprop_implicit_gemm', None, 'patch-embed conv3d (tensor core)', [(1024, 1024, 1536, 1, 1)]),
    # GR00T N1.7 prefill (Qwen3-VL LLM, 16 layers)
    ('groot_n17', 'prefill', r'gemm', (48, 3, 2), 'GEMM gate/up 273x6144x2048 (split-K 2)', [(273, 6144, 2048, 1, 1)]),
    ('groot_n17', 'prefill', r'gemm', (16, 3, 1), 'GEMM down 273x2048x6144', [(273, 2048, 6144, 1, 1)]),
    ('groot_n17', 'prefill', r'gemm', (16, 5, 1), 'GEMM q/o 273x2048x2048', [(273, 2048, 2048, 1, 1)]),
    ('groot_n17', 'prefill', r'gemm', (8, 5, 1), 'GEMM k/v 273x1024x2048', [(273, 1024, 2048, 1, 1)]),
    ('groot_n17', 'prefill', r'gemm', (40, 75, 1), 'lm_head 273x151936x2048 (logits discarded)', [(273, 151936, 2048, 1, 1)]),
    # SmolVLA vision (2 images, per-image calls)
    ('smolvla', 'vision', r'gemm_bf16_128x64_ldg8_relu', (6, 16, 1), 'GEMM q/k/v 1024x768x768', [(1024, 768, 768, 1, 1)]),
    ('smolvla', 'vision', r'gemm_bf16_128x64_ldg8_f2f_stages_64x3', (6, 16, 1), 'GEMM o 1024x768x768', [(1024, 768, 768, 1, 1)]),
    ('smolvla', 'vision', r'gemm', (24, 8, 1), 'GEMM fc1 1024x3072x768', [(1024, 3072, 768, 1, 1)]),
    ('smolvla', 'vision', r'gemm', (6, 8, 1), 'GEMM fc2 1024x768x3072', [(1024, 768, 3072, 1, 1)]),
    ('smolvla', 'vision', r'gemm', (8, 1, 9), 'GEMM connector 64x960x12288', [(64, 960, 12288, 1, 1)]),
    # SmolVLA prefill (16 layers)
    ('smolvla', 'prefill', r'gemm', (20, 3, 1), 'GEMM gate/up 177x2560x960', [(177, 2560, 960, 1, 1)]),
    ('smolvla', 'prefill', r'gemm', (8, 2, 3), 'GEMM down 177x960x2560 (split-K 3)', [(177, 960, 2560, 1, 1)]),
    ('smolvla', 'prefill', r'gemm', (8, 3, 1), 'GEMM q/o 177x960x960', [(177, 960, 960, 1, 1)]),
    ('smolvla', 'prefill', r'gemm', (5, 3, 1), 'GEMM k/v 177x320x960', [(177, 320, 960, 1, 1)]),
    ('smolvla', 'prefill', r'^ampere_sgemm_128x128_tn', (2, 2, 15), 'attention QK^T fp32 SIMT sgemm (15 heads, 177x177x64)', [(177, 177, 64, 15, 1)]),
    ('smolvla', 'prefill', r'cutlass_75_tensorop_bf16_s1688gemm_bf16_64x64_nn', (3, 1, 15), 'attention PV (15 heads, 177x64x177)', [(177, 64, 177, 15, 1)]),
]
WEIGHT_GEMM = lambda c: c.startswith('GEMM') or c.startswith('lm_head')

def classify(model, stage, k):
    n, g = k['name'], k['grid']
    for m, s, rx, grid, cls, shp in SHAPES:
        if m == model and s == stage and re.search(rx, n) and (grid is None or grid == g):
            return cls, shp
    nl = n.lower()
    if 'flash' in nl or 'fmha' in nl or 'attention' in nl and 'triton' not in nl:
        return 'attention kernel (fused)', None
    if 'softmax' in nl:
        return 'attention softmax (SIMT)', None
    if 'cos' in nl and 'sin' in nl:
        return 'rope (SIMT)', None
    if 'convolve' in nl or 'nhwcaddpadding' in nl:
        return 'patch-embed conv (SIMT / layout)', None
    if 'memcpy' in nl or 'memset' in nl or 'copy' in nl and 'triton' not in nl or 'catarray' in nl or 'clone' in nl:
        return 'copies / layout / casts', None
    if 'gemm' in nl:
        return 'GEMM other', None
    if 'layer_norm' in nl or 'rsqrt' in nl or 'mean' in nl and 'triton' in nl:
        return 'norm (+ fused residual / bias)', None
    if 'gelu' in nl or 'silu' in nl:
        return 'activation (GELU / SiLU x up)', None
    return 'other elementwise / index / reduce (SIMT)', None

# ---------------------------------------------------------------- counters
STALLS = ['long_scoreboard', 'short_scoreboard', 'wait', 'no_instruction', 'math_pipe_throttle', 'mio_throttle',
          'lg_throttle', 'barrier', 'membar', 'drain', 'dispatch_stall', 'imc_miss', 'misc', 'selected']
def ncu_counters(model, phase):
    f = os.path.join(R, f'20260923_E4g_c1_{model}_best/ncu/{model}_best_{phase}_details.csv')
    per = collections.defaultdict(dict)
    for r in csv.DictReader(open(f)):
        key = (r['ID'], r['Kernel Name'], r['Grid Size'])
        v = r['Metric Value'].replace(',', '')
        try: v = float(v)
        except ValueError: continue
        mn = r['Metric Name']
        if mn == 'Duration' and r['Metric Unit'] == 'ms': v *= 1000
        if mn == 'Duration' and r['Metric Unit'] == 'ns': v /= 1000
        per[key][mn] = v
    agg = collections.defaultdict(list)
    for (i, name, grid), m in per.items():
        g = tuple(int(x) for x in grid.strip('()').split(','))
        agg[(norm(name), g)].append(m)
    out = {}
    for k, ms in agg.items():
        def mean(x): return sum(m.get(x, float('nan')) for m in ms) / len(ms)
        st = {s: mean(f'smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio') for s in STALLS if s != 'selected'}
        tot = sum(v for v in st.values() if v == v)
        top = sorted(st.items(), key=lambda x: -x[1])[:2]
        lim = [mean(x) for x in ('Block Limit Registers', 'Block Limit Shared Mem', 'Block Limit Warps', 'Block Limit SM')]
        out[k] = {'n': len(ms), 'ncu_us': mean('Duration'), 'sm_pct': mean('Compute (SM) Throughput'),
                  'mem_pct': mean('Memory Throughput'), 'l2_hit': mean('L2 Hit Rate'),
                  'occ': mean('Achieved Occupancy'), 'cta_per_sm': min(lim) if all(x == x for x in lim) else None,
                  'stall': ', '.join(f'{s} {v / tot * 100:.0f} %' for s, v in top) if tot else ''}
    return out
PHASE = {'vision': 'vision', 'prefill': 'vlm_prefix'}

def e13d_bytes():
    """pi0 bf16_both rung prefix, both frames: per (kernel name, grid) mean fills / writes / unique input bytes."""
    agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for r in csv.DictReader(open(os.path.join(R, '20260926_E13d_pi0_prefix_kernel_bytes_MAXN/kernel_bytes.csv'))):
        g = tuple(int(x) for x in r['grid'].strip('()').split(','))
        a = agg[(norm(r['kernel']), g)]
        a[0] += 1; a[1] += float(r['fills_B']); a[2] += float(r['writes_B'])
        if r['weight_B']:
            a[3] += float(r['weight_B']) + float(r['activation_B']); a[4] += float(r['output_B'] or 0)
    return {k: [v[0]] + [x / v[0] for x in v[1:]] for k, v in agg.items()}
def norm(n):
    n = n.replace('at::native::', 'at::')
    n = re.sub(r'\((int|unsigned long|bool|long)\)', '', n)
    return re.sub(r'\s+', '', n)

def lookup(table, name, grid):
    """Join a kernel to a (normalised name, grid) table: exact, else the unique same-grid entry sharing the longest
    name prefix (nsys, ncu and E13d print template arguments differently)."""
    n = norm(name)
    if (n, grid) in table:
        return table[(n, grid)]
    for L in (120, 90, 60, 40):
        hit = [v for (k, g), v in table.items() if g == grid and k[:L] == n[:L]]
        if len(hit) == 1:
            return hit[0]
    return None

# ---------------------------------------------------------------- build the class table
def build():
    rows = []
    ctr_cache = {}
    e13 = e13d_bytes()
    for model in ('pi0', 'groot_n17', 'smolvla'):
        ks, ninf = kernels(model)
        for stage in ('vision', 'prefill'):
            kk = ks[stage]
            # E4g's pi0 capture ran the reduce-overhead rung: its prefill kernels match bf16_both by name and grid, its
            # vision kernels do not (fp32 vision tower, cutlass s1688 / fmha f32), so pi0 vision has no counters
            ctr = {} if (model, stage) == ('pi0', 'vision') else ncu_counters(model, PHASE[stage])
            cls = collections.OrderedDict()
            for k in kk:
                c, shp = classify(model, stage, k)
                e = cls.setdefault(c, {'n': 0, 'ns': 0, 'flops': 0.0, 'ubytes': 0.0, 'grids': collections.Counter(),
                                       'names': collections.Counter(), 'pairs': collections.Counter(), 'fills': 0.0, 'writes': 0.0, 'e13n': 0})
                e['n'] += 1; e['ns'] += k['dur']; e['grids'][k['grid']] += 1; e['names'][k['name']] += 1; e['pairs'][(k['name'], k['grid'])] += k['dur']
                if shp:
                    for M, N, K, B, share in shp:
                        e['flops'] += 2 * M * N * K * B * share
                        e['ubytes'] += 2 * (M * K + K * N + M * N) * B * share
                if model == 'pi0':
                    b = lookup(e13, k['name'], k['grid'])
                    if b:
                        e['fills'] += b[1]; e['writes'] += b[2]; e['e13n'] += 1
            kernel_sum = sum(k['dur'] for k in kk) / 1e6 / ninf
            un = union_ms(kk) / ninf
            wall = WALL[(model, stage)]
            for c, e in cls.items():
                ms = e['ns'] / 1e6 / ninf
                fl_ms = e['flops'] / PEAK_TF / 1e9 / ninf
                by_ms = e['ubytes'] / PEAK_GB / 1e6 / ninf
                cnt, cov = [], 0
                for (nm, gg), t in sorted(e['pairs'].items(), key=lambda x: -x[1]):
                    c_ = lookup(ctr, nm, gg)
                    if c_:
                        if not cnt: cnt = [c_]
                        cov += t
                cov = cov / e['ns'] if e['ns'] else 0
                grid = e['grids'].most_common(1)[0][0]
                cps = next((x['cta_per_sm'] for x in cnt if x['cta_per_sm']), None)
                ctas = grid[0] * grid[1] * grid[2] if grid else None
                waves = ctas / (16 * cps) if (ctas and cps) else None
                rows.append({
                    'model': model, 'stage': stage, 'class': c, 'launches_per_inf': e['n'] / ninf,
                    'ms': ms, 'pct_of_wall': ms / wall * 100, 'tflops': (e['flops'] / ninf / 1e12) / (ms / 1e3) if e['flops'] else None,
                    'flop_floor_ms': fl_ms, 'byte_floor_ms': by_ms, 'floor_ms': max(fl_ms, by_ms),
                    'dram_GBs': ((e['fills'] + e['writes']) / ninf / 1e9) / (ms / 1e3) if e['e13n'] else None,
                    'dram_GB': (e['fills'] + e['writes']) / ninf / 1e9 if e['e13n'] else None,
                    'unique_GB': e['ubytes'] / ninf / 1e9 if e['ubytes'] else None,
                    'grid': grid, 'ctas': ctas, 'cta_per_sm': cps, 'waves': waves,
                    'sm_pct': cnt[0]['sm_pct'] if cnt else None, 'mem_pct': cnt[0]['mem_pct'] if cnt else None,
                    'l2_hit': cnt[0]['l2_hit'] if cnt else None, 'occ': cnt[0]['occ'] if cnt else None,
                    'stall': cnt[0]['stall'] if cnt else 'not captured', 'ncu_n': cnt[0]['n'] if cnt else 0, 'ncu_cov': cov,
                })
            rows.append({'model': model, 'stage': stage, 'class': 'idle / launch (wall - kernel union)',
                         'launches_per_inf': 0, 'ms': wall - un, 'pct_of_wall': (wall - un) / wall * 100,
                         'floor_ms': 0.0, 'flop_floor_ms': 0.0, 'byte_floor_ms': 0.0})
            rows.append({'model': model, 'stage': stage, 'class': '_TOTAL', 'ms': wall, 'kernel_sum': kernel_sum,
                         'union': un, 'launches_per_inf': len(kk) / ninf})
    return rows

def get(rows, model, stage, cls):
    hits = [r for r in rows if r['model'] == model and r['stage'] == stage and r['class'] == cls]
    assert len(hits) == 1, (model, stage, cls, len(hits))
    return hits[0]
def getp(rows, model, stage, prefix):
    return [r for r in rows if r['model'] == model and r['stage'] == stage and r['class'].startswith(prefix)]

# ---------------------------------------------------------------- root causes (formulas are the documentation)
def causes(rows):
    C = []
    def add(model, stage, cause, resource, measured, floor, evidence, src, kind='efficiency', conf='high', rec=None):
        C.append({'model': model, 'stage': stage, 'cause': cause, 'resource': resource,
                  'measured_ms': round(measured, 2), 'floor_ms': round(floor, 2),
                  'recoverable_ms_upper': round(measured - floor if rec is None else rec, 2),
                  'evidence': evidence, 'source_file': src, 'kind': kind, 'confidence': conf})
    E5 = {m: f'results/{RUNS[m][0]}/nsys/{RUNS[m][1]}_cuda_gpu_trace.csv' for m in RUNS}
    E4G = lambda m, s: f'results/20260923_E4g_c1_{m}_best/ncu/{m}_best_{PHASE[s]}_details.csv'
    E13 = 'results/20260926_E13d_pi0_prefix_kernel_bytes_MAXN/kernel_bytes.csv'
    SW = 'results/20260926_E13d_pi0_prefix_kernel_bytes_MAXN/ncu/gemm_sweep_raw.csv'
    def ctr(r):
        if r.get('sm_pct') is None: return 'ncu counters not captured'
        return (f"ncu (n={r['ncu_n']}, {r['ncu_cov'] * 100:.0f} % of class time has counters): SM {r['sm_pct']:.0f} %, Mem {r['mem_pct']:.0f} %, L2 hit {r['l2_hit']:.0f} %, "
                f"occ {r['occ']:.0f} %, stalls {r['stall']}")
    def gemm(r):
        s = f"{r['launches_per_inf']:.0f} launches, {r['ms']:.2f} ms, {r['tflops']:.1f} TFLOP/s ({r['tflops'] / PEAK_TF * 100:.0f} % of 33.85)"
        if r.get('waves'): s += f", grid {r['grid']} = {r['ctas']} CTAs, {r['cta_per_sm']:.0f} CTA/SM -> {r['waves']:.2f} waves"
        elif r.get('ctas'): s += f", grid {r['grid']} = {r['ctas']} CTAs"
        if r.get('dram_GB'): s += f", DRAM {r['dram_GB']:.2f} GB vs unique {r['unique_GB']:.2f} GB ({r['dram_GBs']:.0f} GB/s)"
        return s
    sweep = gemm_sweep()

    # ---------------- pi0 prefill
    m, s = 'pi0', 'prefill'
    dn = get(rows, m, s, 'GEMM down 816x2048x16384 (K = 16384)')
    gu = get(rows, m, s, 'GEMM gate/up 816x16384x2048')
    k16 = (min(v for k, v in sweep.items() if k.endswith('K16384')), max(v for k, v in sweep.items() if k.endswith('K16384')))
    k2 = (min(v for k, v in sweep.items() if k.endswith('K2048')), max(v for k, v in sweep.items() if k.endswith('K2048')))
    sw = ', '.join(f"{k.split('_', 1)[1].replace('_', ' ')} {v:.1f}" for k, v in sweep.items() if k[:2] in ('g0', 'g2', 'g3'))
    add(m, s, 'K = 16384 GEMM (MLP down-projection) runs at ~10 TFLOP/s', 'L2 / DRAM latency (memory pipe), not tensor throughput',
        dn['ms'], dn['floor_ms'],
        f"{gemm(dn)}; {ctr(dn)}; isolated sweep TFLOP/s (E13d, same kernel family): {sw}: every K = 16384 shape at any M, N, "
        f"split-K ({k16[0]:.1f}-{k16[1]:.1f}), K = 2048 shapes {k2[0]:.1f}-{k2[1]:.1f} (under ncu); gate/up in the trace (same kernel, K = 2048) {gu['tflops']:.1f} TFLOP/s; at that rate the "
        f"down-projection would take {dn['ms'] * dn['tflops'] / gu['tflops']:.1f} ms; forced Triton autotune cut a pi0 prefix "
        f"203.70 -> 156.57 ms (E4i, other rung). Mechanism (row pitch 32 KiB aliasing vs cp.async pipeline) not isolated",
        f"{E5[m]}; {E4G(m, s)}; {SW}; results/20260923_E4i_pi0_prefix_autotune/forced_autotune.json")
    att = getp(rows, m, s, 'attention') + getp(rows, m, s, 'rope') + [r for r in getp(rows, m, s, 'copies') if 'clone' in str(r)]
    att_ms = sum(r['ms'] for r in getp(rows, m, s, 'attention')) + get(rows, m, s, 'rope (SIMT)')['ms']
    fa = FLOORS[(m, s)][2] / PEAK_TF / 1e9
    qk = get(rows, m, s, 'attention QK^T (8 heads, 816x816x256)'); sm_ = get(rows, m, s, 'attention softmax (SIMT)')
    add(m, s, 'attention unfused: QK^T GEMM + SIMT softmax + PV GEMM + rope, scores (8 x 816 x 816) round-trip DRAM',
        'SIMT cores + DRAM bandwidth (tensor cores idle during softmax)', att_ms, fa,
        f"QK^T {qk['ms']:.2f} ms ({qk['tflops']:.1f} TFLOP/s, {ctr(qk)}); softmax {sm_['ms']:.2f} ms ({ctr(sm_)}"
        + (f", DRAM {sm_['dram_GB']:.2f} GB at {sm_['dram_GBs']:.0f} GB/s" if sm_.get('dram_GB') else '') +
        f"); PV {get(rows, m, s, 'attention PV (8 heads, 816x256x816)')['ms']:.2f} ms; rope {get(rows, m, s, 'rope (SIMT)')['ms']:.2f} ms; "
        f"floor = attention FLOPs 98.18 GFLOP / peak; no flash / SDPA kernel in the compiled prefill",
        f"{E5[m]}; {E4G(m, s)}; {E13}")
    act = get(rows, m, s, 'activation (GELU / SiLU x up)')
    add(m, s, 'GELU(gate) x up as a separate SIMT pass: 3 x 26.7 MB per layer round-trip DRAM', 'DRAM bandwidth (SIMT)',
        act['ms'], 0.0, f"{act['launches_per_inf']:.0f} launches, {act['ms']:.2f} ms"
        + (f", DRAM {act['dram_GB']:.2f} GB at {act['dram_GBs']:.0f} GB/s (peak 187.47)" if act.get('dram_GB') else '')
        + f"; {ctr(act)}; floor 0: fusable into the gate/up GEMM epilogue", f"{E5[m]}; {E4G(m, s)}; {E13}")
    nm = get(rows, m, s, 'norm (+ fused residual / bias)')
    add(m, s, 'RMSNorm (+ residual) SIMT passes', 'DRAM bandwidth / latency (SIMT)', nm['ms'], 0.0,
        f"{nm['launches_per_inf']:.0f} launches, {nm['ms']:.2f} ms; {ctr(nm)}; floor 0: fusable into the neighbouring GEMM prologue / epilogue",
        f"{E5[m]}; {E4G(m, s)}")
    qo = get(rows, m, s, 'GEMM q/o 816x2048x2048'); kv = get(rows, m, s, 'GEMM k/v 816x256x2048')
    add(m, s, 'q/o (split-K 2) and k/v (26 CTAs) projections below peak: wave quantisation / split-K overhead', 'tensor cores (under-filled SMs)',
        qo['ms'] + kv['ms'], qo['floor_ms'] + kv['floor_ms'], f"q/o: {gemm(qo)}; {ctr(qo)}. k/v: {gemm(kv)}; {ctr(kv)}",
        f"{E5[m]}; {E4G(m, s)}")
    add(m, s, 'gate/up GEMM: M 816 tiled to 896 (7 x 128, 9 % padded rows)', 'tensor cores', gu['ms'], gu['floor_ms'],
        f"{gemm(gu)}; {ctr(gu)}; already at {gu['tflops'] / PEAK_TF * 100:.0f} % of the measured peak", f"{E5[m]}; {E4G(m, s)}")
    idle = get(rows, m, s, 'idle / launch (wall - kernel union)')
    tot = get(rows, m, s, '_TOTAL')
    add(m, s, 'launch / idle gaps', 'host / launch', max(idle['ms'], 0), 0.0,
        f"E4c bf16_both wall {tot['ms']:.2f} ms - nsys kernel union {tot['union']:.2f} ms; graphs_both (E14b pad) vlm_prefix "
        f"{E14B[('pad', 'vlm_prefix')]:.2f} ms is below the kernel sum: no idle left once graph-captured",
        f"results/20260922_E4c_pi0_opt_ladder/opt_ladder.csv; {E5[m]}")
    # dead work
    nocam = E14B[('pad', 'vlm_prefix')] - E14B[('nocam', 'vlm_prefix')]
    add(m, s, 'dead work: empty (padding) camera, 256 of 816 prefix tokens', 'all (removes work)', nocam, 0.0,
        f"E14b graphs_both: vlm_prefix pad {E14B[('pad', 'vlm_prefix')]:.2f} -> nocam {E14B[('nocam', 'vlm_prefix')]:.2f} ms (n 180 / 120); "
        f"OD-1 baseline decision (user); trim passed E14b's action criterion", 'results/20260927_E14b_pi0_padtrim_stages_MAXN/stages.csv',
        kind='dead_work', conf='high (measured)')
    tp = E14B[('pad', 'vlm_prefix')] - E14B[('trimprompt', 'vlm_prefix')]
    add(m, s, 'dead work: prompt padding (48 pad tokens)', 'all (removes work)', 0.0, 0.0,
        f"measured trimprompt vlm_prefix {E14B[('trimprompt', 'vlm_prefix')]:.2f} vs pad {E14B[('pad', 'vlm_prefix')]:.2f} ms "
        f"({-tp:+.2f} ms: slower, integer-alignment kernel switch, mod-8 hypothesis); nothing recoverable until aligned",
        'results/20260927_E14b_pi0_padtrim_stages_MAXN/stages.csv', kind='dead_work', conf='high (measured)', rec=0.0)
    layer = (dn['ms'] + gu['ms'] + qo['ms'] + att_ms + act['ms'] + nm['ms'] * 0.5) / 18
    add(m, s, 'dead work: last decoder layer after its K/V (q, attention, o, MLP; final norm)', 'all (removes work)', layer, 0.0,
        f"prefix last_hidden_state discarded (lerobot f0d2b37 modeling_pi0.py:847 `_, past_key_values = ...`); per-layer mean "
        f"of down + gate/up + q/o + attention + GELU + half the norms = {layer:.2f} ms (structural; not measured as a variant)",
        f"{E5[m]}; microarch/01_models/lerobot-f0d2b37/src/lerobot/policies/pi0/modeling_pi0.py", kind='dead_work', conf='medium (structural)')

    # ---------------- pi0 vision
    m, s = 'pi0', 'vision'
    f1 = get(rows, m, s, 'GEMM fc1 256x4304x1152')
    add(m, s, 'fc1 weight re-read: 128x64 tile over N-major raster, 4 M-tiles x 9.9 MB weight > 4 MB L2', 'DRAM bandwidth (re-reads)',
        f1['ms'], f1['floor_ms'], f"{gemm(f1)}; isolated (E13d): M 256 {sweep_get('g1_M256_N4304_K1152'):.1f} TFLOP/s with the 128x64 kernel; "
        f"2x the work runs faster (M 512: {sweep_get('g1_M512_N4304_K1152'):.1f}, N 8608: {sweep_get('g1_M256_N8608_K1152'):.1f} TFLOP/s, 128x128 tile); "
        f"E13d fit weight-reread share 1.66; counters at bf16_both not captured (E4g vision ran the fp32 vision tower)",
        f"{E5[m]}; {E13}; {SW}; results/20260926_E13d_pi0_prefix_kernel_bytes_MAXN/gemm_sweep_fit.json")
    q = get(rows, m, s, 'GEMM q/k/v/o 256x1152x1152'); f2 = get(rows, m, s, 'GEMM fc2 256x1152x4304')
    add(m, s, 'q/k/v/o (separate, N 1152) and fc2: 36 CTAs per GEMM on 16 SMs (wave quantisation, short K loop)', 'tensor cores (under-filled SMs)',
        q['ms'] + f2['ms'], q['floor_ms'] + f2['floor_ms'], f"q/k/v/o: {gemm(q)}. fc2: {gemm(f2)}; counters at bf16_both not captured",
        f"{E5[m]}; {E13}")
    at = get(rows, m, s, 'attention kernel (fused)')
    add(m, s, 'attention: FlashAttention-2 with head dim 72 padded to 96', 'tensor cores', at['ms'],
        FLOORS[(m, s)][2] / PEAK_TF / 1e9, f"{at['launches_per_inf']:.0f} launches, {at['ms']:.2f} ms; kernel traits <96, 128, 64> for hd 72 (33 % padded MACs)",
        E5[m])
    simt = sum(r['ms'] for r in rows if r['model'] == m and r['stage'] == s and r['class'] in
               ('norm (+ fused residual / bias)', 'activation (GELU / SiLU x up)', 'other elementwise / index / reduce (SIMT)', 'copies / layout / casts'))
    add(m, s, 'SIMT passes: LayerNorm + residual, GELU, bias, casts (Triton)', 'DRAM bandwidth / latency (SIMT)', simt, 0.0,
        'sum of the Triton / elementwise classes; floor 0: fusable into GEMM prologues / epilogues', E5[m])
    cv = get(rows, m, s, 'patch-embed conv (SIMT / layout)')
    add(m, s, 'patch-embed conv on SIMT (implicit_convolve_sgemm, no tensor cores)', 'SIMT cores', cv['ms'],
        3 * 2 * 256 * 1152 * 588 / PEAK_TF / 1e9, f"{cv['ms']:.2f} ms for 3 x 0.35 GFLOP", E5[m])
    idle = get(rows, m, s, 'idle / launch (wall - kernel union)'); tot = get(rows, m, s, '_TOTAL')
    add(m, s, 'launch / idle gaps', 'host / launch', max(idle['ms'], 0), 0.0,
        f"E4c wall {tot['ms']:.2f} - kernel union {tot['union']:.2f} ms; graphs_both vision {E14B[('pad', 'vision')]:.2f} ms",
        f"results/20260922_E4c_pi0_opt_ladder/opt_ladder.csv; {E5[m]}")
    nc = E14B[('pad', 'vision')] - E14B[('nocam', 'vision')]
    add(m, s, 'dead work: empty (padding) camera, 1 of 3 image-encoder calls', 'all (removes work)', nc, 0.0,
        f"E14b graphs_both: vision pad {E14B[('pad', 'vision')]:.2f} -> nocam {E14B[('nocam', 'vision')]:.2f} ms; OD-1 user decision",
        'results/20260927_E14b_pi0_padtrim_stages_MAXN/stages.csv', kind='dead_work', conf='high (measured)')

    # ---------------- GR00T N1.7
    m = 'groot_n17'
    for s in ('vision', 'prefill'):
        idle = get(rows, m, s, 'idle / launch (wall - kernel union)'); tot = get(rows, m, s, '_TOTAL')
        add(m, s, 'host-bound: GPU idle between kernels (no CUDA graph; graph breaks on .item(); host syncs)', 'host / launch',
            idle['ms'], 0.0,
            f"E4d vlm_compile wall {tot['ms']:.2f} ms - nsys kernel union {tot['union']:.2f} ms = {idle['ms']:.2f} ms idle "
            f"({idle['pct_of_wall']:.0f} %); {tot['launches_per_inf']:.0f} GPU ops per inference incl. {count_d2h(m, s)} D2H copies "
            + ("(vision cannot be compiled: flash_attn_varlen SymInt, E4i)" if s == 'vision' else
               "(LLM max-autotune-no-cudagraphs; dynamo graph break at max_length_q.item(), recompile limit hit, E5 stdout)"),
            f"results/20260923_E4d_groot_n17_ladder/raw/rows/vlm_compile.json; {E5[m]}; results/{RUNS[m][0]}/raw/best_stdout.log",
            conf='high (number) / medium (mechanism)')
    s = 'vision'
    simt = [r for r in rows if r['model'] == m and r['stage'] == s and r['class'] in
            ('other elementwise / index / reduce (SIMT)', 'copies / layout / casts', 'norm (+ fused residual / bias)', 'activation (GELU / SiLU x up)', 'rope (SIMT)')]
    add(m, s, 'eager SIMT glue (fp32 elementwise, casts, rope cat, LayerNorm, GELU): uncompiled encoder', 'DRAM bandwidth / latency (SIMT)',
        sum(r['ms'] for r in simt), 0.0,
        '; '.join(f"{r['class']} {r['ms']:.2f} ms" for r in simt if r['ms'] >= 0.005) + "; E14 measured vision DRAM 4.31 GB read + 2.31 GB written vs 0.81 GB analytic",
        f"{E5[m]}; results/20260926_E14_groot_n17_savings_MAXN; reports/E14_prefix_savings.md")
    gm = getp(rows, m, s, 'GEMM')
    add(m, s, 'vision GEMMs (M 1024 / 256) below peak', 'tensor cores', sum(r['ms'] for r in gm), sum(r['floor_ms'] for r in gm),
        '; '.join(f"{r['class']}: {r['tflops']:.1f} TFLOP/s" for r in gm) + '; counters not captured (E4g window = setup kernels)', E5[m])
    cv = get(rows, m, s, 'patch-embed conv3d (tensor core)')
    pc = get(rows, m, s, 'patch-embed conv (SIMT / layout)')
    add(m, s, 'patch-embed conv3d at 2.3 TFLOP/s + NHWC padding', 'tensor cores / layout', cv['ms'] + pc['ms'], cv['floor_ms'],
        f"{cv['ms']:.2f} ms ({cv['tflops']:.1f} TFLOP/s) + nhwcAddPadding {pc['ms']:.2f} ms; {ctr(cv)}", f"{E5[m]}; {E4G(m, s)}")
    at = get(rows, m, s, 'attention kernel (fused)')
    add(m, s, 'attention (flash varlen, 4 images)', 'tensor cores', at['ms'], FLOORS[(m, s)][2] / PEAK_TF / 1e9,
        f"{at['launches_per_inf']:.0f} launches, {at['ms']:.2f} ms", E5[m])
    s = 'prefill'
    lm = get(rows, m, s, 'lm_head 273x151936x2048 (logits discarded)')
    add(m, s, 'dead work: lm_head over all 273 prefix tokens, logits discarded', 'all (removes work)', lm['ms'], 0.0,
        f"{gemm(lm)}; E14 keep1 (logits_to_keep=1) measured -8.1 ms e2e and -1.35 GB reads, bitwise 50/50; floor 0 if dropped entirely",
        f"{E5[m]}; results/20260926_E14_groot_n17_savings_MAXN; reports/E14_prefix_savings.md", kind='dead_work', conf='high (measured)')
    for c, why in (('GEMM gate/up 273x6144x2048 (split-K 2)', 'M 273 tiled to 384 (29 % padded rows), split-K 2'),
                   ('GEMM down 273x2048x6144', 'M 273 tiled to 384, 48 CTAs (1.5 waves)'),
                   ('GEMM q/o 273x2048x2048', 'M 273 on 64-row tiles, memory-latency bound (long_scoreboard)'),
                   ('GEMM k/v 273x1024x2048', '40 CTAs on 16 SMs, memory-latency bound')):
        r = get(rows, m, s, c)
        add(m, s, f"{c.split(' ', 1)[1]}: {why}", 'tensor cores (under-filled) / DRAM latency', r['ms'], r['floor_ms'],
            f"{gemm(r)}; {ctr(r)}", f"{E5[m]}; {E4G(m, s)}")
    simt = [r for r in rows if r['model'] == m and r['stage'] == s and not r['class'].startswith(('GEMM', 'lm_head', 'idle', '_TOTAL', 'attention'))]
    add(m, s, 'SIMT glue: eager rope-index / masked-scatter setup, casts, norms, SiLU x up', 'DRAM latency (SIMT) + host syncs', sum(r['ms'] for r in simt), 0.0,
        '; '.join(f"{r['class']} {r['ms']:.2f} ms" for r in simt if r['ms'] >= 0.005), E5[m])
    at = get(rows, m, s, 'attention kernel (fused)')
    add(m, s, 'attention (flash)', 'tensor cores', at['ms'], FLOORS[(m, s)][2] / PEAK_TF / 1e9, f"{at['ms']:.2f} ms", E5[m])

    # ---------------- SmolVLA
    m, s = 'smolvla', 'vision'
    at = get(rows, m, s, 'attention kernel (fused)')
    add(m, s, 'attention on the memory-efficient (cutlassF) kernel, not flash: a mask is materialised per layer', 'tensor cores (low utilisation)',
        at['ms'], FLOORS[(m, s)][2] / PEAK_TF / 1e9,
        f"{at['launches_per_inf']:.0f} launches, {at['ms']:.2f} ms = {FLOORS[(m, s)][2] / 1e9 / at['ms']:.1f} TFLOP/s on 77.3 GFLOP; {ctr(at)}; "
        f"preceded by triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_expand (mask build)",
        f"{E5[m]}; {E4G(m, s)}")
    gm = getp(rows, m, s, 'GEMM q/k/v') + getp(rows, m, s, 'GEMM o ')
    add(m, s, 'q/k/v/o GEMMs (N 768, 96 CTAs, 3 waves) below peak', 'tensor cores', sum(r['ms'] for r in gm), sum(r['floor_ms'] for r in gm),
        '; '.join(f"{r['class']}: {gemm(r)}; {ctr(r)}" for r in gm), f"{E5[m]}; {E4G(m, s)}")
    gm2 = getp(rows, m, s, 'GEMM fc') + getp(rows, m, s, 'GEMM connector')
    add(m, s, 'fc1 / fc2 / connector GEMMs', 'tensor cores', sum(r['ms'] for r in gm2), sum(r['floor_ms'] for r in gm2),
        '; '.join(f"{r['class']}: {r['tflops']:.1f} TFLOP/s" for r in gm2), E5[m])
    cv = get(rows, m, s, 'patch-embed conv (SIMT / layout)')
    add(m, s, 'patch-embed conv on SIMT (implicit_convolve_sgemm, no tensor cores)', 'SIMT cores', cv['ms'],
        2 * 2 * 1024 * 768 * 768 / PEAK_TF / 1e9, f"{cv['ms']:.2f} ms for 2 x 1.21 GFLOP; {ctr(cv)}", f"{E5[m]}; {E4G(m, s)}")
    simt = [r for r in rows if r['model'] == m and r['stage'] == s and r['class'] in
            ('norm (+ fused residual / bias)', 'activation (GELU / SiLU x up)', 'other elementwise / index / reduce (SIMT)', 'copies / layout / casts')]
    add(m, s, 'SIMT passes: LayerNorm + residual, GELU, mask build, casts (Triton)', 'DRAM bandwidth / latency (SIMT)', sum(r['ms'] for r in simt), 0.0,
        '; '.join(f"{r['class']} {r['ms']:.2f} ms" for r in simt if r['ms'] >= 0.005), E5[m])
    idle = get(rows, m, s, 'idle / launch (wall - kernel union)'); tot = get(rows, m, s, '_TOTAL')
    add(m, s, 'launch / idle gaps', 'host / launch', max(idle['ms'], 0), 0.0, f"E4c wall {tot['ms']:.2f} - kernel union {tot['union']:.2f} ms",
        f"results/20260922_E4c_smolvla_opt_ladder/opt_ladder.csv; {E5[m]}")
    s = 'prefill'
    gm = getp(rows, m, s, 'GEMM')
    add(m, s, 'small-M GEMMs (M 177; 15-60 CTAs on 16 SMs): neither compute- nor bandwidth-bound', 'DRAM latency / under-filled SMs',
        sum(r['ms'] for r in gm), sum(r['floor_ms'] for r in gm),
        '; '.join(f"{r['class']}: {gemm(r)}; {ctr(r)}" for r in gm), f"{E5[m]}; {E4G(m, s)}")
    att = getp(rows, m, s, 'attention') + getp(rows, m, s, 'rope')
    add(m, s, 'attention unfused, QK^T in fp32 on SIMT (ampere_sgemm), SIMT softmax, rope', 'SIMT cores (tensor cores idle)',
        sum(r['ms'] for r in att), FLOORS[(m, s)][2] / PEAK_TF / 1e9,
        '; '.join(f"{r['class']} {r['ms']:.2f} ms ({ctr(r)})" for r in att), f"{E5[m]}; {E4G(m, s)}")
    simt = [r for r in rows if r['model'] == m and r['stage'] == s and r['class'] in
            ('norm (+ fused residual / bias)', 'activation (GELU / SiLU x up)', 'other elementwise / index / reduce (SIMT)', 'copies / layout / casts')]
    add(m, s, 'SIMT passes and KV copies: RMSNorm, SiLU x up, clones, D2D memcpy', 'DRAM latency (SIMT)', sum(r['ms'] for r in simt), 0.0,
        '; '.join(f"{r['class']} {r['ms']:.2f} ms" for r in simt if r['ms'] >= 0.005), E5[m])
    idle = get(rows, m, s, 'idle / launch (wall - kernel union)'); tot = get(rows, m, s, '_TOTAL')
    add(m, s, 'launch / idle gaps', 'host / launch', max(idle['ms'], 0), 0.0, f"E4c wall {tot['ms']:.2f} - kernel union {tot['union']:.2f} ms",
        f"results/20260922_E4c_smolvla_opt_ladder/opt_ladder.csv; {E5[m]}")
    allk = sum(r['ms'] for r in rows if r['model'] == m and r['stage'] == s and not r['class'].startswith(('idle', '_TOTAL')))
    add(m, s, 'dead work: last decoder layer after its K/V', 'all (removes work)', allk / 16, 0.0,
        f"prefix output discarded (lerobot f0d2b37 modeling_smolvla.py:824); 1/16 of the prefill kernel time (upper; K/V of the "
        f"layer still needed)", f"{E5[m]}; microarch/01_models/lerobot-f0d2b37/src/lerobot/policies/smolvla/modeling_smolvla.py",
        kind='dead_work', conf='medium (structural)')
    return C

def count_d2h(model, stage):
    ks, ninf = kernels(model)
    return round(sum(1 for k in ks[stage] if 'Device-to-Host' in k['name']) / ninf)

def gemm_sweep():
    p = os.path.join(R, '20260926_E13d_pi0_prefix_kernel_bytes_MAXN/ncu/gemm_sweep_raw.csv')
    rows = list(csv.reader(open(p)))
    h = rows[0]
    ti = h.index('gpu__time_duration.sum'); ni = h.index('thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg')
    unit = rows[1][ti]
    acc = collections.defaultdict(list)
    for r in rows[2:]:
        mm = re.search(r'(g\d+_M(\d+)_N(\d+)_K(\d+))', r[ni])
        t = float(r[ti]) * {'ms': 1e-3, 'msecond': 1e-3, 'us': 1e-6, 'usecond': 1e-6, 'ns': 1e-9, 'nsecond': 1e-9}[unit]
        acc[mm.group(1)].append((t, 2 * int(mm.group(2)) * int(mm.group(3)) * int(mm.group(4))))
    out = collections.OrderedDict()
    for k, v in acc.items():
        t = sorted(x[0] for x in v)[len(v) // 2]
        out[k] = v[0][1] / t / 1e12
    global _SWEEP; _SWEEP = out
    return out
_SWEEP = {}
def sweep_get(k):
    if not _SWEEP: gemm_sweep()
    return _SWEEP[k]

# ---------------------------------------------------------------- output
def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)): return ''
    return f'{x:.{nd}f}' if isinstance(x, float) else str(x)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    note = check_floors()
    rows = build()
    # FLOP check: GEMM classes vs analytic linear FLOPs
    chk = []
    for (m, s) in (('pi0', 'vision'), ('pi0', 'prefill'), ('groot_n17', 'vision'), ('groot_n17', 'prefill'), ('smolvla', 'vision'), ('smolvla', 'prefill')):
        fl = sum((r.get('flop_floor_ms') or 0) for r in rows if r['model'] == m and r['stage'] == s and r['class'] != '_TOTAL'
                 and not r['class'].startswith(('attention', 'lm_head'))) * PEAK_TF * 1e9
        chk.append((m, s, fl / FLOORS[(m, s)][1]))
    C = causes(rows)
    fields = ['model', 'stage', 'cause', 'resource', 'measured_ms', 'floor_ms', 'recoverable_ms_upper', 'evidence', 'source_file', 'kind', 'confidence']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for c in sorted(C, key=lambda c: -c['recoverable_ms_upper']): w.writerow(c)
    cf = ['model', 'stage', 'class', 'launches_per_inf', 'ms', 'pct_of_wall', 'tflops', 'flop_floor_ms', 'byte_floor_ms', 'floor_ms',
          'dram_GB', 'unique_GB', 'dram_GBs', 'grid', 'ctas', 'cta_per_sm', 'waves', 'sm_pct', 'mem_pct', 'l2_hit', 'occ', 'stall', 'ncu_n', 'ncu_cov']
    with open(os.path.join(OUT_DIR, 'classes.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(cf)
        for r in rows:
            if r['class'] == '_TOTAL': continue
            w.writerow([fmt(r.get(k), 3) if isinstance(r.get(k), float) else ('' if r.get(k) is None else r.get(k)) for k in cf])
    # tables
    L = [f'<!-- generated by analysis/prefix_root_cause.py; {note} -->', '']
    L += ['### Stage closure (per inference; ms)', '',
          '| model | stage | measured wall (untraced) | floor (FLOP / byte) | nsys kernel sum | kernel union | idle = wall - union | kernel sum vs wall | linear-FLOP check |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for (m, s, ratio) in chk:
        t = get(rows, m, s, '_TOTAL'); fl, ff, fb = stage_floor(m, s)
        L.append(f"| {m} | {s} | {t['ms']:.2f} | {fl:.2f} ({ff:.2f} / {fb:.2f}) | {t['kernel_sum']:.2f} | {t['union']:.2f} | "
                 f"{t['ms'] - t['union']:.2f} | {(t['kernel_sum'] / t['ms'] - 1) * 100:+.1f} % | {ratio:.3f} |")
    L += ['', f"pi0 graphs_both (E14b `pad`, n 180): vision {E14B[('pad', 'vision')]:.2f}, vlm_prefix {E14B[('pad', 'vlm_prefix')]:.2f} ms "
          f"(kernel sums above: {get(rows, 'pi0', 'vision', '_TOTAL')['kernel_sum']:.2f} / {get(rows, 'pi0', 'prefill', '_TOTAL')['kernel_sum']:.2f}: "
          f"{(get(rows, 'pi0', 'vision', '_TOTAL')['kernel_sum'] / E14B[('pad', 'vision')] - 1) * 100:+.1f} % / "
          f"{(get(rows, 'pi0', 'prefill', '_TOTAL')['kernel_sum'] / E14B[('pad', 'vlm_prefix')] - 1) * 100:+.1f} %). "
          "GR00T prefill wall includes lm_head (E4d `vlm_prefix` = backbone - vision).", '']
    for m in ('pi0', 'groot_n17', 'smolvla'):
        for s in ('vision', 'prefill'):
            t = get(rows, m, s, '_TOTAL')
            L += [f'### {m} {s}: kernel classes (wall {t["ms"]:.2f} ms)', '',
                  '| class | launches | ms | % wall | TFLOP/s | floor ms | DRAM GB (unique in+out) | GB/s | CTAs / waves | SM % | Mem % | L2 hit % | occ % | top stalls (ncu cover) |',
                  '|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|']
            cl = [r for r in rows if r['model'] == m and r['stage'] == s and r['class'] != '_TOTAL']
            for r in sorted(cl, key=lambda r: -r['ms']):
                if r['ms'] < 0.05 * 1 and not r['class'].startswith('idle'): continue
                dg = f"{r['dram_GB']:.2f} ({r['unique_GB']:.2f})" if r.get('dram_GB') and r.get('unique_GB') else (f"{r['dram_GB']:.2f}" if r.get('dram_GB') else '')
                cw = (f"{r['ctas']} / {r['waves']:.2f}" if r.get('waves') else (str(r['ctas']) if r.get('ctas') else ''))
                L.append(f"| {r['class']} | {r['launches_per_inf']:.0f} | {r['ms']:.2f} | {r['pct_of_wall']:.1f} | {fmt(r.get('tflops'), 1)} | "
                         f"{fmt(r.get('floor_ms'))} | {dg} | {fmt(r.get('dram_GBs'), 0)} | {cw} | {fmt(r.get('sm_pct'), 0)} | {fmt(r.get('mem_pct'), 0)} | "
                         f"{fmt(r.get('l2_hit'), 0)} | {fmt(r.get('occ'), 0)} | {r.get('stall') or ''}{(' (%.0f %%)' % (r['ncu_cov'] * 100)) if r.get('ncu_cov') else ''} |")
            L.append('')
    L += ['### Ranked root causes (recoverable upper bound = measured - floor; `dead_work` rows overlap the efficiency rows)', '',
          '| # | model | stage | cause | resource | measured ms | floor ms | recoverable ms (upper) | kind | confidence |',
          '|---:|---|---|---|---|---:|---:|---:|---|---|']
    for i, c in enumerate(sorted(C, key=lambda c: -c['recoverable_ms_upper']), 1):
        L.append(f"| {i} | {c['model']} | {c['stage']} | {c['cause']} | {c['resource']} | {c['measured_ms']:.2f} | {c['floor_ms']:.2f} | "
                 f"**{c['recoverable_ms_upper']:.2f}** | {c['kind']} | {c['confidence']} |")
    L += ['', '### Efficiency-only recoverable per stage (additive within a stage; bounded by wall - floor)', '',
          '| model | stage | wall | floor | wall - floor | sum of efficiency rows | residual (unattributed) |', '|---|---|---:|---:|---:|---:|---:|']
    for (m, s, _) in chk:
        t = get(rows, m, s, '_TOTAL'); fl = stage_floor(m, s)[0]
        if m == 'groot_n17' and s == 'prefill': fl += 0.0
        e = sum(c['recoverable_ms_upper'] for c in C if c['model'] == m and c['stage'] == s and c['kind'] == 'efficiency')
        L.append(f"| {m} | {s} | {t['ms']:.2f} | {fl:.2f} | {t['ms'] - fl:.2f} | {e:.2f} | {t['ms'] - fl - e:.2f} |")
    open(os.path.join(OUT_DIR, 'tables.md'), 'w').write('\n'.join(L) + '\n')
    # refresh the tables block of the report
    rp = os.path.join(ROOT, 'analysis', 'prefix_root_cause.md')
    if os.path.exists(rp):
        txt = open(rp).read()
        a, b = txt.index('<!-- TABLES -->') + len('<!-- TABLES -->'), txt.index('<!-- /TABLES -->')
        body = '\n'.join(L).replace('### Ranked root causes', '### C. Ranked root causes, full list').replace('### Stage closure', '### B. Stage closure')
        body = body.replace('### Efficiency-only recoverable', '### D. Efficiency-only recoverable')
        open(rp, 'w').write(txt[:a] + '\n' + body + '\n' + txt[b:])
    print('\n'.join(L))
    print(note)

if __name__ == '__main__':
    main()
