# E0 secrets re-test (T001 step 6, A17) — 2026-10-22 — verdict: **NOT CLEAN**

| capture | canary of an unrelated process | `CLAUDE_CODE_MESSAGING_TOKEN` | `CLAUDE_CODE_SESSION_ID` | protocol §3.5 regexes |
|---|---|---|---|---|
| nsys, normal shell (`.nsys-rep`, `.sqlite`) | found in both | found in both | found in both | not found |
| nsys, token variables unset + `HF_HUB_OFFLINE=1` | found in both | found in both | found in both | not found |
| nsys variant `--sample=none --cpuctxsw=none` (information only, live canary2) | found in `.sqlite` | found in both | found in both | not found |
| ncu `--set full` via `env/rootdo.sh` (`env -i`), live canary2 | not found | not found | not found | not found |
| ncu toy reports of `results/20261022_E0_toolcheck/raw/` | not found | not found | not found | not found |

nsys records the environment of processes it did not launch, as on the Orin: `.nsys-rep` and `.sqlite` stay in
`scratch/`, listed in `meta.json.raw_files` with `committed: false`; `.gitignore` is unchanged; the `nsys stats`
CSV exports are the committed record. ncu reports may be committed, each scanned first. Per file and class, with
byte offsets only: `verdict_per_file_class.csv` and `scan_*.csv`. The matched bytes are never written anywhere.
