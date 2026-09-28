# The four large `.ncu-rep` files in this directory are NOT committed

| file | bytes | sha256 (first 16) |
|---|---:|---|
| `ae_layer0.ncu-rep` | 92514833 | `10a0c37a7082e89e` |
| `ae_step0.ncu-rep` | 210259318 | `3c0edb0e8373e63a` |
| `vision.ncu-rep` | 192300298 | `d282128a88cece6d` |
| `vlm_prefix.ncu-rep` | 212558463 | `81bced14a293f2b4` |

`committed: false`. GitHub rejects files over 100 MB and protocol §6 caps a committed file
at 50 MB; three of these exceed the hard limit and the fourth exceeds the cap. They are in
`.gitignore` so a broad `git add` cannot stage them.

**Secret scan before any publication (protocol §3.5):** all four were scanned for
`CLAUDE_CODE_MESSAGING_TOKEN`, `ANTHROPIC_API_KEY`, `hf_*`, `ghp_*`, AWS keys, private-key
headers, `password=`, and for environment-variable-shaped strings (`^[A-Z_]{4,}=`).
**Zero hits in all four**, which confirms `.ncu-rep` does not embed the process
environment — unlike `.nsys-rep`, verified on this device to carry system-wide process
metadata and therefore never published.

**Published 2026-09-23 11:55:**
`https://huggingface.co/datasets/yang-gon/vla-orin-ncu-raw`, a **private** dataset repo
(707.6 MB). Uploaded byte counts match the sha256 table above exactly. Private because
this is measurement data for an unpublished study; making it public is one call to
`HfApi().update_repo_settings(..., private=False)`.

The upload happened only after the secret scan above came back clean on all four files.
Nothing about `.ncu-rep` is assumed safe by analogy: `.nsys-rep` on this device WAS
verified to embed system-wide process metadata including a session token, and those are
never published.

**Superseded for analysis.** These came from `--set full -c 300`. The 2026-09-23 C1
captures request four sections plus the 14 `smsp__average_warps_issue_stalled_*` metrics
by name and are ~900 KB each after `zstd -19`, answering the same question. These remain
the fuller raw record.

## Update 2026-09-28 (completeness audit): now committed, compressed

`zstd -19` shrinks the four reports to 2.8–8.6 MB (`ncu/<name>.ncu-rep.zst`), under the §6 cap, so they are committed
as protocol §6 prefers; the uncompressed originals stay ignored. Restore with `zstd -d ncu/<name>.ncu-rep.zst`; the
original sha256 of each is in `meta.json.raw_files` and matches the table above.
