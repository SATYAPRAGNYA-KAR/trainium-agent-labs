# Verifier-grounded RL experiment (v6)

Additive experiment for `projects/02-kernel-agent`. It reuses `agent.py` (prompts and grader) and
`nkibench.py` (verifier) **unchanged**. All new logic is in the files below.

## Files (replace whole files; nothing else changes)

| File | Status | What it is |
|------|--------|-----------|
| `rl_trace_agent.py` | replaced (v5 -> v6) | the controller |
| `summarize_rl_trace.py` | replaced | per-sample pass@k with intervals, arm table, confound warnings |
| `test_rl_trace_agent.py` | replaced | 64 offline tests, no Neuron SDK and no model needed |
| `README-RL-TRACE.md` | replaced | this file |
| `kernel_lint.py` | **new** | static check for hard-coded tile sizes |
| `holdout.py` | **new** | runs a kernel on shapes the checker never showed it |
| `run_matrix.py` | **new** | arms x levels, interleaved, resumable, with one summary |

## What the v5 A/B said, and what it did not

`plain --hints none` vs `reflect --hints api`, level 2, 5 episodes each, `--no-seed --same-temp`.

**The headline was weaker than it looked.**

- Reflect verified 5/5 and plain 3/5, but **every reflect episode verified in round 0 and stopped**, so
  the repair loop, the restart logic and the lint-in-repair path were never exercised in that arm.
- Per sample in round 0 (no feedback, so independent draws): plain 3/20 = 15%, reflect 7/20 = 35%.
  Wilson 95% intervals are [0.05, 0.36] and [0.18, 0.57]; Fisher exact p = 0.27, and samples inside an
  episode share a prompt, so even that is optimistic. Suggestive, not shown.
- **The arms differed in more than `--mode` and `--hints`.** `plain` is `agent.py`'s prompt only.
  `reflect` also adds the rules card, per-sample variant hints, duplicate re-asks, lint, the probe and the
  located analysis. A gap between them cannot be credited to the API card. v6 lets you ablate each piece
  (`--no-variants`, `--no-analysis`, `--hints`) and the summary prints what every arm contained.
- `solved_attempts=14/88` counted cached grades (58/88 rows) and byte-identical samples (48/88 rows) as
  separate wins.

**What the plain arm showed about failure.**

| Observation | Cause | v6 |
|---|---|---|
| 58% `tile_limits`, nearly all on shape (32, 12): `got src=384, dst=512 / 16384 / 128`, or "index range [0, 127] exceed dimension size of 32" | the kernel allocates or loops to a fixed size instead of the tensor's own | `kernel_lint` + shape rule + a rewritten message (below) |
| Episodes 2 and 4: 5 rounds of the identical kernel, flagged `STUCK-RESTART`, each re-sent the same prompt | in `--mode plain`, `build_prompt` returned `agent.py`'s repair prompt and **ignored `stuck`**. The "restart" restarted nothing. | plain now restarts with a fresh first prompt (`--no-plain-restart` restores v5) |
| The repair prompt contains the failing code, so the model copies it; temperature 0.9 did not help | low-entropy copying; diversity has to come from the prompt, not the sampler | restart = fresh prompt + grounded analysis + one hint tier up |
| Rounds 2 to 5 of those episodes: zero new information | `--patience 5`, and stuck needed 3 identical failures | stuck = 2 identical failures; episode ends after `--max-restarts 2` unchanged restarts |
| Group statistics: 4 identical samples counted as 4 observations | `group_advantages` ran over all rows | advantages over **distinct** kernels; duplicates flagged, not exported, not used to update the bandit |
| `trace_reward 0.0` everywhere | `--trace` was off, the field is a constant 0 | summarizer reports it only for `--trace` rows |
| "static API finding: 0/74" | not a bug: the failures were wrong **sizes**, and every call had a legal signature. `lint_api` cannot see a bad literal. | `kernel_lint` asks where each size came from |

One thing the v5 text did right and v6 keeps: the verifier and the interpreter are more reliable than the
model's opinion of its own mistake.

## What changed in v6

| # | Change | Why |
|---|--------|-----|
| 1 | `kernel_lint.lint_hardcoded_dims` | Taint analysis over the kernel: parameters and anything derived from `.shape` are derived; names bound only to constants are constants; an `nl.ndarray` dimension or a slice bound that folds to a constant >= 2 is hard-coded. "Likely" when a checker input is smaller than the literal on that axis. Attached only to failures of category `tile_limits`, never to a kernel that passed. |
| 2 | `rewrite_tile_feedback` | `agent.enrich` answered a dma size mismatch with an example that allocates a **128x512** tile, the same mistake the trace repeats. v6 replaces it with: the tile must have the shape of the data; here are this checker's input shapes. |
| 3 | `--hints shape` (new default tier) | `none < api < shape < algo`. `shape` = `api` + one paragraph: the checker runs several shapes, some smaller than a tile; read sizes from the tensor. It documents the checker, not the algorithm. The `api` tier's cards are unchanged from v5, but the rest of the harness changed, so a v6 `api` run is not directly comparable to a v5 one. |
| 4 | Probe over **all** shapes for every failing sample | The verifier reports the first failing shape. v6 adds `PER-SHAPE RESULT: (32,12): RAISES ... \| (128,64): passes \| ...` and logs `probe_cases`. |
| 5 | Dense progress from the probe | A shape that passes counts 1, one that runs but is wrong counts half its correct-element fraction, one that raises counts 0. v5 gave every crash 0.30. |
| 6 | Real restarts | `stuck` now means two identical failures (was three) or an all-identical group. A restart is a fresh prompt (every mode, including plain) carrying the grounded analysis of the last failure. |
| 7 | Hint escalation | Each restart climbs one tier (`none -> api -> shape`). It stops before `algo`, which is close to the answer, unless `--escalate-algo`. Rows log `hints_effective`, and the summary warns when it differs from the arm's label. `--no-escalate` turns it off. |
| 8 | Budget | `--patience 3` (was 5) and `--max-restarts 2`. A stuck episode now costs about 3 rounds, not 7+. |
| 9 | Deduplicated learning signal | `dedup_advantages`; bandit updated once per distinct kernel; `--export-groups` and `--export-sft` skip duplicates. |
| 10 | Held-out shapes | `holdout.py`: extra shapes per level, graded with a different input seed, never shown in feedback. Shapes the shipped reference kernel fails are excluded from the verdict. `--holdout report` (default) logs it; `require` makes failing it a failure, with the message as feedback. |
| 11 | `--independent-episodes` | Resets policy, lessons and API facts before each episode. Without it, episode k inherits episodes 0..k-1 and per-episode solve rates are not independent samples. |
| 12 | `--no-analysis` | Skips lint, probe and analysis: the model sees only the verifier's text. The clean ablation of everything computed here. |
| 13 | Summarizer | pass@1 and pass@4 from round 0 with a bootstrap interval over episodes; episodes solved with a Wilson interval; failures counted over distinct kernels; arm labels and a confound report. |
| 14 | `run_matrix.py` | The experiment as a command (below). |

Unchanged: sanitizer, extractor, `ShrunkUCB`, lesson bank, exemplars, SFT export, per-request seed, variant
hints, duplicate re-ask, the level-2 flow check, `--mode bandit`.

## Run

From `projects/02-kernel-agent`:

```bash
python -m py_compile rl_trace_agent.py summarize_rl_trace.py run_matrix.py kernel_lint.py holdout.py
python test_rl_trace_agent.py            # 64 tests, offline
```

### 1. The comparison you meant to run (level 2)

```bash
export KERNEL_AGENT_BASE_URL=http://localhost:8000/v1 KERNEL_AGENT_MODEL=Qwen/Qwen3-8B
python run_matrix.py --levels 2 --episodes 20 \
    --arms plain/none plain/api reflect/none reflect/api reflect/shape \
    -- --context 8192 --no-seed --same-temp
```

`plain/api` is `agent.py`'s prompt through this harness, so it ignores the hint tier by design; keep it only
as a second baseline, or drop it. Isolate one factor at a time with arms that differ in one flag:

```bash
--arms reflect/api reflect/api+no-analysis reflect/api+no-variants reflect/shape reflect/shape+no-escalate
```

### 2. Does it solve the other levels?

```bash
python run_matrix.py --levels 1 2 3 4 --episodes 6 --arms reflect/shape \
    -- --context 8192 --no-seed --same-temp
python summarize_rl_trace.py matrix_runs/*.jsonl --brief
```

Read per level: `solved` with its interval, and `pass@1 r0`. "Best reward" hides how unreliable a level is.
Levels 5 to 7 are graded on HBM traffic and 8 is attention; run them after 1 to 4 are solid
(`--levels 5 6 7 8`; `holdout.py` has shapes for 8 but not for 5 to 7).

### 3. Is a "verified" kernel actually right?

```bash
python holdout.py --level 2 --check my_kernel.py      # exits 1 if it fails a held-out shape
```

### Flags (new or changed)

| Flag | Default | Meaning |
|------|---------|---------|
| `--hints` | `shape` | `none`, `api`, `shape`, `algo` (see change 3). |
| `--patience` | 3 | Rounds without a better kernel before the episode ends (0 = off). |
| `--max-restarts` | 2 | Unchanged restarts before the episode ends (0 = off). |
| `--no-escalate` | off | A restart keeps its hint tier. |
| `--escalate-algo` | off | Let escalation reach the level-2 algorithm note. |
| `--no-plain-restart` | off | v5 behaviour: plain mode re-sends the repair prompt when stuck. |
| `--no-analysis` | off | No lint, probe or located analysis. |
| `--no-probe-all` | off | Probe only the best failing sample of a round. |
| `--holdout` | `report` | `off`, `report`, `require`. |
| `--independent-episodes` | off | Reset learned state per episode. `run_matrix.py` always sets it. |

Unchanged: `--mode`, `--samples`, `--episodes`, `--rounds`, `--trace`, `--seed-references`, `--state`,
`--export-sft`, `--export-groups`, `--no-variants`, `--no-seed`, `--same-temp`, `--no-flow`, `--exploration`,
`--verbose`, `--context`, `--max-tokens`.

Each JSONL row is `schema_version` 7. New fields: `hints_effective`, `restarts`, `escalation`, `duplicate`,
`shape_lint`, `probe_cases`, `holdout`, `config`. Older logs still load; arms are then labelled from
`mode/hints` only.

## Results: `reflect/shape`, levels 1, 3, 4 (Qwen3-8B, v6)

Command (from `projects/02-kernel-agent`, seat-224 pod, simulator only, no device timing):

```bash
python run_matrix.py --levels 1 3 4 --episodes 6 --arms reflect/shape \
    -- --context 8192 --no-seed --same-temp
```

The matrix wrapper expanded this to `--rounds 4 --samples 4 --episodes 5 --patience 3
--independent-episodes --seed 7` per level (so **5 episodes, not 6**). Level 2 was not in this run; its
earlier numbers are in [README.md](README.md) ("One run is not a result"). Logs:
`matrix_runs/reflect_shape_L{1,3,4}.jsonl`.

> **Level 4 is incomplete.** The console capture ends inside episode 4, after round 2. Episodes 1 to 3 are
> complete; episode 4 is partial (best seen so far 0.62); episode 5 and the closing "Learned values" block
> are missing. Rerun `python summarize_rl_trace.py matrix_runs/reflect_shape_L4.jsonl` from the JSONL to fill
> the gaps; the numbers marked † below are from the partial capture.

### Headline

| Level | Operation | Episodes solved | Wilson 95% | Best reward per episode | Samples generated |
|---|---|---|---|---|---|
| 1 | average pooling 2D | **0 / 5** | [0.00, 0.43] | 0.30, 0.30, 0.30, 0.30, 0.30 | 18, 18, 21, 18, 22 = 97 |
| 2 | transpose | not run here | | | |
| 3 | matmul, single tile | **4 / 5** | [0.38, 0.96] | 1.00, 1.00, 1.00, 1.00, 0.30 | 15, 14, 21, 4, 20 = 74 |
| 4 | matmul, tiled | **0 / 3 complete** † | [0.00, 0.56] | 0.30, 0.30, 0.30, then 0.62 (ep 4, partial) | 19, 23, 21, ... |

Reward scale: 0.10 breaks a rule, 0.30 runs but is wrong, 0.62 passes 1 shape of 4, 1.00 passes every
checker shape. A solved level 3 kernel reads `MEMORY BOUND: 19.7 Flops/Byte against a ridge of 222, so 9%
of what the engine could sustain`, identical for every solved sample.

### Where the level 3 solves came from

| Source of the sample | Samples | Verified | Rate |
|---|---|---|---|
| Round 0, `direct` prompt | 20 | 1 (ep 4, r0.2) | 5% |
| Repair rounds, before any restart (prompt carries the failing code) | 20 | **0** | 0% |
| After a `STUCK-RESTART` (fresh prompt + grounded analysis) | 20 | 5 (ep 1 x2, ep 2 x1, ep 3 x2) | 25% |

All three restart solves happened in the first round after the restart. The repair rounds, which are the
core of `reflect`, produced no solve at level 3 in any episode. The restart did the work. Counts use rows
as logged, duplicates included, so treat them as descriptive; the intervals are wide (restart [0.11, 0.47]).

### Level 1: average pooling, 0/5

Every kernel in every episode scored 0.30 except one 0.10 (ep 1, r2.3: called `mean`, a rule violation).
`pass@1` at round 0 is 0/20. Every failing sample passes 0 of 4 shapes, and (32, 32, 32) with pool 2 is the
first failure each time. The failures fall into four families, repeated across all five episodes:

| Family | Representative message | Notes |
|---|---|---|
| Hard-coded tile | `dma_copy ... got src=32768, dst=16384` (also dst=65536, 131072, 1024, 128; `src=4, dst=16384`) | flagged `HARDCODED` / `LINT` by `kernel_lint`; the dominant failure |
| Reduction over a non-trailing axis | `tensor_reduce axis must be the last contiguous dim(s) ... For a 5D tensor, expected axis=(3, 4) ... Got axis=(2, 4)` | model builds a (C, H/p, p, W/p, p) view and reduces `[2, 4]` or `[2, 3]` |
| 1-D result | `SBUF and PSUM tensors must have at least 2 dimensions` | `nl.sum(view, axis=[1, 2])` collapses to 1-D |
| One-offs | `nki.isa has no attribute 'divide'`; `+=` on two tiles; `Partition dim size must be preserved, got 1 -> 4`; out-of-bound on dimension 4 | each seen once |

The same reduction error survived the grounded analysis: ep 1 r1 repeated `axis=(2, 3)` after being told
`axis=(2, 4)` was wrong, so the located analysis changed the axes without changing the idea. The shipped
reference works the other way round: it builds the 5D view with `in_tile.ap([...])` and puts the two pool
axes **last**, so `nl.sum(pool_view, axis=[3, 4])` is legal. None of the failing traces shown mention `.ap()`.

Learned temperature values (shrunk estimate, pulls): `t=0.3: 0.36 (4)`, `t=0.6: 0.37 (2)`, `t=0.9: 0.33 (7)`.
Differences are within noise. Learned API fact: `nisa.nc_matmul` signature (1 hit).

### Level 3: single-tile matmul, 4/5

Ep 5 never solved: 20 samples, all 0.30. Its failures are a different mix from the solved episodes:
`dma_copy` sized 32768 into 8192 (or reversed), `tensor_copy` into `shared_hbm`, `dma_copy(... dst_idx=...)`
(not a real argument), and `value array of shape (32768,) ... indexing result of shape (65536,)`.
Failure families across all five episodes:

| Family | Representative message |
|---|---|
| Tile sized for the wrong tensor | `dma_copy ... got src=8192, dst=32768` (src is lhsT, 128x64 = 8192; the tile was sized 64x512 = 32768, the output shape) |
| Wrong buffer for `nc_matmul` | `moving must be in ['sbuf'], got private_hbm / psum`; `dst must be in ['psum'], got sbuf` |
| Wrong copy primitive | `tensor_copy dst must be in ['sbuf','psum'], got shared_hbm` (needs `dma_copy` for HBM) |
| Invented arguments / reshape | `dma_copy() got an unexpected keyword argument 'dst_idx'`; `cannot reshape array of size 32768 into shape (128,512)`; `'range' object cannot be interpreted as an integer` |
| Python operators on tiles | `unsupported operand type(s) for *: 'NkiTensor' and 'NkiTensor'` |

Learned temperature values: `t=0.3: 0.34 (6)`, `t=0.6: 0.33 (4)`, `t=0.9: 0.33 (4)`. Learned API facts:
`nisa.dma_copy` (9 hits), `nisa.tensor_copy` (1 hit).

**Held-out shapes: 0/0 on every verified kernel.** This is not a pass. `reference_level3.py` asserts
`K == 128`, `M == 64`, `N == 512`, so it fails both held-out shapes, `check_holdout` excludes them as
`ref_fails`, and nothing is left to score. Level 3 also has a single checker shape (`K=128 M=64 N=512`).
A level 3 "VERIFIED" therefore means correct on one shape and untested elsewhere.

### Level 4: tiled matmul, 0/3 complete †

| Episode | Best | Samples | What happened |
|---|---|---|---|
| 1 | 0.30 | 19 | `+=` on tiles (`TypeError ... 'NkiTensor' and 'NkiTensor'`) and `NameError: name 'n'` on the same accumulate line, in every round |
| 2 | 0.30 | 23 | same two errors; the analysis names line 19/20, the next round rewrites it with the same operator |
| 3 | 0.30 | 21 | moves to `Matmul contraction dimension mismatch: stationary[0]=1 != moving[0]=128` (slicing one row as the stationary operand), then after restart `'range' object ...`, `nl.ceil_div` (does not exist), `dst_idx` |
| 4 † | **0.62** | 16+ | r0.1 passes `K=128 M=128 N=512` only; `K=256 M=256 N=1024`, `K=512 M=128 N=512`, `K=256 M=512 N=1024` raise `dma_copy dst partition dimension 256 exceeds maximum 128`. Rounds 1 repeated that kernel (REPEAT / DUP) and never fixed it. After the restart, `'DynamicSlice' object cannot be interpreted as an integer` from `nl.ndarray(shape=(nl.ds(k, 128), M))` and `tensor_copy` into `shared_hbm` |

The 0.62 kernel is the same one the earlier README reports (0.62, one shape of four). It has stayed the
ceiling for this model across README runs, `agent.py`, and now `reflect/shape`.

### What the three levels show together

- **`reflect` plus hint tier `shape` clears level 3 (4/5, one shape) and nothing else.** Level 1 and level 4
  sit where the earlier README had them: 0.30, with 0.62 as the level 4 ceiling.
- **Restarts are the productive mechanism; repair rounds are not**, at least at level 3 (0 of 20 repair
  samples against 5 of 20 post-restart). Level 1 and level 4 restarts did not solve anything.
- **Many rounds carry little new information.** Duplicates are common (`REPEAT DUP cached` rows in every
  episode; 2 to 3 re-asks per group in several), and each episode ends by patience after about 4 rounds.
- **Round-0 `pass@1`:** level 1 0/20, level 3 1/20, level 4 0/16 †. No level reaches a rate that a 5
  episode run can resolve; intervals are wide everywhere.
- **Not shown:** whether `shape` hints help at all (no `none` or `api` arm in this run), and whether the
  analysis or the variants matter (no ablation arm). This table is one arm.

## v7: failure analysis and level-agnostic changes

Written after the level 1/3/4 run above. Nothing here is specific to a level, an operation, or a shape;
`tile_lint.py` and the prompt card contain no pooling, transpose or matmul vocabulary (a test enforces it
for the card). **None of this has been run against the model yet.** It passes 79 offline tests (64 old, 15
new) and the shipped reference kernels lint clean. Whether it raises any solve rate is what the A/B below
measures.

### What the traces say, independent of level

| # | Observation (from the logs above) | Root cause | v7 change |
|---|---|---|---|
| 1 | Level 4 ep 1 and 2: `NameError: name 'n'` then `TypeError: ... 'NkiTensor' * 'NkiTensor'` on the same line, round after round. Level 3 ep 5: `tensor_copy` into `shared_hbm`, found a round after another error. | The simulator stops at the first violation, so the model learns one language rule per round and has 3 to 4 rounds. | `tile_lint.lint_tile_model` reads the code and reports **every** violation at once: undefined names, Python operators between tiles, tiles passed to an op that needs another memory. |
| 2 | Level 1 `nisa.divide` (exists as `nl.divide`); level 4 `nl.ceil_div`, `dst_idx=`. The "closest names" offered for `divide` were `dge_mode, nc_find_index8, activate2`. | Spelling-only suggestions across one module. | `sibling_hint`: the same name in the other nki module, then close spellings, plus a note that size arithmetic is plain Python. |
| 3 | Level 1: `axis=(2, 4)` then `axis=(2, 3)` then `axis=(2, 4)`; the same family in every episode. | The error says what is wrong, not what the call does. | `call_docs`: the real signature and docstring of the call on the failing line, taken from the installed SDK at failure time. |
| 4 | Level 3: 0/20 repair samples solved, 5/20 post-restart samples did. Level 4 ep 4: the 0.62 kernel was reproduced unchanged in 3 repair samples. | A repair prompt contains the failing code and the model copies it. A restart waited for two identical failures. | `--repair-mix 0.5`: in every repair round half the samples (at least one, never sample 0) are asked fresh, with the same grounded analysis. Rows log `prompt_kind`. |
| 5 | Level 3 `Held-out shapes: 0/0` on every verified kernel. | The reference asserts one exact shape, so every held-out shape is excluded and the check is vacuous. | The VERIFIED line now says so ("NONE USABLE ... checker's shapes only"). The check itself is unchanged. |

### Files

| File | Status |
|---|---|
| `tile_lint.py` | **new**: undefined names, tile operators, buffer kinds, sibling-module hints, SDK docs for a call |
| `test_tile_lint.py` | **new**: 15 tests, including "every `reference_level*.py` lints clean" |
| `rl_trace_agent.py` | edited: wires the above in, `--repair-mix`, `--model-card`, `--no-docs`, `--no-lang-lint`, schema 8 |
| `summarize_rl_trace.py` | edited: "Solves by prompt kind" table and a count of failures with a language finding |
| `run_matrix.py` | edited: arm flags `+no-docs`, `+no-lang-lint`, `+no-repair-mix`, `+model-card` |

### Design rules kept from earlier findings

- **Feedback side, not prompt side.** Items 1 to 3 touch only the text attached to a kernel that already
  failed. The README records an earlier prompt addition that made every level worse, so the one change that
  adds text to **every** prompt, the TILE MODEL card, is off by default (`--model-card`) and is its own arm.
- **No hint carries an answer.** The checks say where and why (line, memory, signature); they do not give a
  replacement kernel. The level 1 `.ap()` view is **not** mentioned anywhere: if the SDK docstring for the
  failing call mentions it, the model sees it, otherwise it does not.
- **The buffer rules are the checker's own.** `BUFFER_RULES` holds exactly the constraints the checker
  printed in the traces (`tensor_copy`, `nc_matmul`). Extend it only from verifier messages.
- **Advisory.** Findings are attached only to failures, are logged in `lang_lint`, and a kernel that passes
  is never penalised. `lint_clean` now also counts them in the shaped reward, the same as `lint_api`.

### Run the A/B

Same levels, same seeds, one factor per arm. 10 episodes per arm is the minimum that separates 4/5 from 2/5
at all; the earlier 5-episode intervals were [0.38, 0.96] for the best level.

```bash
python run_matrix.py --levels 1 3 4 --episodes 10 \
    --arms reflect/shape+no-lang-lint+no-docs+no-repair-mix \
           reflect/shape \
           reflect/shape+no-repair-mix \
           reflect/shape+no-docs \
           reflect/shape+model-card \
    -- --context 8192 --no-seed --same-temp
python summarize_rl_trace.py matrix_runs/*.jsonl --brief
```

The first arm approximates v6 (the sibling-module hint inside `lint_api` has no switch, so it is not an
exact v6). The second is the v7 default. Arms 3 and 4 remove one change each; arm 5 adds the card. Read
`Solves by prompt kind` before the arm table: it is the direct test of finding 4.

### Limitations of v7

- **Untested against a model or the real SDK.** `call_docs` and `sibling_hint` are tested only against fake
  `nki` modules; on the pod they use the real ones. If the real docstrings are empty or very long, `call_docs`
  returns nothing or truncates to 700 characters. Check one failing row's `analysis` field first.
- **`lint_tile_operators` is conservative and incomplete.** It flags tile-with-tile operators on names
  bound to `nl.ndarray`-style allocations and on parameters the code reads `.shape` from. It does not follow
  results of other calls, so `nl.sum(x) * nl.sum(y)` is not caught. It does not flag tile-with-scalar,
  because the traces never showed whether that is legal.
- **The undefined-name check is flow-insensitive.** A name bound anywhere counts as bound, so it misses a
  use before assignment; it never flags a legal kernel.
- **Repair-mix is supported by thin evidence**: repair [0.00, 0.16] against post-restart [0.11, 0.47] at
  level 3 only, with duplicate rows counted. It may do nothing at levels where restarts did not help.
- **It cannot give the model what it never sees.** If a level needs a construct the model does not know
  and no docstring describes (level 1's strided view is the example), v7 reports the failure better but does
  not supply the construct. That is the remaining gap, and it is deliberately not closed with a per-level hint.
- Level 4's 0.62 ceiling (one 256-row tile instead of a loop) is not addressed by a specific rule. Nothing in
  v7 tells the model to loop; the dma error message already does, and the model still did not.

## Limitations

- **Results are now in the section above (levels 1, 3, 4, one arm).** The earlier statement that no number existed applies to the level 2 comparison in section 1, which has still not been run at 20 episodes.
- **Original caveat, kept for the harness tests:** the 64 tests were not run against a model or the real simulator. The 64 tests use a NumPy stand-in that enforces the
  signatures and the dma element-count assertion the real verifier printed in your trace. They check the
  harness. A mock-server smoke test of the real CLI path also passed. Whether v6 raises the level-2 solve
  rate on Qwen3-8B is exactly what step 1 above measures; I have no number for it.
- `kernel_lint` is a heuristic. It cannot see a size computed through a helper function or an attribute
  chain it does not model, and it reports a literal as "likely" only against the checker's input shapes.
  It is advisory, attached only to kernels that already failed, and the summary warns if a verified kernel
  ever carried a finding. Levels 3 and 4 legitimately allocate literal 128-row tiles; their findings, if
  any, will be `tile_limits` failures only.
- `holdout.py` shapes for levels 3, 4 and 8 are my choices and are untested against the shipped reference
  kernels on the real SDK. `check_holdout` excludes any shape the reference fails, but run
  `python holdout.py --level N --check reference_levelN.py` once per level and read the output before
  trusting a verdict.
- The per-shape table and the level-2 flow check reveal where outputs came from. That is directional
  feedback, in the spirit of the heat-rod finding that revealing the answer stops the model thinking.
  `--no-flow` removes the flow check; `--no-analysis` removes everything computed.
- Escalation and the `shape` default change what the model is told. Report the tier a result used
  (`hints_effective`), and do not compare a `shape` result to a v5 `api` result as if only the harness changed.
- Re-asking a duplicate costs another generation. When the server is effectively deterministic for a prompt
  the re-ask often returns the same code again; `--no-variants` turns the re-ask off.
- Still prompt-level learning: no model weights change. `nkibench.py` uses the simulator, so this says
  nothing about real Trainium latency.

## Next steps

1. Run section 1 above at 20 episodes per arm and read `pass@1 r0` and its interval before anything else.
2. Run section 2. If a level sits near 0%, read its `PER-SHAPE RESULT` lines before adding hints: the
   failure is usually one shape.
3. GRPO on `--export-groups` rows (`--samples >= 4`; duplicates are already removed). Needs the Neuron
   server stopped while training.
4. Rejection-sampling fine-tune on `--export-sft` rows once the solve rate is nonzero.
5. Flow checks for levels 1, 3 and 4 (trace each output back to its source for data movement; a per-tile
   error map for matmul). Only level 2 has one.