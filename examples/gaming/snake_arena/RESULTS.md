# Recorded Snake comparison — 20 September 2026

System1 completed the 24×16 board in **13.44 seconds**, with no safety overrides,
using a **16.3 KiB** taught planner-reading skill. Laya ran locally through MLX;
Jev used the real TypeSafe API. All three received the original demo's compact
planner hints. This measures execution of a bounded instruction vocabulary, not
independent discovery of Snake strategy. The planner itself already knows a safe
preferred move.

## Thirty-second race

Seed 2026, initial length 6, common start and deadline, optional cycle shield on:

| Engine | Score | Moves | Inference P50 | Inference P95 | E2E P50 | Shield changes |
|---|---:|---:|---:|---:|---:|---:|
| System1 / NumPy CPU | **378** (board filled) | 34,804 | **0.274 ms** | 0.428 ms | 0.331 ms | 0 |
| Laya English 421M / MLX GPU | 19 | 682 | 40.729 ms | 55.984 ms | 41.149 ms | 0 |
| Jev 1.13.0 / remote API | 5 | 84 | 341.235 ms | 460.450 ms | 341.525 ms | 0 |

System1 matched the planner's preferred direction on all 34,804 moves. It raised
one review flag; this evaluation executes argmax even when flagged. Laya matched
on 554/682 moves and Jev on 84/84. The shield changed no engine's decisions.
Jev made 85 requests; its final response arrived after the deadline and did not
execute a move. Laya likewise had one late response. System1 stopped when it
filled the board; its displayed decisions/second uses its 13.44 seconds of play.

## Equal moves, with the shield off

Each engine received 120 raw moves on each of three seeds. All nine games
survived, and System1 raised no review flags in these runs.

| Seed | System1 score | Laya score | Jev score |
|---|---:|---:|---:|
| 2026 | 6 | 5 | 6 |
| 2027 | 3 | 2 | 3 |
| 2028 | 5 | 3 | 5 |

Across the three runs, System1 and Jev each selected the planner's preferred move
**360/360** times. Laya did so **283/360** times; its other moves remained safe.
Pooled inference P50 was 0.291 ms / 42.400 ms / 315.855 ms for System1 / Laya / Jev.
These are short runs on three seeds, not a general accuracy benchmark. Once paths
diverge, subsequent food placements and visited states can differ.

## Teaching and reproducibility

System1's 284 synthetic teaching examples and 86 calibration examples took
100 ms to compile in this session. All **94/94** held-out planner-language cases
had correct applicable answers; 22 requested review. Six of those 94 cases are
trapped states with no move label, so this is not a count of 94 correct moves.
Save/reload preserved values, probabilities, prediction sets and review flags.
The report includes every held-out prediction in [teaching.json](results/teaching.json).

Hardware: **Apple M4 Pro, 24 GiB**, macOS 27, Python 3.13.5, NumPy 2.5.3,
MLX 0.32.2. Laya used FP16, compiled inference, padding to 16 tokens, three-question
batches and tokenization prefix caching; every move still performed inference.
Loading and warm-up were excluded. Loops ran concurrently; these timings include
their shared-machine contention. This Mac and runtime do not reproduce the
reference screenshot's hardware or exact latency.

All completed measurement runs are included, with every executed move, final
board, probabilities and timings:

- [30-second race, seed 2026](results/timed-seed2026.json.gz)
- [120 raw moves, seed 2026](results/turns-seed2026.json.gz)
- [120 raw moves, seed 2027](results/turns-seed2027.json.gz)
- [120 raw moves, seed 2028](results/turns-seed2028.json.gz)
- [Versions and raw report checksums](results/manifest.json)
- [Exact Python sources used for these measurements](results/recorded-source.json.gz)

The source archive is a gzip-compressed JSON mapping of filenames to source text.
The report's `source_sha256` hashes the concatenated UTF-8 source bytes in filename
order. The final demo additionally verifies Laya's weight hash, supports offline
replay and hardens the local page headers; the measured decision logic is unchanged.
The checkpoint is pinned to `aac6fef/laya-mlx` revision
`20aed815fc6acde75733882e7ec0e3f28aeb9717`; its weight hash is in the manifest.
Game, planner and MLX port provenance are in [NOTICE.md](NOTICE.md).

Replay any recording in the full GUI without MLX, a model download or API access:

```bash
python -m examples.gaming.snake_arena \
  --replay examples/gaming/snake_arena/results/timed-seed2026.json.gz
```

Open http://127.0.0.1:8787 and press **Replay**. The page explicitly labels this
as a recording and disables live race controls. See [README.md](README.md) to run
fresh comparisons. Credentials and model weights are not included in the results.
