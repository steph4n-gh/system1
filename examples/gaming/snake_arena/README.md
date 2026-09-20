# Snake arena: System1 × Laya × Jev

A local browser GUI with three live Snake boards, timed races, equal move budgets,
direction probabilities, latency percentiles, safety-intervention counts, replay
and JSON export. System1 runs a taught skill on CPU, Laya runs through MLX on GPU,
and Jev makes a real TypeSafe API request for each decision.

The [Laya-MLX project](https://github.com/mizchi/laya-mlx/tree/dc3aa6b150cb861d0788fbd421cfd1303de4ed57)
provided the original Snake rules, planner and compact question wording. Its
Python demo uses Rich; its browser port uses TypeScript and Canvas. This GUI is
a new three-player implementation in plain HTML/Canvas with a Python standard
library server. The exact two-panel webpage in the reference screenshot was not
found. See [attribution](NOTICE.md) and the [measured results](RESULTS.md).

To watch the actual recorded race immediately, with only System1 installed:

```bash
python -m examples.gaming.snake_arena \
  --replay examples/gaming/snake_arena/results/timed-seed2026.json.gz
```

Open http://127.0.0.1:8787 and click **Replay**. This mode uses no API key or MLX
installation and is explicitly labeled as recorded. The full live setup follows.

## Run it

Use a source checkout containing the TF-IDF projector and this example. The
three-player live demo requires an Apple Silicon Mac, Python 3.11+, about 850 MB
for Laya weights, and a working TypeSafe API key. It does not add dependencies to
the System1 package. From the repository root, with [uv](https://docs.astral.sh/uv/):

```bash
uv venv .system1/snake-env
uv pip install --python .system1/snake-env/bin/python -e . \
  'laya-mlx @ git+https://github.com/mizchi/laya-mlx.git@dc3aa6b150cb861d0788fbd421cfd1303de4ed57'
.system1/snake-env/bin/hf download aac6fef/laya-mlx \
  --revision 20aed815fc6acde75733882e7ec0e3f28aeb9717 \
  --local-dir .system1/snake-model
```

Put `TYPESAFE_API_KEY=your-key` in the repository's ignored `.env` file, or set it
in the server environment. Then:

```bash
.system1/snake-env/bin/python -m examples.gaming.snake_arena \
  --model .system1/snake-model --credentials .env
```

Open **http://127.0.0.1:8787**. Initialization teaches and reloads System1's skill,
loads Laya and warms the engines. Jev receives one live warm-up request. Start a
30-second race, then select **Equal moves** for a 120-move comparison. Each Jev
move uses the API; **Stop** ends the run after any in-flight request finishes.
Provider errors stop that lane and never substitute a baseline or invented result.

The server binds only to loopback. Its key stays in the Python process and is not
sent to the page or written to reports. Run reports, the saved skill and teaching
evaluation are written under ignored `.system1/snake/`. **Replay** plays the last
completed trace without making inference requests; **JSON** exports that trace
and its measurements. Keep the server running while using the GUI.

## The skill System1 learns

The planner turns the board into a small, explicit input. For example:

```json
{
  "state": "Safe route: yes. Food reachable through empty cells: yes.",
  "move_descriptions": {
    "UP": "Blocked. Collision.",
    "DOWN": "Safe. Slower route.",
    "LEFT": "Unsafe. Traps the snake.",
    "RIGHT": "Safe. Best route to food."
  },
  "labels": {"move": "RIGHT", "risk": true, "food": true}
}
```

This is an explanatory view of one lesson; [policy.py](policy.py) generates the
actual three-question contract and labels. The `risk` field follows upstream
naming: its answer means **a safe route is available**. System1 is taught from
284 synthetic examples, calibrated on 86 separate examples and checked on 94
held-out examples. Each combination of four move descriptions stays entirely
within one split, including both variants of the food-reachability answer.
Trapped examples teach the Boolean fields and have no correct move label.

The existing TF-IDF projector retains which direction each word belongs to
(`move_UP_safe`, `move_RIGHT_best`, etc.). The existing compiler fits three tiny
decision heads. The resulting skill is saved, reloaded, checked for identical
answers and then used for every live move with the decision cache disabled.
There is no custom Snake inference rule inside System1's player, and no call to a
teacher while it plays. This demonstrates a taught vocabulary, not arbitrary
language understanding or learning Snake strategy from pixels.

## Compare fairly

- All engines receive the same compact planner information and three questions.
  Laya and Jev receive the structured question contract; System1 receives its
  field-namespaced word representation. System1 is specifically taught this
  vocabulary; Laya and Jev use their existing models without task-specific fitting.
- **Timed race** measures throughput and resulting progress. The clock starts
  together; replies after the deadline do not buy extra moves. A lane stops when
  it fills the board. **Equal moves** gives each engine the same move budget,
  subject to death or a five-minute time limit.
- The optional cycle shield changes an unsafe proposal to the model's
  highest-probability safe move. It is identical for all engines and each change
  is recorded. Disable it to compare raw choices. The game uses raw argmax even
  when System1 flags a decision for review; that flag remains visible. This is
  an evaluation mode, not an example of production abstention handling.
- Initial board, dimensions, food, seed and length match. Subsequent food
  locations can differ because food is spawned on an unoccupied cell and paths
  may diverge. Planner agreement refers to each engine's own visited states.
- P50/P95 cover encoding, inference and result formatting, including network time
  for Jev. E2E also includes planning and the game update. Engine loading, warm-up,
  report writing and UI rendering are excluded. The three loops run concurrently
  and share the Mac, so these are application timings, not isolated kernel benchmarks.
- **The planner already knows a safe preferred move.** A direct planner is
  sufficient to solve this game. This comparison demonstrates the cost of
  interpreting a bounded decision contract; it does not prove that any model
  discovered Snake strategy or outperforms another model on unrelated tasks.

Offline verification needs only System1's normal development environment:

```bash
python -m pytest tests/test_snake_arena.py -q
```
