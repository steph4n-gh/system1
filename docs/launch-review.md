# Launch readiness review — 19 September 2026

The focused observation/cutover and compatibility corrections from the
[complete example review](examples-review.md) are implemented for **0.2.2 beta**.
The complete lifecycle now has an executable [flagship](../examples/observe_routing.py)
and a precise [SDK compatibility contract](typesafe.md). Earlier checks below
are historical checkpoints; the final release verification is recorded at the
end. Publication follows green GitHub CI on the release commit.

Repository: `steph4n-gh/system1`, reviewed from `110038937ee5b0db0553fb3e37a01f31229070cb`. Changes are prepared on local branch `codex/launch-readiness`.

## Findings and changes

| Finding | Evidence and resulting change |
|---|---|
| LangChain could execute denied tools | A harmless sentinel executed through both `invoke()` and `ainvoke()` after an explicit deterministic DENY. LangChain's callback manager swallowed the exception because `raise_error` defaulted to false. The handler now propagates errors and runs inline; real-dispatch regressions verify zero calls for DENY and REQUIRE_APPROVAL. |
| LangChain replaced policy identity with random run IDs | The callback now preserves the configured principal for authorization and success/failure outcome records. Eight tests cover sync/async denial, approval, permitted execution, and execution failure. |
| ASGI request handling could crash or replay indefinitely | Non-object JSON passes downstream, malformed multipart text is ignored, a mid-body disconnect ends processing, and buffered bodies replay once before forwarding subsequent receive events. |
| ASGI could ignore structured uncertainty | Default conformal gating now honors `is_ambiguous` and `escalated_fields`, including when a categorical prediction set is a singleton. |
| Audit metadata exposed a schema digest as a receipt digest | Gateway response headers and MCP error metadata now contain the actual receipt digest when a receipt exists. |
| Release checks missed real integration and package behavior | Added optional LangChain dependencies and dispatcher coverage, Python 3.14 CI, lockfile validation, distribution validation, clean wheel smoke checks, and release-tag/version checks. Corrected the setuptools minimum for SPDX license metadata and refreshed the lockfile. |
| Source distributions omitted support files | Added a manifest including examples, benchmarks, scripts, documentation, assets, and nested tests needed to use and inspect the source release. |
| Public claims exceeded demonstrated behavior | Rewrote the README with executable examples, the complete seed-model quality metrics, and explicit beta/deployment boundaries. Removed missing Docker/Kubernetes directions, fixed static test-count claims, and withdrew unsupported conformal-paper result tables as release evidence. |
| Quality benchmark errors returned success | Failed benchmark runs now retain an error report and return a nonzero exit status. |

LangChain behavior was checked against the installed `langchain-core` 1.6.3 dispatcher and its [official callback reference](https://reference.langchain.com/python/langchain-core/callbacks/base/BaseCallbackHandler). Statistical wording follows the distinction between marginal coverage and conditional acceptance discussed in the [conformal prediction introduction](https://arxiv.org/abs/2107.07511).

## Validation

- Baseline: **981 passed, 14 skipped, 1 warning** on Python 3.13.5.
- Initial launch-fix Python 3.13.5 suite: **1,002 passed, 14 skipped, no warnings**, 41.11 seconds.
- Initial launch-fix Python 3.14.4 suite: **1,002 passed, 14 skipped, no warnings**, 42.56 seconds.
- Python 3.11.12 focused quickstart, LangChain, ASGI, receipt, and externally generated gRPC checks: **22 passed**.
- `uv lock --check`, `uv build`, and `twine check --strict` passed for 0.2.2.
- Clean wheel installed outside the checkout: CLI decision, version/import parity, both proto resources, and license inclusion passed. Installed-wheel gRPC health roundtrip also passed on Python 3.11.
- `pip-audit` reported **no known vulnerabilities** in installed dependencies. The unpublished local package itself is not audited against PyPI advisories.
- Repository Markdown relative-link checks passed. A working-tree pattern scan found no common private-key blocks or GitHub/AWS token patterns; this is not a complete history secret scan.

Tests ran on macOS/Apple M4 Pro. Optional skips covered absent MLX, absent proprietary Game Boy ROMs, and a missing-gRPC branch that cannot run with gRPC installed. Linux CI has been configured but the changed branch has not been pushed, so there is no remote CI result for this patch.

## Quality evidence and release scope

The recorded benchmark has **50% security-triage accuracy**, **51% intent-routing accuracy**, and **3.068 threat-score MAE**. Security BLOCK recall is 0.371, including 8 BLOCK examples classified as ALLOW. Median decision latency was 0.410–0.488 ms. See [raw results and environment](../benchmarks/quality/results/launch_review.json).

These seed-model results support experimentation and local structured routing. They do not justify marketing a general-purpose security detector, guaranteed safe singleton predictions, fixed 95–99% local retention, or production-wide latency guarantees. Deterministic enforcement remains dependent on correct rules, authenticated identity supplied by the application, and protected execution boundaries.

Before announcing the package, land the prepared changes, require the revised CI to pass, and publish the matching 0.2.2 release through the configured PyPI workflow. This review does not alter GitHub environment protections or trusted-publisher settings. See [deployment boundaries](deployment.md) for the assumptions applications must satisfy.

## Teaching follow-up

The classifier-quality work stays within the existing compiler and local runtime.
`compile(examples, augment=False)` teaches only from supplied examples. The CLI
uses that behavior by default with `--dataset`, and `decide --model skill.s1m`
loads the saved skill with strict uncertainty gating. Invalid fields and labels
now fail instead of silently changing the teaching data. Repeated normalized
prompts remain together during example-only calibration splits. Separate explicit
calibration data is supported, and an uncalibrated skill stays uncalibrated after
saving and loading. No runtime dependencies were added for teaching.

The [teaching comparison](../benchmarks/quality/README.md#teaching-comparison)
measured 68% triage and 61% intent accuracy at the unchanged default of 384
features, versus 50% and 51% for starter classifiers. With 2048 features, taught
accuracy was 71% and 68%. These are small development datasets, with no grouping
of related workflows or paraphrases, so the results can be optimistic. Every
case still needed review under strict gating; the calibration sets were too
small to establish useful automatic routing. This work repairs the teaching
workflow, but does not establish production classifier quality.

Validation after the teaching changes:

- Python 3.13.5 full suite: **1,026 passed, 14 skipped**, 40.50 seconds.
- Python 3.11.12 and 3.14.4 teaching/compiler/CLI/calibration/example checks: **54 passed each**.
- Lockfile, distribution build, and strict Twine checks passed.
- Installed 0.2.2 wheel outside the checkout: teach, save, load, and CLI decision passed; uncertainty was preserved.
- Changed documentation links and `git diff --check` passed.

The [small teaching example](../examples/teach_skill.py) and
[teaching guide](guides/training_experts.md) describe the product loop without
requiring a language model or a separate training platform.

## Final 0.2.2 release verification

The completed observation, compatibility and example corrections passed:

- Python 3.13.5 full suite: **1,048 passed, 14 skipped**, 25.87 seconds.
- Python 3.11 and 3.14 focused observation, teaching, primary-example and campaign checks: **68 passed each**.
- Lockfile, wheel/sdist build and strict Twine validation passed.
- Changed Markdown link targets and `git diff --check` passed. A scan of the changed files found no common private-key, GitHub-token or AWS-key patterns.
- The offline flagship promoted after **190 observations** using the normal gates: 38/38 validation labels agreed, 37/38 validation decisions were accepted. After disconnecting the teacher, **40/40 fresh synthetic tickets were correct and accepted**, with identical responses after reload. Its roughly 5 KB artifact and 0.45 ms median local latency are measured demonstration results, not Jev parity evidence.
- The natural-language examples retain their original labels and raw accuracies; strict review was required on 23/24 support, 24/24 model-routing and 22/24 operation-triage cases. All three accepted answers were correct. Updated reports are under `examples/teaching/results/`.

No live Jev quality comparison was performed. Optional MLX and ROM-dependent
checks remain skipped. The release procedure pushes this source to `main`, waits
for its GitHub CI matrix and package job to pass, and only then creates `v0.2.2`
to trigger trusted publishing. Remote CI and publishing runs are the authoritative
record of those later steps.
