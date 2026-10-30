# Publication and Evidence Standards

**Purpose:** This is a curated, outward-facing engineering repository—not a mirror of private course material or a daily progress log.

A result belongs here when a reader can distinguish what was *derived*, what was *observed*, what has been *released*, and what can be *independently rerun*.

## Four levels of evidence

| Level | Meaning | How we present it |
| --- | --- | --- |
| Mathematical analysis | Derived from a specified model and tensor contract | Explain assumptions, formula, and boundary |
| Recorded internal observation | Experiment completed in the development repository, but artifacts not all public | Explicit historical/internal label, exact environment and measurement boundary |
| Publicly inspectable | Relevant source, config, tests, and artifacts have been released | Provide paths and reproduction instructions |
| Independently rerun | Re-executed from the public release with retained measurements | Publish raw results, run environment, sample counts, and statistics |

**These are not interchangeable.** Internal regression counts are not public CI validation. A reduced arithmetic ledger is not a measured GPU speedup. Two-layer tiny-model behavior is not evidence of deep long-context robustness.

## Standard structure for case studies

Each case should clearly identify:

- **Problem:** a concrete unresolved engineering question.
- **Semantics and architecture:** what the implementation computes, including tensor axes and invariants when needed.
- **Controls:** comparisons that actually isolate the named variable; same-support comparisons for sparse correctness.
- **Evidence:** model size, hardware, dtype, shape, config, seed, timing boundary, statistical method, and artifact provenance.
- **Explanation:** mechanism consistent with the evidence, separated from alternative hypotheses.
- **Limitations:** assumptions and untested conditions, explicitly called out.
- **Reproduction:** source commit, test or script, expected outputs, and paths to data once released.

## Explicit public-release boundary

Keep private learning profiles, agent/coaching rules, project discussions, hostnames or access configuration, credentials, and unreviewed dataset/checkpoint provenance out of this public repository.

Transfer only reviewed technical source and required tests. Do not mirror the private repo's branches or commit history. Review copyright, licensing, dataset attribution, code dependencies, and potentially embedded personal or third-party information before publication.

For GPU performance, always distinguish:

1. **Math:** QK/PV FLOPs and number of supported pairs.
2. **Execution:** kernels, synchronization, memory transfer, layout, and scheduler.
3. **Measurement:** latency benchmark methodology versus profiler or trace attribution.
4. **Fairness:** exact callable boundary, named baseline, and appropriate correctness semantics.

A transparent correctness reference is expected to be readable, not fast. Any optimized backend must prove parity against it rather than quietly replacing it.

## Incremental release plan

| Component | Public status |
| --- | --- |
| Project overview | Published |
| P1 sparse reference execution case study | Published with [benchmark harness](../benchmarks/reference/benchmark_forward.py), [A4000 historical timing CSV](../evidence/p1_c05_a4000.csv), and [Profiler tables](../evidence/profiler_tables/minimal_self_time_tables.txt); independent GPU rerun pending |
| P2 routing-to-task case study | [Study 02](case-studies/02-routing-error-and-task-impact.md), [historical values and source hashes](../evidence/p2_historical_summary.json) published; raw paired model audit absent |
| Fixed-support causal contract | [Source](../src/msa_lab/contracts/support.py) and [tests](../tests/test_support.py) published; [public CPU acceptance run passed](https://github.com/yza-thu-eiri/msa-systems-lab/actions/runs/40890498194) |
| Masked-dense and gathered forward references | [Oracle](../src/msa_lab/reference/masked_dense.py), [gathered](../src/msa_lab/reference/gathered_sparse.py), [forward parity guide](reference/forward-parity.md), and [public passing CPU tests](https://github.com/yza-thu-eiri/msa-systems-lab/actions/runs/40891800371) |
| Output/LSE and fixed-selection backward parity tests | [Frozen source](../src/msa_lab/reference/gathered_sparse_backward.py), [7 original gradient acceptance cases](../tests/test_gathered_sparse_backward.py), [independent finite differences](../tests/test_backward_finite_difference.py) published; [51 public tests passed](https://github.com/yza-thu-eiri/msa-systems-lab/actions/runs/40892923837) |
| P1 benchmark harness, logs, profiler extracts | Published; see benchmark README and P1 evidence files |
| P2 routing and Main reference | Published with CPU tests; see routing guide |
| P2 TinyLM training engine, eval/generate, synthetic CPU config and exact resume | [Training source and guide](training.md), [smoke config](../configs/p2_cpu_smoke.json), [public end-to-end resume test](../tests/test_public_p2_training_resume.py); [164 tests + CLI smoke passed](https://github.com/yza-thu-eiri/msa-systems-lab/actions/runs/40988032322) |
| P2 historical sparse-switch and two-layer cases | [Study 03](case-studies/03-adapting-to-sparse-execution.md) · [Study 04](case-studies/04-do-sparse-errors-cascade-across-layers.md), [public provenance guide](p2-experiment-evidence.md); original checkpoints/replays unavailable |
| Original P2 diagnostic charts | Four byte-identical private-source plots now published inside [Case Study 02](case-studies/02-routing-error-and-task-impact.md), with [source Git blob hashes](p2-experiment-evidence.md); no claim of independent chart recreation |
| Generated inference modes | Fixed generation policy mismatch; [checkpoint-based FULL/SPARSE/per-layer regression](../tests/test_generate_main_modes.py) available |
| P2 public analysis utilities | [Token transition auditor](../scripts/audit_paired_predictions.py), [C07 logical work estimator](../scripts/analyze_c07_work_crossover.py), [C07 population aggregator](../scripts/aggregate_c07_interaction_audits.py); synthetic contract tests and CI published |
| GPU kernel implementation and speedup measurements | Not yet established |

The frozen support, forward, and manual fixed-selection backward references are public, with [support](reference/causal-support.md), [forward parity](reference/forward-parity.md), and [manual backward](reference/manual-backward.md) explanations. The [public reference workflow](https://github.com/yza-thu-eiri/msa-systems-lab/actions/workflows/reference-tests.yml) now also tests generation-mode correctness, Dense GQA reference behavior and internal Markdown/image links. The CUDA exact-resume test is skipped in CPU-only CI and requires separate GPU execution. The historical C06/C07 result ledger is **a transcription, not a raw dataset**. The original run source had a dirty working tree without a complete retained patch, and the checkpoints/replay records have not been publicly supplied; these limitations block independent reproduction of the recorded model-level results.

The README and individual case studies should be updated each time their reproducibility level changes. No performance or training result is promoted to a stronger claim level simply because the prose was published.
