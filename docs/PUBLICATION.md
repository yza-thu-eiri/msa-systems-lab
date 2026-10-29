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
| P1 sparse reference execution case study | Published; underlying source/evidence pending |
| P2 routing-to-task case study | Published; underlying source/evidence pending |
| Fixed-support causal contract | [Source](../src/msa_lab/contracts/support.py) and [tests](../tests/test_support.py) published; independent check pending |\n| Masked-dense and gathered forward references | Pending source and dependency review |
| Output/LSE and backward parity tests | Pending |
| P1 benchmark harness, logs, profiler extracts | Pending |
| P2 training and intervention scripts/configs | Pending |
| GPU kernel implementation and speedup measurements | Not yet established |

The causal-support contract is now public as a minimal inspectable reference slice, with a [provenance note](reference/causal-support.md) and a CPU test workflow. Publication of source/tests does **not** by itself certify a public passing test run.\n\nThe README and individual case studies should be updated each time their reproducibility level changes. No performance or training result is promoted to a stronger claim level simply because the prose was published.
