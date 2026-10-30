# P2 experiment evidence: what you can independently run

This directory does **not** contain the historical C06/C07 checkpoint corpus or raw replay records. Its [P2 historical summary ledger](../evidence/p2_historical_summary.json) is a curated transcription from the private, frozen P2 review.

### Three noninterchangeable evidence levels

| Item | Available now? | What a reader can verify |
| --- | --- | --- |
| Model/router/training semantics | **Yes** | P1/P2 public CPU tests, bounded TinyLM training, exact resume |
| Historical experimental summary | **Partly** | Recorded figures, original document blob SHA, locally recorded artifact SHA-256, elementary internal arithmetic |
| Same-weight historical replay and GPU BF16 population | **No** | Needs original checkpoints, data/corpus version, raw replay files and the exact historical source state |

The source checkout used for the original historical experiments had uncommitted modifications without a retained complete patch. A recorded base commit and config hashes do **not** restore that source by themselves. Therefore **we cannot claim exact independent reproduction** of the published historical P2 numerical results, even though the current semantics and tools are public.

## Independently executable analysis tools

**1. Synthetic paired correctness transitions**

```bash
python scripts/audit_paired_predictions.py \
  --input examples/p2_paired_predictions_synthetic.csv
```

The input columns `full_prediction,sparse_prediction,target_token` define per-token paired decisions. The command returns overall accuracy, changed identity rate and the three distinct transition classes. The five synthetic rows are for checking the auditor, not samples from the historical 83,616-position study.

To analyze a real data export, supply your own CSV in the same format. When computing a historical statistic, ensure the input population applies the **same sparse-active mask** and uses identical sample identities and weights.

**2. Recompute mathematical two-layer execution work**

```bash
python scripts/analyze_c07_work_crossover.py \
  --config configs/c07_work_proxy.json \
  --lengths 128 512 2048 \
  --output artifacts/p2-c07-work-proxy.json
```

This is the adapted, published C07 work model. It uses only model geometry, with logical Main QK/AV pair counts and a forward-only arithmetic proxy. It excludes kernel launches, memory traffic, non-matmul operations, backward and actual GPU timings. Sequence lengths beyond `model.max_seq_len=128` are **hypothetical geometry extrapolations**, not supported runtime inputs. Unit tests verify the rounded 128/512/2048 ratios.

**3. Reaggregate four-state held-out batches, if you have the original JSON files**

```bash
python scripts/aggregate_c07_interaction_audits.py \
  --audit-root /path/to/c07_population_interaction \
  --precision CUDA-BF16 \
  --output artifacts/c07_population_summary.json
```

Expected layout: `<audit-root>/<checkpoint-name>/batch_*.json`; the source schema is preserved from the original C07 aggregation script, with CLI input/output paths adapted. It summarizes FF/SF/FS/SS loss, layer-1 support changes, token-level top-1 transitions and non-additivity for each checkpoint. Its public [synthetic contract test](../tests/test_c07_population_aggregation.py) checks ingestion and aggregation.

**Important:** the actual historical JSON inputs are not present. This command cannot reconstruct missing checkpoints or replay files from the historical summary ledger.

## Historical source and artifact provenance

The machine-readable [`p2_historical_summary.json`](../evidence/p2_historical_summary.json) records the original private review document Git blob IDs and the locally recorded SHA-256 IDs of selected artifacts, including:

- C06 token effects summary
- C07 common dual-FF anchor checkpoint
- C07 population interaction summary
- C07 schedule analysis summary

These hashes identify what was recorded **if** the original files can be recovered; they do not attest that those files were uploaded, fetched, rehashed, or reexecuted in this public release.

## Study distinctions

The KL-fitting plateau, C06 layer-1 held-out functional replay, forced-local counterfactual, single-layer 100-step switch, and C07 two-layer population/schedule experiments used different controls and should **not** be combined as one set of paired observations.

C07 synchronous-vs-staged endpoint comparisons use FP32 held-out at their respective matched horizons; the frozen-population four-state replay used CUDA BF16. Neither comparison is a multi-seed confidence interval.

## Future promotion conditions

A case study can move from *recorded internal observation* to *independent reproduction* only after publishing or otherwise providing a provenance-audited original input set, source/runtime snapshot, exact replay command, output schema and newly executed results. A model/weight checkpoint should be loaded only from a trusted source due to Python pickle serialization.
