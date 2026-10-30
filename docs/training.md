# P2 TinyLM: Training, Mode Switching and Exact Resume

**Status:** Public source, a bounded CPU configuration, and regression/CI checks are available. This page describes a tiny synthetic next-token demonstration. It does **not** reproduce the private Tiny Shakespeare Router experiments or their 83,616-position functional audit.

## What is included

The public training path connects a small tied-embedding causal language model to the already released MSA Attention layer:

```text
Synthetic next-token batches / optional pinned Tiny Shakespeare
    ↓
TinyCausalLM (Dense layers + optional MSA layer(s))
    ↓
FULL Main during warmup → SPARSE Main afterward
    ├── Language-model NTP loss
    └── Optional Routing-KL (Index/Router parameters)
    ↓
Main / Index optimizer groups, checkpoint, JSONL metrics
    ↓
Resume and evaluate using restored step, state and data RNG
```

The same-seed [dense-anchor initializer](../src/msa_lab/model/initialization.py) aligns shared LM/Main parameters for controlled Dense/MSA comparisons; added Index projections are initialized separately.

The Main execution path can operate in `full` or `sparse` mode. `scheduled` uses `global_step < warmup_steps` for FULL; later steps use SPARSE. `full_only` and fixed per-layer modes exist for controlled studies, subject to configuration validation. The training loss combines NTP with an optional weighted Routing-KL whose gradient ownership is tested separately.

## Fast, bounded CPU run

Python 3.12 and PyTorch 2.11.0 are required. From the repository root:

```bash
python -m pip install -e ".[test]"
python -m msa_lab.training.cli --config configs/p2_cpu_smoke.json
python -m msa_lab.training.eval_cli \
  --config configs/p2_cpu_smoke.json \
  --checkpoint artifacts/p2-cpu-smoke/checkpoint_final.pt
python -m msa_lab.training.generate_cli \
  --config configs/p2_cpu_smoke.json \
  --checkpoint artifacts/p2-cpu-smoke/checkpoint_final.pt \
  --prompt 0,1,2,3 --max-new-tokens 2
```

[Config](../configs/p2_cpu_smoke.json) uses B=1, N=4, two Transformer layers, one MSA layer, GQA 4:2, `warmup_steps=1`, three training steps, a small synthetic increment task and no external dataset download. These are **wiring checks**, not meaningful language-model performance experiments. For this config, `top_k=2` and `block_size=2` require a starting prefix covering **at least two key blocks**; a one-token prompt fails the selection contract. The CLI example uses four prompt tokens and generates two more.

The run writes `config.json`, `run_manifest.json`, `metrics.jsonl`, and a final `checkpoint_final.pt` under the configured `artifacts/` directory (gitignored). Metrics include layer-level execution mode, Router KL, optimizer/Index gradient norms, step time and token counts. No checkpoint/model weight is bundled in this repository.

### Resume semantics

Checkpoint contents include model, optimizer, scheduler, optional scaler, trainer step/tokens, global RNG state and the independent data-generator state. A resume validates the training contract; model shape, optimizer-defining and mode-policy changes are rejected. Logging, output path and the **additional number of requested steps** may change.

[Exact-resume tests](../tests/test_msa_exact_resume.py) independently compare uninterrupted vs split execution across warmup boundaries, including parameter/optimizer/scheduler/RNG state. A [full TinyLM integration check](../tests/test_public_p2_training_resume.py) also compares the final model from a three-step run to a one-step + two-step resumed run, and checks the FULL → SPARSE mode log.

**Checkpoint safety:** PyTorch's `torch.load(weights_only=False)` loads Python-serialized objects. Only load checkpoints you created or obtained from a trusted source.

## Validation and evidence

| Claim | Evidence |
| --- | --- |
| Dense and MSA model construction | [TinyLM](../src/msa_lab/model/tiny_lm.py), [dense anchor](../src/msa_lab/model/initialization.py), published CPU tests |
| FULL/SPARSE switching and loss wiring | [Mode policy](../src/msa_lab/training/mode.py), [warmup integration tests](../tests/test_msa_warmup_integration.py) |
| Main / Index gradient ownership | [Parameter groups](../src/msa_lab/training/parameter_groups.py), [routing integration tests](../tests/test_msa_routing_integration.py) |
| Checkpoint contract and deterministic split replay | [Checkpoint](../src/msa_lab/training/checkpoint.py), [exact resume tests](../tests/test_msa_exact_resume.py) |
| Runnable training and evidence logging | [Engine](../src/msa_lab/training/engine.py), [CPU config](../configs/p2_cpu_smoke.json), [CI workflow](../.github/workflows/reference-tests.yml) |
| Tiny Shakespeare corpus identity | [SHA-256 pinned loader](../src/msa_lab/data/tiny_shakespeare.py); source text is not mirrored here |

For a public acceptance run, execute `python -m pytest tests/` or inspect the [GitHub Actions workflow](../.github/workflows/reference-tests.yml). The [completed public CPU run](https://github.com/yza-thu-eiri/msa-systems-lab/actions/runs/40988032322) reports **164 passing tests** and successful train, eval and generate CLI steps. A passing CPU run supports the tested semantics and small-model closure only. It does **not** certify high-throughput training, GPU kernel acceleration, multiple-node distributed execution, long-context routing behavior or the previously reported held-out task statistics.

## Next experimental evidence

The separately recorded [routing-to-task case study](case-studies/02-routing-error-and-task-impact.md) requires provenance-checked training configs, checkpoints and paired audit outputs before its historical metrics can be rerun. We intentionally keep that evidence maturity separate from this newly released **synthetic CPU training smoke test**.
