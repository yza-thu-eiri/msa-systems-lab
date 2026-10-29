# Causal support: the first public reference slice

This is the first reviewed code migration from the private MSA development lab. It is a **semantic contract**, not a sparse kernel or a learned top-k router.

## Inputs and output

```text
Raw selected block IDs: [B, N, Hkv, Ksel]  (integer)
       |
       v
Validate ID range, unique selections, and -1 tail padding
       |
       v
Expand block intervals to candidate token positions
       |
       v
Intersect with 0 <= key < N and causal key <= query
       |
       v
Effective token support: [B, N, Hkv, N] (boolean)
```

A `True` entry means the key token is in a selected block **and** visible to the current causal query. A selected future-only block is a no-op if other valid support remains; a row with no effective visible key is rejected. Raw block order does not affect the resulting mathematical support.

This layer does not define top-k ranking, gather order, learned routing, physical layout, or GPU scheduling.

## Files

- [Implementation](../../src/msa_lab/contracts/support.py): `validate_fixed_block_ids`, `block_ids_to_group_mask`
- [Acceptance tests](../../tests/test_support.py): causal prefix, final partial block, independent GQA groups, order invariance, future-only/no-support rows, invalid metadata

The implementation and test file were copied without semantic changes from the author's frozen Phase 1 development sources. Their source Git blob identities were:

| Original frozen artifact | Source Git blob SHA |
| --- | --- |
| `src/msa_lab/contracts/support.py` | `f9fb3cc150d6d2787c23a28eaed8058f6cd67f4d` |
| Phase 1 C01 acceptance tests | `09dda3aa37f29e5684a4add406158ff4db07b820` |

## Reproduction command

The package targets Python 3.12 and PyTorch 2.11.0.

```bash
python -m pip install -e ".[test]"
python -m pytest -q tests/test_support.py
```

**Evidence boundary:** the full Phase 1 acceptance matrix passed in the private development environment. This public subset has been released for review, but the commands above have **not yet been executed from an independent checkout**. Until a public CI or local rerun is observed, do not label the public tests as passing.

## Why this comes first

Every later oracle, gathered forward implementation, fixed-selection backward path, and routing experiment must agree on what an effective selected token means. Publishing the contract before optimized execution keeps that invariant independently inspectable.
