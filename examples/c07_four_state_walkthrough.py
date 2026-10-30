"""Tiny deterministic same-weight FF/SF/FS/SS replay on CPU.

This demonstrates four-state mechanics; it is NOT the historical C07 held-out
population and is not evidence for any of its reported numerical percentages.
"""

from __future__ import annotations

import torch

from msa_lab.config import ModelConfig
from msa_lab.model.tiny_lm import TinyCausalLM


MODES = {
    "FF": ("full", "full"),
    "SF": ("sparse", "full"),
    "FS": ("full", "sparse"),
    "SS": ("sparse", "sparse"),
}


def run_four_state_replay() -> dict[str, float]:
    torch.manual_seed(79)
    cfg = ModelConfig(
        vocab_size=16, max_seq_len=8, d_model=16, n_layers=2,
        n_q_heads=4, n_kv_heads=2, d_ff=32, dropout=0.0,
        msa_layer_ids=(0, 1), msa_index_dim=4, msa_block_size=2, msa_top_k=2,
    )
    model = TinyCausalLM(cfg).eval()
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    targets = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])

    with torch.no_grad():
        output = {
            name: model(ids, targets, main_modes=modes)
            for name, modes in MODES.items()
        }

    logits = {name: item.logits for name, item in output.items()}
    interaction = logits["SS"] - logits["SF"] - logits["FS"] + logits["FF"]
    ff_scale = torch.linalg.vector_norm(logits["FF"])
    interaction_ratio = (torch.linalg.vector_norm(interaction) / ff_scale).item()
    results = {f"ntp_{name.lower()}": float(item.loss) for name, item in output.items()}
    results["four_state_interaction_over_ff_rms"] = interaction_ratio
    results["layer0_only_logits_delta_rms"] = float(
        (logits["SF"] - logits["FF"]).square().mean().sqrt()
    )
    results["layer1_only_logits_delta_rms"] = float(
        (logits["FS"] - logits["FF"]).square().mean().sqrt()
    )
    assert all(torch.isfinite(torch.tensor(x)) for x in results.values())
    return results


if __name__ == "__main__":
    print("Synthetic two-layer CPU same-weight replay:")
    for key, value in run_four_state_replay().items():
        print(f"  {key}: {value:.8f}")
    print("PASS: four execution paths evaluated without changing model weights")
