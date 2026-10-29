"""Tiny CPU walkthrough: Index -> Selection -> Main and routing supervision.

Run after installing the public package:
    python examples/routing_walkthrough.py

This illustrates fixed-budget routing semantics, not throughput or trained quality.
"""
import torch

from msa_lab.config import ModelConfig
from msa_lab.contracts.support import block_ids_to_group_mask
from msa_lab.model.msa_attention import CausalMSAAttention


def main() -> None:
    torch.manual_seed(23)
    cfg = ModelConfig(
        vocab_size=16, max_seq_len=6, d_model=8, n_layers=1,
        n_q_heads=4, n_kv_heads=2, d_ff=16,
    )
    cfg.validate()
    layer = CausalMSAAttention(
        cfg, layer_id=0, index_dim=3, block_size=2, top_k=2,
    )
    x = torch.randn(1, 6, 8)
    out, metrics = layer(
        x,
        main_mode="sparse",
        compute_routing_kl=True,
        return_aux=True,
    )
    selected = metrics["layer_0.block_ids"]
    valid = metrics["layer_0.valid_count"]
    support = block_ids_to_group_mask(selected, seq_len=6, block_size=2)

    print(f"Output shape: {tuple(out.shape)}")
    print(f"Selected blocks at final query: {selected[0, -1].tolist()}")
    print(f"Valid blocks per KV group: {valid[0, -1].tolist()}")
    print(f"Effective causal tokens at final query: {support[0, -1].sum(-1).tolist()}")
    print(f"Routing KL: {metrics['layer_0.routing_kl'].item():.6f}")
    assert out.shape == (1, 6, 8)
    assert (valid == (selected >= 0).sum(-1)).all()
    assert support.any(-1).all()
    assert torch.isfinite(metrics["layer_0.routing_kl"])
    print("PASS: dynamic sparse MSA forward and routing supervision")


if __name__ == "__main__":
    main()
