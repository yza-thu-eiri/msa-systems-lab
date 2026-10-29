import torch

from msa_lab.config import ModelConfig
from msa_lab.model.msa_attention import CausalMSAAttention
from msa_lab.contracts.support import block_ids_to_group_mask


def _layer():
    cfg = ModelConfig(
        vocab_size=16, max_seq_len=4, d_model=8, n_layers=1,
        n_q_heads=4, n_kv_heads=2, d_ff=16,
    )
    cfg.validate()
    return CausalMSAAttention(
        cfg, layer_id=0, index_dim=3, block_size=2, top_k=2
    )


def test_full_and_complete_sparse_support_match_under_shared_weights():
    torch.manual_seed(41)
    layer = _layer()
    x = torch.randn(1, 4, 8)
    ids = torch.tensor(
        [[[[0, -1], [0, -1]],
          [[0, -1], [0, -1]],
          [[0, 1], [0, 1]],
          [[0, 1], [0, 1]]]]
    )
    out_full, _ = layer(x, main_mode="full")
    out_sparse, _ = layer(
        x, main_mode="sparse", fixed_block_ids=ids,
    )
    torch.testing.assert_close(out_full, out_sparse, rtol=1e-5, atol=1e-6)


def test_dynamic_selection_returns_valid_causal_support_and_metrics():
    torch.manual_seed(43)
    layer = _layer()
    x = torch.randn(1, 4, 8)
    out, aux = layer(
        x, main_mode="sparse", return_aux=True,
        compute_routing_kl=True, compute_routing_diagnostics=True,
    )
    ids = aux["layer_0.block_ids"]
    counts = aux["layer_0.valid_count"]
    support = block_ids_to_group_mask(ids, seq_len=4, block_size=2)
    assert out.shape == (1, 4, 8)
    assert ids.shape == (1, 4, 2, 2)
    assert (counts == (ids >= 0).sum(-1)).all()
    assert support.shape == (1, 4, 2, 4)
    assert support.any(dim=-1).all()
    assert torch.isfinite(out).all()
    assert torch.isfinite(aux["layer_0.routing_kl"])
