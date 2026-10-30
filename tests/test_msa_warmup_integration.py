from unittest.mock import patch

import pytest
import torch

from msa_lab.backends.dense_sdpa import DenseSDPABackend
from msa_lab.config import ModelConfig
from msa_lab.model.msa_attention import CausalMSAAttention
from msa_lab.reference.routing_kl import routing_kl_gqa
from msa_lab.training.mode import main_mode_for_step
from msa_lab.training.state import TrainerState


def _make_attention() -> CausalMSAAttention:
    cfg = ModelConfig(
        vocab_size=16,
        max_seq_len=4,
        d_model=8,
        n_layers=1,
        n_q_heads=4,
        n_kv_heads=2,
        d_ff=16,
    )
    cfg.validate()
    return CausalMSAAttention(
        cfg,
        layer_id=0,
        index_dim=3,
        block_size=2,
        top_k=2,
    )


def test_full_main_matches_dense_backend_with_shared_projections() -> None:
    torch.manual_seed(41)
    attention = _make_attention()
    x_attn = torch.randn(1, 4, 8)

    actual, aux = attention(x_attn, main_mode="full", return_aux=True)

    batch_size, sequence_length, _ = x_attn.shape
    q = attention.q_proj(x_attn).view(
        batch_size, sequence_length, attention.n_q_heads, attention.head_dim
    )
    k = attention.k_proj(x_attn).view(
        batch_size, sequence_length, attention.n_kv_heads, attention.head_dim
    )
    v = attention.v_proj(x_attn).view(
        batch_size, sequence_length, attention.n_kv_heads, attention.head_dim
    )
    expected_main = DenseSDPABackend()(
        q,
        k,
        v,
        causal=True,
        return_lse=True,
    )
    expected = attention.o_proj(expected_main.output.flatten(2, 3))

    torch.testing.assert_close(actual, expected)
    assert expected_main.lse is not None
    torch.testing.assert_close(aux["layer_0.main_lse"], expected_main.lse)

    # FULL diagnostics 可以观察 router selection，但 metadata 不控制 dense Main。
    assert set(aux) == {
        "layer_0.main_lse",
        "layer_0.block_ids",
        "layer_0.valid_count",
        "layer_0.midx",
    }


def test_full_routing_kl_matches_explicit_causal_support_and_gradients() -> None:
    torch.manual_seed(43)
    attention = _make_attention()
    x_attn = torch.randn(1, 4, 8, requires_grad=True)

    _, aux = attention(
        x_attn,
        main_mode="full",
        compute_routing_kl=True,
        return_aux=False,
    )
    actual = aux["layer_0.routing_kl"]

    batch_size, sequence_length, _ = x_attn.shape
    q = attention.q_proj(x_attn).view(
        batch_size, sequence_length, attention.n_q_heads, attention.head_dim
    )
    k = attention.k_proj(x_attn).view(
        batch_size, sequence_length, attention.n_kv_heads, attention.head_dim
    )
    x_idx = x_attn.detach()
    q_idx = attention.q_idx_proj(x_idx).view(
        batch_size, sequence_length, attention.n_kv_heads, attention.index_dim
    )
    k_idx = attention.k_idx_proj(x_idx).unsqueeze(2)

    positions = torch.arange(sequence_length)
    causal = positions[None, :] <= positions[:, None]
    token_support = causal[None, :, None, :].expand(
        batch_size,
        -1,
        attention.n_kv_heads,
        -1,
    )
    expected = routing_kl_gqa(
        q,
        k,
        q_idx,
        k_idx,
        token_support,
        token_capacity=sequence_length,
    )
    torch.testing.assert_close(actual, expected)

    actual.backward()
    assert x_attn.grad is None
    assert attention.q_proj.weight.grad is None
    assert attention.k_proj.weight.grad is None
    assert attention.q_idx_proj.weight.grad is not None
    assert attention.k_idx_proj.weight.grad is not None
    assert attention.q_idx_proj.weight.grad.abs().sum() > 0
    assert attention.k_idx_proj.weight.grad.abs().sum() > 0


def test_full_training_kl_does_not_run_block_selection() -> None:
    torch.manual_seed(47)
    attention = _make_attention()
    x_attn = torch.randn(1, 4, 8)

    with (
        patch(
            "msa_lab.model.msa_attention.causal_index_blockmax",
            side_effect=AssertionError("FULL training must not run BlockMax"),
        ),
        patch(
            "msa_lab.model.msa_attention.select_topk_blocks",
            side_effect=AssertionError("FULL training must not run Top-k"),
        ),
    ):
        _, aux = attention(
            x_attn,
            main_mode="full",
            compute_routing_kl=True,
            return_aux=False,
        )

    assert set(aux) == {"layer_0.routing_kl"}


def test_full_rejects_sparse_control_metadata() -> None:
    attention = _make_attention()
    x_attn = torch.randn(1, 4, 8)
    fixed_block_ids = torch.zeros(1, 4, 2, 2, dtype=torch.long)

    with pytest.raises(ValueError, match="fixed_block_ids"):
        attention(
            x_attn,
            main_mode="full",
            fixed_block_ids=fixed_block_ids,
        )


def test_unknown_main_mode_is_rejected() -> None:
    attention = _make_attention()
    x_attn = torch.randn(1, 4, 8)

    with pytest.raises(ValueError, match="main_mode"):
        attention(x_attn, main_mode="FULL")


def test_training_state_drives_full_to_sparse_boundary_on_same_attention() -> None:
    torch.manual_seed(53)
    attention = _make_attention()
    optimizer = torch.optim.SGD(attention.parameters(), lr=1e-2)
    state = TrainerState(global_step=0)
    warmup_steps = 2
    x_attn = torch.randn(1, 4, 8)

    module_identity = id(attention)
    parameter_identities = {
        name: id(parameter) for name, parameter in attention.named_parameters()
    }
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    observed_modes: list[str] = []

    def record_main_mode(_module, _args, kwargs) -> None:
        observed_modes.append(kwargs["main_mode"])

    hook = attention.register_forward_pre_hook(record_main_mode, with_kwargs=True)
    try:
        for _ in range(4):
            main_mode = main_mode_for_step(state.global_step, warmup_steps)

            optimizer.zero_grad(set_to_none=True)
            attn_out, aux = attention(
                x_attn,
                main_mode=main_mode,
                compute_routing_kl=True,
            )
            loss = attn_out.square().mean() + aux["layer_0.routing_kl"]
            loss.backward()
            optimizer.step()

            state.global_step += 1
    finally:
        hook.remove()

    assert observed_modes == ["full", "full", "sparse", "sparse"]
    assert state.global_step == 4
    assert id(attention) == module_identity
    assert {
        name: id(parameter) for name, parameter in attention.named_parameters()
    } == parameter_identities
    assert optimizer_parameter_ids == set(parameter_identities.values())
