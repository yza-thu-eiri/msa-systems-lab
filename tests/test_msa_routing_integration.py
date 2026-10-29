import torch

from msa_lab.config import ModelConfig
from msa_lab.contracts.support import block_ids_to_group_mask
from msa_lab.model.msa_attention import CausalMSAAttention
from msa_lab.reference.routing_kl import routing_kl_gqa


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


def test_fixed_support_routing_kl_has_expected_gradient_ownership() -> None:
    """fixed support 跳过 selection，但仍训练 Index Q/K 并阻断 shared/Main 梯度。"""
    torch.manual_seed(17)
    attention = _make_attention()
    x_attn = torch.randn(1, 4, 8, requires_grad=True)

    # 前两个 query 只能看到 block 0；后两个 query 可以看到 block 0/1。
    rows = torch.tensor([[0, -1], [0, -1], [0, 1], [0, 1]])
    fixed_block_ids = rows[None, :, None, :].expand(1, 4, 2, 2).clone()

    _, aux = attention(
        x_attn,
        fixed_block_ids=fixed_block_ids,
        compute_routing_kl=True,
        return_aux=False,
    )

    # routing_kl 是训练量，不应被诊断开关 return_aux=False 隐藏。
    assert set(aux) == {"layer_0.routing_kl"}
    routing_kl = aux["layer_0.routing_kl"]
    assert torch.isfinite(routing_kl)
    routing_kl.backward()

    assert x_attn.grad is None
    assert attention.q_proj.weight.grad is None
    assert attention.k_proj.weight.grad is None
    assert attention.q_idx_proj.weight.grad is not None
    assert attention.k_idx_proj.weight.grad is not None
    assert attention.q_idx_proj.weight.grad.abs().sum() > 0
    assert attention.k_idx_proj.weight.grad.abs().sum() > 0


def test_dynamic_selection_replays_routing_kl_through_fixed_support() -> None:
    """相同最终 block_ids 必须得到相同 Main output 与 routing KL。"""
    torch.manual_seed(23)
    attention = _make_attention()
    x_attn = torch.randn(1, 4, 8)

    real_output, real_aux = attention(
        x_attn,
        compute_routing_kl=True,
        return_aux=True,
    )
    block_ids = real_aux["layer_0.block_ids"]

    fixed_output, fixed_aux = attention(
        x_attn,
        fixed_block_ids=block_ids,
        compute_routing_kl=True,
        return_aux=True,
    )

    torch.testing.assert_close(fixed_output, real_output)
    torch.testing.assert_close(
        fixed_aux["layer_0.routing_kl"],
        real_aux["layer_0.routing_kl"],
    )
    assert "layer_0.midx" in real_aux
    assert "layer_0.midx" not in fixed_aux


def test_zero_routing_weight_disables_index_update() -> None:
    """lambda=0 时 routing 图可以存在，但不能改变 Index projection。"""
    torch.manual_seed(29)
    attention = _make_attention()
    x_attn = torch.randn(1, 4, 8, requires_grad=True)
    rows = torch.tensor([[0, -1], [0, -1], [0, 1], [0, 1]])
    fixed_block_ids = rows[None, :, None, :].expand(1, 4, 2, 2).clone()

    output, aux = attention(
        x_attn,
        fixed_block_ids=fixed_block_ids,
        compute_routing_kl=True,
        return_aux=False,
    )
    main_loss = output.square().mean()
    total_loss = main_loss + 0.0 * aux["layer_0.routing_kl"]
    total_loss.backward()

    assert x_attn.grad is not None
    assert attention.q_proj.weight.grad is not None
    assert attention.k_proj.weight.grad is not None
    assert attention.q_idx_proj.weight.grad is not None
    assert attention.k_idx_proj.weight.grad is not None
    torch.testing.assert_close(
        attention.q_idx_proj.weight.grad,
        torch.zeros_like(attention.q_idx_proj.weight.grad),
    )
    torch.testing.assert_close(
        attention.k_idx_proj.weight.grad,
        torch.zeros_like(attention.k_idx_proj.weight.grad),
    )


def test_fault_without_index_input_detach_reaches_shared_input() -> None:
    """受控绕过 x_attn.detach 后，routing KL 应沿 Index projection 回到输入。"""
    torch.manual_seed(31)
    attention = _make_attention()
    x_attn = torch.randn(1, 4, 8, requires_grad=True)
    rows = torch.tensor([[0, -1], [0, -1], [0, 1], [0, 1]])
    block_ids = rows[None, :, None, :].expand(1, 4, 2, 2).clone()

    batch_size, sequence_length, _ = x_attn.shape
    q = attention.q_proj(x_attn).view(
        batch_size, sequence_length, attention.n_q_heads, attention.head_dim
    )
    k = attention.k_proj(x_attn).view(
        batch_size, sequence_length, attention.n_kv_heads, attention.head_dim
    )

    # Controlled fault：Index projections 直接消费 x_attn，而不是 x_attn.detach()。
    q_idx = attention.q_idx_proj(x_attn).view(
        batch_size, sequence_length, attention.n_kv_heads, attention.index_dim
    )
    k_idx = attention.k_idx_proj(x_attn).unsqueeze(2)
    token_support = block_ids_to_group_mask(
        block_ids,
        seq_len=sequence_length,
        block_size=attention.block_size,
    )
    loss = routing_kl_gqa(
        q,
        k,
        q_idx,
        k_idx,
        token_support,
        token_capacity=sequence_length,
    )
    loss.backward()

    # teacher detach 仍在，所以这条输入梯度只能来自故意打开的 Index 路径。
    assert attention.q_proj.weight.grad is None
    assert attention.k_proj.weight.grad is None
    assert x_attn.grad is not None
    assert x_attn.grad.abs().sum() > 0
