import math
import torch
from msa_lab.reference.routing_kl import routing_kl_gqa


def test_kl_direction_and_value():
    q = torch.ones(1, 2, 1, 1)
    k = torch.tensor([[[[math.log(.75)]], [[math.log(.25)]]]])
    qi = torch.ones_like(q)
    ki = torch.tensor([[[[math.log(.60)]], [[math.log(.40)]]]])
    support = torch.ones(1, 2, 1, 2, dtype=torch.bool)
    observed = routing_kl_gqa(q, k, qi, ki, support, token_capacity=2)
    expected = .75 * math.log(.75/.60) + .25 * math.log(.25/.40)
    torch.testing.assert_close(observed, torch.tensor(expected), atol=1e-6, rtol=1e-6)


def test_teacher_detached_and_index_trainable():
    q = torch.ones(1, 2, 1, 1, requires_grad=True)
    k = torch.tensor([[[[math.log(.75)]], [[math.log(.25)]]]], requires_grad=True)
    qi = torch.ones(1, 2, 1, 1, requires_grad=True)
    ki = torch.tensor([[[[math.log(.60)]], [[math.log(.40)]]]], requires_grad=True)
    support = torch.ones(1, 2, 1, 2, dtype=torch.bool)
    loss = routing_kl_gqa(q, k, qi, ki, support, token_capacity=2)
    loss.backward()
    assert q.grad is None and k.grad is None
    assert qi.grad is not None and ki.grad is not None
    assert qi.grad.abs().sum() > 0 and ki.grad.abs().sum() > 0


def test_gqa_teacher_averages_probabilities_not_logits():
    q = torch.tensor([[[[2.0], [0.0]], [[2.0], [0.0]]]])
    k = torch.tensor([[[[1.0]], [[-1.0]]]])
    support = torch.ones(1, 2, 1, 2, dtype=torch.bool)
    head_a = torch.softmax(torch.tensor([2.0, -2.0]), dim=-1)
    head_b = torch.tensor([0.5, 0.5])
    avg_probs = (head_a + head_b) / 2
    idx_q = torch.ones(1, 2, 1, 1)
    idx_k = avg_probs.log().reshape(1, 2, 1, 1)
    loss = routing_kl_gqa(q, k, idx_q, idx_k, support, token_capacity=2)
    torch.testing.assert_close(loss, torch.zeros_like(loss), rtol=0, atol=1e-6)


def test_routing_kl_empty_support_is_finite_and_zero():
    q = torch.ones(1, 3, 1, 1)
    k = torch.randn(1, 3, 1, 1)
    qi = torch.ones_like(q, requires_grad=True)
    ki = torch.randn(1, 3, 1, 1, requires_grad=True)
    support = torch.zeros(1, 3, 1, 3, dtype=torch.bool)
    loss = routing_kl_gqa(q, k, qi, ki, support, token_capacity=3)
    assert loss.requires_grad
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.zeros_like(loss))
