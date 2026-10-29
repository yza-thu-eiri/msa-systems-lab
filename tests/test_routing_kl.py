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
