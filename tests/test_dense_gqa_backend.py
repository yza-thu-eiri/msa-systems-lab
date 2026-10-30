import math

import torch

from msa_lab.backends.dense_sdpa import DenseSDPABackend


def manual_dense_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, t, hq, d = q.shape
    hkv = k.shape[2]
    group = hq // hkv
    outputs = []
    lses = []
    causal = torch.ones(t, t, dtype=torch.bool).tril()
    for h in range(hq):
        r = h // group
        scores = q[:, :, h].float() @ k[:, :, r].float().transpose(-1, -2)
        scores = scores / math.sqrt(d)
        scores = scores.masked_fill(~causal, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outputs.append(probs @ v[:, :, r].float())
        lses.append(torch.logsumexp(scores, dim=-1))
    return torch.stack(outputs, dim=2), torch.stack(lses, dim=2)


def test_dense_gqa_matches_manual_reference() -> None:
    torch.manual_seed(1)
    q = torch.randn(2, 5, 4, 8)
    k = torch.randn(2, 5, 2, 8)
    v = torch.randn(2, 5, 2, 6)

    actual = DenseSDPABackend()(q, k, v, causal=True, return_lse=True)
    expected_o, expected_lse = manual_dense_gqa(q, k, v)

    torch.testing.assert_close(actual.output.float(), expected_o, rtol=1e-5, atol=1e-5)
    assert actual.lse is not None
    torch.testing.assert_close(actual.lse, expected_lse, rtol=1e-5, atol=1e-5)


def test_dense_backend_rejects_bad_head_ratio() -> None:
    q = torch.randn(1, 4, 3, 8)
    k = torch.randn(1, 4, 2, 8)
    v = torch.randn(1, 4, 2, 8)
    try:
        DenseSDPABackend()(q, k, v, causal=True)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("expected invalid GQA ratio to fail")
