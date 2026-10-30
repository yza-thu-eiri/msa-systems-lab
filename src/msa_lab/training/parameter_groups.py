from __future__ import annotations

import torch

from msa_lab.model.msa_attention import CausalMSAAttention
from msa_lab.model.tiny_lm import TinyCausalLM


def split_main_index_parameters(
    model: TinyCausalLM,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """把所有 MSA 层的 Index 参数与其余 LM/Main 参数精确分开。"""
    index_parameters: list[torch.nn.Parameter] = []
    for block in model.blocks:
        if isinstance(block.attn, CausalMSAAttention):
            index_parameters.extend(block.attn.q_idx_proj.parameters())
            index_parameters.extend(block.attn.k_idx_proj.parameters())

    all_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    index_ids = {id(parameter) for parameter in index_parameters}
    main_parameters = [
        parameter for parameter in all_parameters if id(parameter) not in index_ids
    ]

    main_ids = {id(parameter) for parameter in main_parameters}
    all_ids = {id(parameter) for parameter in all_parameters}
    if main_ids & index_ids:
        raise RuntimeError("Main 与 Index 参数组发生重叠")
    if main_ids | index_ids != all_ids:
        raise RuntimeError("Main 与 Index 参数组没有覆盖全部可训练参数")
    if len(index_ids) != len(index_parameters):
        raise RuntimeError("Index 参数组包含重复参数")

    return main_parameters, index_parameters
