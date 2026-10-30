from __future__ import annotations


def main_mode_for_step(global_step: int, warmup_steps: int) -> str:
    output: str
    if global_step < warmup_steps:
        output = "full"
    else:
        output = "sparse"

    return output


def main_mode_for_policy(
    global_step: int,
    warmup_steps: int,
    policy: str,
) -> str:
    """按训练策略解析 Main execution mode；scheduled 保留 C05 原语义。"""
    if policy == "full_only":
        return "full"
    if policy == "scheduled":
        return main_mode_for_step(global_step, warmup_steps)
    raise ValueError(f"unsupported main_mode_policy: {policy}")


def main_modes_for_policy(
    global_step: int,
    warmup_steps: int,
    policy: str,
    *,
    n_layers: int,
    fixed_modes: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """将 segment policy 解析为逐层执行状态。"""
    if policy == "fixed_per_layer":
        if fixed_modes is None or len(fixed_modes) != n_layers:
            raise ValueError("fixed_per_layer requires one fixed mode per layer")
        return tuple(fixed_modes)
    mode = main_mode_for_policy(global_step, warmup_steps, policy)
    return (mode,) * n_layers
