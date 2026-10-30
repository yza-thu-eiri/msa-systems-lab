from __future__ import annotations

import torch

from msa_lab.config import ExperimentConfig
from msa_lab.data.synthetic import make_increment_batch
from msa_lab.data.tiny_shakespeare import iter_validation_batches
from msa_lab.model.tiny_lm import TinyCausalLM


ROUTING_DISTRIBUTION_METRICS = (
    "main_teacher_entropy",
    "index_student_entropy",
    "main_teacher_top1_mass",
    "index_student_top1_mass",
)


def evaluate_increment_task(
    model: TinyCausalLM,
    config: ExperimentConfig,
    *,
    batches: int,
    generator: torch.Generator,
    device: torch.device,
    main_mode: str | None = "sparse",
    main_modes: tuple[str, ...] | None = None,
    compute_routing_kl: bool = False,
    compute_routing_diagnostics: bool = False,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    layer_ids = model.cfg.resolved_msa_layer_ids()
    total_routing_kl_by_layer = {layer_id: 0.0 for layer_id in layer_ids}
    total_routing_rows = 0
    routing_diagnostic_sums = {
        name: 0.0 for name in ROUTING_DISTRIBUTION_METRICS
    }
    with torch.no_grad():
        for _ in range(batches):
            input_ids, labels = make_increment_batch(
                batch_size=config.train.batch_size,
                seq_len=config.train.seq_len,
                vocab_size=config.model.vocab_size,
                generator=generator,
                device=device,
            )
            output = model(
                input_ids,
                labels,
                main_mode=main_mode,
                main_modes=main_modes,
                compute_routing_kl=compute_routing_kl,
                compute_routing_diagnostics=compute_routing_diagnostics,
            )
            assert output.loss is not None
            losses.append(float(output.loss.cpu()))
            predictions = output.logits.argmax(dim=-1)
            correct += int((predictions == labels).sum().item())
            total += labels.numel()
            routing_rows = 0
            if compute_routing_kl or compute_routing_diagnostics:
                routing_rows = input_ids.shape[0] * input_ids.shape[1] * model.cfg.n_kv_heads
                total_routing_rows += routing_rows
            if compute_routing_kl:
                for layer_id in layer_ids:
                    routing_kl = output.aux[f"layer_{layer_id}.routing_kl"]
                    total_routing_kl_by_layer[layer_id] += (
                        float(routing_kl.cpu()) * routing_rows
                    )
            if compute_routing_diagnostics:
                for layer_id in layer_ids:
                    for name in ROUTING_DISTRIBUTION_METRICS:
                        value = output.aux[f"layer_{layer_id}.{name}"]
                        key = f"layer_{layer_id}.{name}"
                        routing_diagnostic_sums.setdefault(key, 0.0)
                        routing_diagnostic_sums[key] += float(value.cpu()) * routing_rows
    model.train(was_training)
    result = {
        "eval_loss": sum(losses) / len(losses),
        "eval_accuracy": correct / max(total, 1),
    }
    if compute_routing_kl:
        layer_values = []
        for layer_id, total_value in total_routing_kl_by_layer.items():
            value = total_value / total_routing_rows
            result[f"eval_layer_{layer_id}_routing_kl"] = value
            layer_values.append(value)
        result["eval_routing_kl"] = sum(layer_values) / len(layer_values)
        result["eval_routing_rows"] = float(total_routing_rows)
    if compute_routing_diagnostics:
        for layer_id in layer_ids:
            for name in ROUTING_DISTRIBUTION_METRICS:
                value = routing_diagnostic_sums[f"layer_{layer_id}.{name}"] / total_routing_rows
                result[f"eval_layer_{layer_id}_{name}"] = value
        if len(layer_ids) == 1:
            layer_id = layer_ids[0]
            for name in ROUTING_DISTRIBUTION_METRICS:
                result[f"eval_{name}"] = result[f"eval_layer_{layer_id}_{name}"]
    return result


def evaluate_token_stream(
    model: TinyCausalLM,
    token_ids: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    main_mode: str | None = "sparse",
    main_modes: tuple[str, ...] | None = None,
    compute_routing_kl: bool = False,
    compute_routing_diagnostics: bool = False,
) -> dict[str, float]:
    """顺序扫描 held-out stream；LM loss 按 token、Routing-KL 按有效 row 加权。"""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    layer_ids = model.cfg.resolved_msa_layer_ids()
    total_routing_kl_by_layer = {layer_id: 0.0 for layer_id in layer_ids}
    total_routing_rows = 0
    routing_diagnostic_sums = {
        name: 0.0 for name in ROUTING_DISTRIBUTION_METRICS
    }
    with torch.no_grad():
        for input_ids, labels in iter_validation_batches(
            token_ids,
            batch_size=batch_size,
            seq_len=seq_len,
        ):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            output = model(
                input_ids,
                labels,
                main_mode=main_mode,
                main_modes=main_modes,
                compute_routing_kl=compute_routing_kl,
                compute_routing_diagnostics=compute_routing_diagnostics,
            )
            assert output.loss is not None
            token_count = labels.numel()
            total_loss += float(output.loss.cpu()) * token_count
            total_tokens += token_count
            routing_rows = 0
            if compute_routing_kl or compute_routing_diagnostics:
                routing_rows = input_ids.shape[0] * input_ids.shape[1] * model.cfg.n_kv_heads
                total_routing_rows += routing_rows
            if compute_routing_kl:
                for layer_id in layer_ids:
                    routing_kl = output.aux[f"layer_{layer_id}.routing_kl"]
                    # 合法 causal/local support 保证每个 [b,q,hkv] row 非空。
                    total_routing_kl_by_layer[layer_id] += (
                        float(routing_kl.cpu()) * routing_rows
                    )
            if compute_routing_diagnostics:
                for layer_id in layer_ids:
                    for name in ROUTING_DISTRIBUTION_METRICS:
                        value = output.aux[f"layer_{layer_id}.{name}"]
                        key = f"layer_{layer_id}.{name}"
                        routing_diagnostic_sums.setdefault(key, 0.0)
                        routing_diagnostic_sums[key] += float(value.cpu()) * routing_rows
    model.train(was_training)
    if total_tokens == 0:
        raise ValueError("validation token stream cannot form a complete next-token window")
    result = {
        "eval_loss": total_loss / total_tokens,
        "eval_tokens": float(total_tokens),
    }
    if compute_routing_kl:
        layer_values = []
        for layer_id, total_value in total_routing_kl_by_layer.items():
            value = total_value / total_routing_rows
            result[f"eval_layer_{layer_id}_routing_kl"] = value
            layer_values.append(value)
        result["eval_routing_kl"] = sum(layer_values) / len(layer_values)
        result["eval_routing_rows"] = float(total_routing_rows)
    if compute_routing_diagnostics:
        for layer_id in layer_ids:
            for name in ROUTING_DISTRIBUTION_METRICS:
                value = routing_diagnostic_sums[f"layer_{layer_id}.{name}"] / total_routing_rows
                result[f"eval_layer_{layer_id}_{name}"] = value
        if len(layer_ids) == 1:
            layer_id = layer_ids[0]
            for name in ROUTING_DISTRIBUTION_METRICS:
                result[f"eval_{name}"] = result[f"eval_layer_{layer_id}_{name}"]
    return result
