from __future__ import annotations

from contextlib import nullcontext
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import math
import sys

import torch

from msa_lab.config import ExperimentConfig
from msa_lab.data.synthetic import make_increment_batch
from msa_lab.data.tiny_shakespeare import (
    CharTokenizer,
    load_tiny_shakespeare_text,
    sample_next_token_batch,
    split_train_valid,
)
from msa_lab.model.tiny_lm import TinyCausalLM
from msa_lab.model.initialization import build_model_from_dense_anchor
from msa_lab.model.msa_attention import CausalMSAAttention
from msa_lab.training.checkpoint import (
    initialize_model_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from msa_lab.training.eval import (
    ROUTING_DISTRIBUTION_METRICS,
    evaluate_increment_task,
    evaluate_token_stream,
)
from msa_lab.training.metrics import (
    JsonlMetricWriter,
    StepMetric,
    StepTimer,
    TensorBoardMetricWriter,
)
from msa_lab.training.mode import main_modes_for_policy
from msa_lab.training.parameter_groups import split_main_index_parameters
from msa_lab.training.state import TrainerState
from msa_lab.utils.rng import seed_everything
from msa_lab.utils.run_manifest import write_run_manifest


@dataclass
class TrainResult:
    final_step: int
    final_loss: float
    tokens_seen: int
    checkpoint_path: Path


def _autocast_context(device: torch.device, dtype_name: str):
    if dtype_name == "float32":
        return nullcontext()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    if device.type == "cpu" and dtype is torch.float16:
        raise RuntimeError("CPU float16 training is not supported by the seed")
    return torch.autocast(device_type=device.type, dtype=dtype)


def _clip_index_gradients(
    index_parameters: list[torch.nn.Parameter],
    index_grad_clip: float | None,
    main_grad_clip: float,
) -> torch.Tensor | None:
    if not index_parameters:
        return None
    threshold = main_grad_clip if index_grad_clip is None else index_grad_clip
    return torch.nn.utils.clip_grad_norm_(index_parameters, threshold)


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> torch.Tensor | None:
    """读取裁剪前梯度的 L2 norm，不修改 gradient。"""
    if not parameters:
        return None
    squared_norm = torch.zeros((), device=parameters[0].device, dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is not None:
            squared_norm += parameter.grad.detach().float().square().sum()
    return squared_norm.sqrt()


def _selected_msa_attentions(model: TinyCausalLM) -> dict[int, CausalMSAAttention]:
    attentions: dict[int, CausalMSAAttention] = {}
    for layer_id in model.cfg.resolved_msa_layer_ids():
        attention = model.blocks[layer_id].attn
        if not isinstance(attention, CausalMSAAttention):
            raise RuntimeError("MSA layer id 没有对应 CausalMSAAttention")
        attentions[layer_id] = attention
    return attentions


def _run_phase(
    has_msa: bool,
    main_modes: tuple[str, ...],
    parameter_update_policy: str,
) -> str:
    if not has_msa:
        return "dense"
    if parameter_update_policy == "index_only":
        suffix = main_modes[0] if len(main_modes) == 1 else "".join(
            mode[0] for mode in main_modes
        )
        return f"msa_{suffix}_index_only"
    suffix = main_modes[0] if len(main_modes) == 1 else "".join(
        mode[0] for mode in main_modes
    )
    return f"msa_{suffix}"


def train(
    config: ExperimentConfig,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> TrainResult:
    config.validate()
    train_cfg = config.train
    seed_everything(train_cfg.seed)

    device = torch.device(train_cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    run_dir = Path(train_cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "config.json")
    write_run_manifest(run_dir / "run_manifest.json", command=sys.argv, cwd=Path.cwd())
    metrics = JsonlMetricWriter(run_dir / "metrics.jsonl")
    tensorboard = (
        TensorBoardMetricWriter(run_dir / "tensorboard")
        if train_cfg.tensorboard
        else None
    )

    model = build_model_from_dense_anchor(config.model, seed=train_cfg.seed).to(device)
    initialized_from_step: int | None = None
    if train_cfg.initialize_from is not None:
        initialization_payload = initialize_model_from_checkpoint(
            train_cfg.initialize_from,
            model=model,
            current_config=config,
            map_location=device,
        )
        initialized_from_step = int(
            initialization_payload["trainer_state"]["global_step"]
        )
    main_parameters, index_parameters = split_main_index_parameters(model)
    index_only = train_cfg.parameter_update_policy == "index_only"
    if index_only:
        for parameter in main_parameters:
            parameter.requires_grad_(False)
    msa_attentions = _selected_msa_attentions(model)
    q_index_initial = {
        layer_id: attention.q_idx_proj.weight.detach().clone()
        for layer_id, attention in msa_attentions.items()
    }
    k_index_initial = {
        layer_id: attention.k_idx_proj.weight.detach().clone()
        for layer_id, attention in msa_attentions.items()
    }
    optimizer_groups: list[dict[str, object]] = []
    if not index_only:
        optimizer_groups.append(
            {
                "params": main_parameters,
                "lr": train_cfg.learning_rate,
                "name": "main",
            }
        )
    index_learning_rate = train_cfg.learning_rate * train_cfg.index_lr_multiplier
    if index_parameters:
        optimizer_groups.append(
            {
                "params": index_parameters,
                "lr": index_learning_rate,
                "name": "index",
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and train_cfg.dtype == "float16")
    )

    state = TrainerState()
    resumed_from = None
    data_generator = torch.Generator(device="cpu")
    data_generator.manual_seed(train_cfg.seed + 1)
    eval_generator = torch.Generator(device="cpu")
    eval_generator.manual_seed(train_cfg.seed + 10_001)

    train_token_ids = None
    valid_token_ids = None
    if train_cfg.corpus_path is not None:
        text = load_tiny_shakespeare_text(Path(train_cfg.corpus_path))
        tokenizer = CharTokenizer.from_text(text)
        if len(tokenizer.id_to_char) != config.model.vocab_size:
            raise ValueError("model.vocab_size must match the Tiny Shakespeare character vocabulary")
        token_ids = tokenizer.encode(text)
        train_token_ids, valid_token_ids = split_train_valid(
            token_ids,
            train_fraction=train_cfg.train_fraction,
        )

    if train_cfg.resume_from:
        payload = load_checkpoint(
            train_cfg.resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            current_config=config,
            map_location=device,
        )
        state = TrainerState.from_dict(payload["trainer_state"])
        data_generator.set_state(
            payload["data_generator_state"].detach().to(
                device="cpu", dtype=torch.uint8
            )
        )
        resumed_from = str(train_cfg.resume_from)
        # fixed batch 始终取 seed 对应的第一批；保存的 generator 已越过首次采样。
        if train_cfg.fixed_train_batch:
            data_generator.manual_seed(train_cfg.seed + 1)
        q_index_initial = {
            layer_id: attention.q_idx_proj.weight.detach().clone()
            for layer_id, attention in msa_attentions.items()
        }
        k_index_initial = {
            layer_id: attention.k_idx_proj.weight.detach().clone()
            for layer_id, attention in msa_attentions.items()
        }

    target_step = (
        None if train_cfg.steps is None else state.global_step + train_cfg.steps
    )
    last_loss = math.nan
    model.eval() if index_only else model.train()
    fixed_input_ids: torch.Tensor | None = None
    fixed_labels: torch.Tensor | None = None
    fixed_pool_input_ids: torch.Tensor | None = None
    fixed_pool_labels: torch.Tensor | None = None

    while target_step is None or state.global_step < target_step:
        main_modes = main_modes_for_policy(
            state.global_step,
            train_cfg.warmup_steps,
            train_cfg.main_mode_policy,
            n_layers=config.model.n_layers,
            fixed_modes=train_cfg.main_modes,
        )
        state.phase = _run_phase(
            bool(index_parameters),
            main_modes,
            train_cfg.parameter_update_policy,
        )
        compute_routing_kl = bool(
            index_parameters and train_cfg.routing_kl_weight != 0.0
        )
        fixed_pool_group_index: int | None = None
        if train_cfg.fixed_train_pool_size is not None:
            if fixed_pool_input_ids is None:
                pool_size = train_cfg.fixed_train_pool_size
                if train_token_ids is None:
                    fixed_pool_input_ids, fixed_pool_labels = make_increment_batch(
                        batch_size=pool_size,
                        seq_len=train_cfg.seq_len,
                        vocab_size=config.model.vocab_size,
                        generator=data_generator,
                        device=device,
                    )
                else:
                    fixed_pool_input_ids, fixed_pool_labels = sample_next_token_batch(
                        train_token_ids,
                        batch_size=pool_size,
                        seq_len=train_cfg.seq_len,
                        generator=data_generator,
                    )
                    fixed_pool_input_ids = fixed_pool_input_ids.to(device)
                    fixed_pool_labels = fixed_pool_labels.to(device)
            assert fixed_pool_labels is not None
            group_count = train_cfg.fixed_train_pool_size // train_cfg.batch_size
            fixed_pool_group_index = state.global_step % group_count
            start = fixed_pool_group_index * train_cfg.batch_size
            stop = start + train_cfg.batch_size
            input_ids = fixed_pool_input_ids[start:stop]
            labels = fixed_pool_labels[start:stop]
        elif fixed_input_ids is not None:
            assert fixed_labels is not None
            input_ids, labels = fixed_input_ids, fixed_labels
        else:
            if train_token_ids is None:
                input_ids, labels = make_increment_batch(
                    batch_size=train_cfg.batch_size,
                    seq_len=train_cfg.seq_len,
                    vocab_size=config.model.vocab_size,
                    generator=data_generator,
                    device=device,
                )
            else:
                input_ids, labels = sample_next_token_batch(
                    train_token_ids,
                    batch_size=train_cfg.batch_size,
                    seq_len=train_cfg.seq_len,
                    generator=data_generator,
                )
                input_ids = input_ids.to(device)
                labels = labels.to(device)
            if train_cfg.fixed_train_batch:
                fixed_input_ids, fixed_labels = input_ids, labels

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        with StepTimer() as timer:
            optimizer.zero_grad(set_to_none=True)
            collect_routing_diagnostics = bool(
                compute_routing_kl
                and (state.global_step + 1) % train_cfg.log_every == 0
            )
            with _autocast_context(device, train_cfg.dtype):
                output = model(
                    input_ids,
                    labels,
                    main_modes=main_modes,
                    compute_routing_kl=compute_routing_kl,
                    compute_routing_diagnostics=collect_routing_diagnostics,
                )
                if output.loss is None:
                    raise RuntimeError("training forward returned no loss")
                lm_loss = output.loss
                routing_kl = None
                routing_kl_by_layer: dict[int, torch.Tensor] = {}
                total_loss = lm_loss
                if compute_routing_kl:
                    for layer_id in msa_attentions:
                        routing_key = f"layer_{layer_id}.routing_kl"
                        layer_routing_kl = output.aux.get(routing_key)
                        if layer_routing_kl is None:
                            raise RuntimeError("MSA training forward returned no routing KL")
                        routing_kl_by_layer[layer_id] = layer_routing_kl
                    # 固定层数下 mean 与 sum 只相差 lambda 缩放；mean 保持跨层数的量纲。
                    routing_kl = torch.stack(tuple(routing_kl_by_layer.values())).mean()
                    weighted_routing_kl = train_cfg.routing_kl_weight * routing_kl
                    total_loss = (
                        weighted_routing_kl
                        if index_only
                        else lm_loss + weighted_routing_kl
                    )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"non-finite loss at step {state.global_step}")

            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                q_index_grad_norms = {
                    layer_id: _gradient_norm(list(attention.q_idx_proj.parameters()))
                    for layer_id, attention in msa_attentions.items()
                }
                k_index_grad_norms = {
                    layer_id: _gradient_norm(list(attention.k_idx_proj.parameters()))
                    for layer_id, attention in msa_attentions.items()
                }
                main_grad_norm = (
                    torch.zeros((), device=device)
                    if index_only
                    else torch.nn.utils.clip_grad_norm_(
                        main_parameters, train_cfg.grad_clip
                    )
                )
                index_grad_norm = _clip_index_gradients(
                    index_parameters,
                    train_cfg.index_grad_clip,
                    train_cfg.grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                q_index_grad_norms = {
                    layer_id: _gradient_norm(list(attention.q_idx_proj.parameters()))
                    for layer_id, attention in msa_attentions.items()
                }
                k_index_grad_norms = {
                    layer_id: _gradient_norm(list(attention.k_idx_proj.parameters()))
                    for layer_id, attention in msa_attentions.items()
                }
                main_grad_norm = (
                    torch.zeros((), device=device)
                    if index_only
                    else torch.nn.utils.clip_grad_norm_(
                        main_parameters, train_cfg.grad_clip
                    )
                )
                index_grad_norm = _clip_index_gradients(
                    index_parameters,
                    train_cfg.index_grad_clip,
                    train_cfg.grad_clip,
                )
                optimizer.step()
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        state.global_step += 1
        tokens_this_step = train_cfg.batch_size * train_cfg.seq_len
        state.tokens_seen += tokens_this_step
        last_loss = float(lm_loss.detach().cpu())
        extras: dict[str, float] = {
            "grad_norm": float(main_grad_norm.detach().cpu()),
            "grad_clip": train_cfg.grad_clip,
            "main_grad_norm": float(main_grad_norm.detach().cpu()),
            "main_grad_clip": train_cfg.grad_clip,
            "total_loss": float(total_loss.detach().cpu()),
            "main_mode_is_sparse": float(all(mode == "sparse" for mode in main_modes)),
            "parameter_update_policy_is_index_only": float(index_only),
            "training_batch_is_fixed": float(train_cfg.fixed_train_batch),
            "main_effective_learning_rate": 0.0 if index_only else train_cfg.learning_rate,
        }
        for layer_id, mode in enumerate(main_modes):
            extras[f"layer_{layer_id}_main_mode_is_sparse"] = float(mode == "sparse")
        if train_cfg.fixed_train_pool_size is not None:
            assert fixed_pool_group_index is not None
            extras["fixed_train_pool_size"] = float(train_cfg.fixed_train_pool_size)
            extras["fixed_train_pool_group_index"] = float(fixed_pool_group_index)
        if initialized_from_step is not None:
            extras["initialized_from_step"] = float(initialized_from_step)
        if routing_kl is not None:
            extras["routing_kl"] = float(routing_kl.detach().cpu())
            for layer_id, value in routing_kl_by_layer.items():
                extras[f"layer_{layer_id}_routing_kl"] = float(value.detach().cpu())
        if index_grad_norm is not None:
            extras["index_grad_norm"] = float(index_grad_norm.detach().cpu())
            extras["index_grad_clip"] = (
                train_cfg.grad_clip
                if train_cfg.index_grad_clip is None
                else train_cfg.index_grad_clip
            )
            extras["index_learning_rate"] = index_learning_rate
            extras["index_lr_multiplier"] = train_cfg.index_lr_multiplier
        for layer_id, value in q_index_grad_norms.items():
            if value is not None:
                extras[f"layer_{layer_id}_q_index_grad_norm"] = float(value.detach().cpu())
        for layer_id, value in k_index_grad_norms.items():
            if value is not None:
                extras[f"layer_{layer_id}_k_index_grad_norm"] = float(value.detach().cpu())
        if len(msa_attentions) == 1:
            only_layer = next(iter(msa_attentions))
            if q_index_grad_norms[only_layer] is not None:
                extras["q_index_grad_norm"] = extras[f"layer_{only_layer}_q_index_grad_norm"]
            if k_index_grad_norms[only_layer] is not None:
                extras["k_index_grad_norm"] = extras[f"layer_{only_layer}_k_index_grad_norm"]
        for layer_id, attention in msa_attentions.items():
            q_index_delta = torch.linalg.vector_norm(
                attention.q_idx_proj.weight.detach().float()
                - q_index_initial[layer_id].float()
            )
            k_index_delta = torch.linalg.vector_norm(
                attention.k_idx_proj.weight.detach().float()
                - k_index_initial[layer_id].float()
            )
            extras[f"layer_{layer_id}_q_index_parameter_delta"] = float(q_index_delta.cpu())
            extras[f"layer_{layer_id}_k_index_parameter_delta"] = float(k_index_delta.cpu())
        if len(msa_attentions) == 1:
            only_layer = next(iter(msa_attentions))
            extras["q_index_parameter_delta"] = extras[f"layer_{only_layer}_q_index_parameter_delta"]
            extras["k_index_parameter_delta"] = extras[f"layer_{only_layer}_k_index_parameter_delta"]
        if collect_routing_diagnostics:
            for layer_id in msa_attentions:
                for name in ROUTING_DISTRIBUTION_METRICS:
                    value = output.aux[f"layer_{layer_id}.{name}"]
                    extras[f"layer_{layer_id}_{name}"] = float(value.detach().cpu())
            if len(msa_attentions) == 1:
                only_layer = next(iter(msa_attentions))
                for name in ROUTING_DISTRIBUTION_METRICS:
                    extras[name] = extras[f"layer_{only_layer}_{name}"]
        if train_cfg.eval_every and state.global_step % train_cfg.eval_every == 0:
            eval_main_modes = main_modes_for_policy(
                state.global_step,
                train_cfg.warmup_steps,
                train_cfg.main_mode_policy,
                n_layers=config.model.n_layers,
                fixed_modes=train_cfg.main_modes,
            )
            if fixed_pool_input_ids is not None:
                assert fixed_pool_labels is not None
                with torch.no_grad(), _autocast_context(device, train_cfg.dtype):
                    fixed_pool_output = model(
                        fixed_pool_input_ids,
                        fixed_pool_labels,
                        main_modes=eval_main_modes,
                        compute_routing_kl=True,
                    )
                fixed_pool_values = []
                for layer_id in msa_attentions:
                    value = float(
                        fixed_pool_output.aux[f"layer_{layer_id}.routing_kl"].detach().cpu()
                    )
                    extras[f"fixed_train_pool_layer_{layer_id}_routing_kl"] = value
                    fixed_pool_values.append(value)
                extras["fixed_train_pool_routing_kl"] = sum(fixed_pool_values) / len(fixed_pool_values)
            if valid_token_ids is None:
                extras.update(
                    evaluate_increment_task(
                        model,
                        config,
                        batches=train_cfg.eval_batches,
                        generator=eval_generator,
                        device=device,
                        main_mode=None,
                        main_modes=eval_main_modes,
                        compute_routing_kl=compute_routing_kl,
                        compute_routing_diagnostics=compute_routing_kl,
                    )
                )
            else:
                extras.update(
                    evaluate_token_stream(
                        model,
                        valid_token_ids,
                        batch_size=train_cfg.batch_size,
                        seq_len=train_cfg.seq_len,
                        device=device,
                        main_mode=None,
                        main_modes=eval_main_modes,
                        compute_routing_kl=compute_routing_kl,
                        compute_routing_diagnostics=compute_routing_kl,
                    )
                )

        if state.global_step % train_cfg.log_every == 0:
            metric = StepMetric(
                global_step=state.global_step,
                loss=last_loss,
                learning_rate=(
                    0.0
                    if index_only
                    else float(optimizer.param_groups[0]["lr"])
                ),
                step_time_ms=timer.elapsed_ms,
                tokens_seen=state.tokens_seen,
                tokens_per_sec=tokens_this_step / max(timer.elapsed_ms / 1000.0, 1e-12),
                run_phase=state.phase,
                resumed_from=resumed_from,
                extras=extras,
            )
            metrics.write(metric)
            if tensorboard is not None:
                tensorboard.write(metric)

        if state.global_step % train_cfg.checkpoint_every == 0:
            save_checkpoint(
                run_dir / f"checkpoint_step_{state.global_step}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                trainer_state=state,
                data_generator_state=data_generator.get_state(),
            )

        if should_stop is not None and should_stop():
            break

    final_ckpt = run_dir / "checkpoint_final.pt"
    save_checkpoint(
        final_ckpt,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        trainer_state=state,
        data_generator_state=data_generator.get_state(),
    )
    if tensorboard is not None:
        tensorboard.close()
    return TrainResult(
        final_step=state.global_step,
        final_loss=last_loss,
        tokens_seen=state.tokens_seen,
        checkpoint_path=final_ckpt,
    )
