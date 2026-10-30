from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from msa_lab.config import ExperimentConfig, ModelConfig, TrainConfig
from msa_lab.data.tiny_shakespeare import iter_validation_batches
from msa_lab.model.initialization import build_model_from_dense_anchor
from msa_lab.model.msa_attention import CausalMSAAttention
from msa_lab.training.engine import train
from msa_lab.training.eval import evaluate_token_stream
from msa_lab.training.parameter_groups import split_main_index_parameters


def _base_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        n_layers=2,
        n_q_heads=4,
        n_kv_heads=2,
        d_ff=32,
        dropout=0.0,
    )


def test_dense_anchor_is_shared_and_index_initialization_matches_across_layers() -> None:
    dense_cfg = _base_model_config()
    msa_l0_cfg = replace(
        dense_cfg,
        msa_layer_id=0,
        msa_index_dim=3,
        msa_block_size=2,
        msa_top_k=2,
    )
    msa_l1_cfg = replace(msa_l0_cfg, msa_layer_id=1)

    dense = build_model_from_dense_anchor(dense_cfg, seed=17)
    msa_l0 = build_model_from_dense_anchor(msa_l0_cfg, seed=17)
    msa_l1 = build_model_from_dense_anchor(msa_l1_cfg, seed=17)

    assert isinstance(msa_l0.blocks[0].attn, CausalMSAAttention)
    assert isinstance(msa_l1.blocks[1].attn, CausalMSAAttention)
    for name, dense_value in dense.state_dict().items():
        torch.testing.assert_close(msa_l0.state_dict()[name], dense_value, rtol=0, atol=0)
        torch.testing.assert_close(msa_l1.state_dict()[name], dense_value, rtol=0, atol=0)

    torch.testing.assert_close(
        msa_l0.blocks[0].attn.q_idx_proj.weight,
        msa_l1.blocks[1].attn.q_idx_proj.weight,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        msa_l0.blocks[0].attn.k_idx_proj.weight,
        msa_l1.blocks[1].attn.k_idx_proj.weight,
        rtol=0,
        atol=0,
    )

    input_ids = torch.randint(0, dense_cfg.vocab_size, (2, dense_cfg.max_seq_len))
    dense_logits = dense(input_ids).logits
    msa_l0_logits = msa_l0(input_ids, main_mode="full").logits
    msa_l1_logits = msa_l1(input_ids, main_mode="full").logits
    torch.testing.assert_close(msa_l0_logits, dense_logits)
    torch.testing.assert_close(msa_l1_logits, dense_logits)


def test_two_msa_layers_share_dense_anchor_and_accept_per_layer_modes() -> None:
    dense_cfg = _base_model_config()
    dual_cfg = replace(
        dense_cfg,
        msa_layer_ids=(0, 1),
        msa_index_dim=3,
        msa_block_size=2,
        msa_top_k=2,
    )
    dense = build_model_from_dense_anchor(dense_cfg, seed=29)
    dual = build_model_from_dense_anchor(dual_cfg, seed=29)

    assert isinstance(dual.blocks[0].attn, CausalMSAAttention)
    assert isinstance(dual.blocks[1].attn, CausalMSAAttention)
    for name, dense_value in dense.state_dict().items():
        torch.testing.assert_close(dual.state_dict()[name], dense_value, rtol=0, atol=0)

    input_ids = torch.randint(0, dense_cfg.vocab_size, (2, dense_cfg.max_seq_len))
    dense_logits = dense(input_ids).logits
    dual_ff = dual(input_ids, main_modes=("full", "full")).logits
    torch.testing.assert_close(dual_ff, dense_logits)

    for modes in (("full", "sparse"), ("sparse", "full"), ("sparse", "sparse")):
        output = dual(input_ids, main_modes=modes, compute_routing_kl=True)
        assert output.logits.shape == dense_logits.shape
        assert set(output.aux) == {"layer_0.routing_kl", "layer_1.routing_kl"}
        assert torch.isfinite(output.aux["layer_0.routing_kl"])
        assert torch.isfinite(output.aux["layer_1.routing_kl"])

    with pytest.raises(ValueError, match="不能同时"):
        dual(input_ids, main_mode="full", main_modes=("full", "full"))
    with pytest.raises(ValueError, match="每个 Transformer layer"):
        dual(input_ids, main_modes=("full",))


def test_parameter_groups_are_disjoint_exhaustive_and_index_only() -> None:
    cfg = replace(
        _base_model_config(),
        msa_layer_id=0,
        msa_index_dim=3,
        msa_block_size=2,
        msa_top_k=2,
    )
    model = build_model_from_dense_anchor(cfg, seed=19)
    main_parameters, index_parameters = split_main_index_parameters(model)

    main_ids = {id(parameter) for parameter in main_parameters}
    index_ids = {id(parameter) for parameter in index_parameters}
    all_ids = {id(parameter) for parameter in model.parameters()}
    expected_index_ids = {
        id(model.blocks[0].attn.q_idx_proj.weight),
        id(model.blocks[0].attn.k_idx_proj.weight),
    }

    assert index_ids == expected_index_ids
    assert main_ids.isdisjoint(index_ids)
    assert main_ids | index_ids == all_ids


def test_msa_training_uses_two_adamw_groups_and_logs_separate_grad_norms(
    tmp_path: Path,
) -> None:
    model_cfg = replace(
        _base_model_config(),
        n_layers=1,
        msa_layer_id=0,
        msa_index_dim=3,
        msa_block_size=2,
        msa_top_k=2,
    )
    config = ExperimentConfig(
        model=model_cfg,
        train=TrainConfig(
            seed=23,
            device="cpu",
            dtype="float32",
            batch_size=2,
            seq_len=8,
            steps=1,
            main_mode_policy="full_only",
            warmup_steps=0,
            learning_rate=2e-3,
            index_lr_multiplier=3.5,
            routing_kl_weight=1.0,
            grad_clip=1.0,
            index_grad_clip=0.25,
            log_every=1,
            eval_every=1,
            eval_batches=1,
            tensorboard=True,
            checkpoint_every=1,
            run_dir=str(tmp_path),
        ),
    )

    result = train(config)
    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    optimizer_groups = checkpoint["optimizer"]["param_groups"]
    assert [group["name"] for group in optimizer_groups] == ["main", "index"]
    assert [group["lr"] for group in optimizer_groups] == [2e-3, 7e-3]

    metric = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[-1])
    assert metric["loss"] > 0
    assert metric["extras"]["routing_kl"] >= 0
    assert metric["extras"]["total_loss"] >= metric["loss"]
    assert metric["extras"]["main_grad_clip"] == 1.0
    assert metric["extras"]["index_grad_clip"] == 0.25
    assert metric["extras"]["main_grad_norm"] >= 0
    assert metric["extras"]["index_grad_norm"] >= 0
    assert metric["extras"]["index_lr_multiplier"] == 3.5
    assert metric["run_phase"] == "msa_full"
    assert metric["extras"]["main_mode_is_sparse"] == 0.0
    assert metric["extras"]["q_index_grad_norm"] > 0
    assert metric["extras"]["k_index_grad_norm"] > 0
    assert metric["extras"]["q_index_parameter_delta"] > 0
    assert metric["extras"]["k_index_parameter_delta"] > 0
    assert metric["extras"]["eval_routing_kl"] >= 0
    assert metric["extras"]["eval_routing_rows"] > 0
    assert metric["extras"]["eval_main_teacher_entropy"] >= 0
    assert metric["extras"]["eval_index_student_entropy"] >= 0

    events = EventAccumulator(str(tmp_path / "tensorboard"))
    events.Reload()
    scalar_tags = set(events.Tags()["scalars"])
    assert {
        "01_lm/train_batch_loss",
        "01_lm/validation_loss",
        "02_router_tracking/train_batch_kl_main_to_index",
        "02_router_tracking/validation_kl_main_to_index",
        "03_router_distribution/validation_main_teacher_entropy",
        "03_router_distribution/validation_index_student_entropy",
        "04_index_update/q_grad_norm_before_clip",
        "04_index_update/k_grad_norm_before_clip",
        "04_index_update/q_distance_from_start",
        "04_index_update/k_distance_from_start",
        "04_index_update/lr_multiplier",
        "05_main_update/grad_norm_before_clip",
        "06_execution/main_is_sparse",
        "07_runtime/tokens_per_second",
    } <= scalar_tags


def test_index_only_initialization_freezes_main_and_resets_training_state(
    tmp_path: Path,
) -> None:
    model_cfg = replace(
        _base_model_config(),
        n_layers=1,
        msa_layer_id=0,
        msa_index_dim=3,
        msa_block_size=2,
        msa_top_k=2,
    )
    source_dir = tmp_path / "source"
    source_config = ExperimentConfig(
        model=model_cfg,
        train=TrainConfig(
            seed=31,
            device="cpu",
            dtype="float32",
            batch_size=2,
            seq_len=8,
            steps=1,
            main_mode_policy="full_only",
            learning_rate=2e-3,
            routing_kl_weight=1.0,
            log_every=1,
            eval_every=0,
            checkpoint_every=1,
            run_dir=str(source_dir),
        ),
    )
    source_result = train(source_config)
    source = torch.load(source_result.checkpoint_path, map_location="cpu", weights_only=False)

    probe_dir = tmp_path / "probe"
    probe_config = ExperimentConfig(
        model=model_cfg,
        train=replace(
            source_config.train,
            steps=2,
            parameter_update_policy="index_only",
            fixed_train_batch=True,
            index_lr_multiplier=3.0,
            initialize_from=str(source_result.checkpoint_path),
            run_dir=str(probe_dir),
        ),
    )
    probe_result = train(probe_config)
    probe = torch.load(probe_result.checkpoint_path, map_location="cpu", weights_only=False)

    index_names = {
        "blocks.0.attn.q_idx_proj.weight",
        "blocks.0.attn.k_idx_proj.weight",
    }
    for name, source_value in source["model"].items():
        if name not in index_names:
            torch.testing.assert_close(probe["model"][name], source_value, rtol=0, atol=0)
    assert any(
        not torch.equal(probe["model"][name], source["model"][name])
        for name in index_names
    )
    assert probe["trainer_state"]["global_step"] == 2
    assert [group["name"] for group in probe["optimizer"]["param_groups"]] == ["index"]
    assert probe["optimizer"]["param_groups"][0]["lr"] == pytest.approx(6e-3)

    metric = json.loads((probe_dir / "metrics.jsonl").read_text().splitlines()[-1])
    assert metric["run_phase"] == "msa_full_index_only"
    assert metric["extras"]["initialized_from_step"] == 1.0
    assert metric["extras"]["main_grad_norm"] == 0.0
    assert metric["extras"]["index_grad_norm"] > 0
    assert metric["extras"]["training_batch_is_fixed"] == 1.0


def test_fixed_pool_microbatches_log_full_pool_routing_kl(tmp_path: Path) -> None:
    model_cfg = replace(
        _base_model_config(), max_seq_len=8, msa_layer_id=0, msa_index_dim=4
    )
    source_cfg = ExperimentConfig(
        model=model_cfg,
        train=TrainConfig(
            seed=71, device="cpu", dtype="float32", batch_size=4, seq_len=8,
            steps=1, main_mode_policy="full_only", routing_kl_weight=1.0,
            checkpoint_every=1, run_dir=str(tmp_path / "source"),
        ),
    )
    source = train(source_cfg)
    run_dir = tmp_path / "pool"
    cfg = ExperimentConfig(
        model=model_cfg,
        train=replace(
            source_cfg.train,
            batch_size=2,
            steps=4,
            parameter_update_policy="index_only",
            fixed_train_batch=True,
            fixed_train_pool_size=4,
            initialize_from=str(source.checkpoint_path),
            eval_every=1,
            log_every=1,
            run_dir=str(run_dir),
        ),
    )
    train(cfg)
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [row["extras"]["fixed_train_pool_group_index"] for row in rows] == [0.0, 1.0, 0.0, 1.0]
    assert all(row["extras"]["fixed_train_pool_size"] == 4.0 for row in rows)
    assert all(row["extras"]["fixed_train_pool_routing_kl"] >= 0.0 for row in rows)


def test_held_out_routing_kl_is_weighted_by_actual_validation_rows() -> None:
    cfg = replace(
        _base_model_config(),
        max_seq_len=4,
        msa_layer_id=1,
        msa_index_dim=3,
        msa_block_size=2,
        msa_top_k=2,
    )
    model = build_model_from_dense_anchor(cfg, seed=29)
    token_ids = torch.arange(21, dtype=torch.long) % cfg.vocab_size

    weighted_kl = 0.0
    routing_rows = 0
    with torch.no_grad():
        for input_ids, labels in iter_validation_batches(
            token_ids,
            batch_size=3,
            seq_len=4,
        ):
            output = model(
                input_ids,
                labels,
                main_mode="full",
                compute_routing_kl=True,
            )
            row_count = input_ids.shape[0] * input_ids.shape[1] * cfg.n_kv_heads
            weighted_kl += float(output.aux["layer_1.routing_kl"]) * row_count
            routing_rows += row_count

    actual = evaluate_token_stream(
        model,
        token_ids,
        batch_size=3,
        seq_len=4,
        device=torch.device("cpu"),
        main_mode="full",
        compute_routing_kl=True,
    )

    assert actual["eval_routing_rows"] == float(routing_rows)
    assert actual["eval_routing_kl"] == pytest.approx(weighted_kl / routing_rows)


def test_dual_msa_fixed_segment_trains_and_logs_each_layer(tmp_path: Path) -> None:
    model_cfg = replace(
        _base_model_config(),
        msa_layer_ids=(0, 1),
        msa_index_dim=3,
        msa_block_size=2,
        msa_top_k=2,
    )
    config = ExperimentConfig(
        model=model_cfg,
        train=TrainConfig(
            seed=31,
            batch_size=2,
            seq_len=8,
            steps=1,
            main_mode_policy="fixed_per_layer",
            main_modes=("full", "sparse"),
            learning_rate=2e-3,
            routing_kl_weight=1.0,
            log_every=1,
            eval_every=1,
            eval_batches=1,
            checkpoint_every=1,
            run_dir=str(tmp_path),
        ),
    )

    result = train(config)
    metric = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[-1])
    extras = metric["extras"]

    assert metric["run_phase"] == "msa_fs"
    assert extras["main_mode_is_sparse"] == 0.0
    assert extras["layer_0_main_mode_is_sparse"] == 0.0
    assert extras["layer_1_main_mode_is_sparse"] == 1.0
    assert extras["routing_kl"] == pytest.approx(
        (extras["layer_0_routing_kl"] + extras["layer_1_routing_kl"]) / 2
    )
    assert extras["eval_routing_kl"] == pytest.approx(
        (extras["eval_layer_0_routing_kl"] + extras["eval_layer_1_routing_kl"]) / 2
    )
    for layer_id in (0, 1):
        assert extras[f"layer_{layer_id}_q_index_grad_norm"] > 0
        assert extras[f"layer_{layer_id}_k_index_grad_norm"] > 0
        assert extras[f"layer_{layer_id}_q_index_parameter_delta"] > 0
        assert extras[f"layer_{layer_id}_k_index_parameter_delta"] > 0

    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert [group["name"] for group in checkpoint["optimizer"]["param_groups"]] == [
        "main",
        "index",
    ]
