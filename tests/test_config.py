import pytest

from dataclasses import replace

from msa_lab.config import (
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
    assert_resume_compatible,
)


def test_default_config_validates() -> None:
    ExperimentConfig().validate()


def test_invalid_gqa_ratio_rejected() -> None:
    cfg = ModelConfig(d_model=96, n_q_heads=6, n_kv_heads=4)
    with pytest.raises(ValueError, match="n_q_heads"):
        cfg.validate()


def test_seq_len_must_fit_model() -> None:
    cfg = ExperimentConfig(
        model=ModelConfig(max_seq_len=8),
        train=TrainConfig(seq_len=9),
    )
    with pytest.raises(ValueError, match="max_seq_len"):
        cfg.validate()


def test_train_fraction_must_be_a_proper_split() -> None:
    cfg = ExperimentConfig(train=TrainConfig(train_fraction=1.0))
    with pytest.raises(ValueError, match="train_fraction"):
        cfg.validate()


def test_null_steps_allows_caller_controlled_training() -> None:
    ExperimentConfig(train=TrainConfig(steps=None)).validate()


def test_full_only_policy_requires_msa_and_no_warmup_boundary() -> None:
    ExperimentConfig(
        model=ModelConfig(msa_layer_id=0),
        train=TrainConfig(main_mode_policy="full_only", warmup_steps=0),
    ).validate()

    with pytest.raises(ValueError, match="requires an MSA layer"):
        ExperimentConfig(train=TrainConfig(main_mode_policy="full_only")).validate()

    with pytest.raises(ValueError, match="warmup_steps=0"):
        ExperimentConfig(
            model=ModelConfig(msa_layer_id=0),
            train=TrainConfig(main_mode_policy="full_only", warmup_steps=1),
        ).validate()


@pytest.mark.parametrize("layer_id", [-1, 2])
def test_msa_layer_must_identify_an_existing_transformer_block(layer_id: int) -> None:
    cfg = ExperimentConfig(model=ModelConfig(n_layers=2, msa_layer_id=layer_id))
    with pytest.raises(ValueError, match="msa_layer_id"):
        cfg.validate()


def test_dense_model_rejects_nonzero_routing_kl_weight() -> None:
    cfg = ExperimentConfig(train=TrainConfig(routing_kl_weight=0.1))
    with pytest.raises(ValueError, match="routing_kl_weight"):
        cfg.validate()


def test_multiple_msa_layers_round_trip_and_reject_dual_authority(tmp_path) -> None:
    cfg = ExperimentConfig(model=ModelConfig(n_layers=2, msa_layer_ids=(0, 1)))
    path = tmp_path / "dual_msa.json"
    cfg.save(path)
    loaded = ExperimentConfig.load(path)
    assert loaded == cfg
    assert loaded.model.resolved_msa_layer_ids() == (0, 1)

    with pytest.raises(ValueError, match="cannot both"):
        ExperimentConfig(
            model=ModelConfig(n_layers=2, msa_layer_id=0, msa_layer_ids=(0, 1))
        ).validate()


def test_fixed_per_layer_modes_round_trip_and_validate(tmp_path) -> None:
    cfg = ExperimentConfig(
        model=ModelConfig(n_layers=2, msa_layer_ids=(0, 1)),
        train=TrainConfig(
            main_mode_policy="fixed_per_layer",
            main_modes=("full", "sparse"),
        ),
    )
    path = tmp_path / "fixed_modes.json"
    cfg.save(path)
    assert ExperimentConfig.load(path) == cfg

    with pytest.raises(ValueError, match="one main_modes entry per layer"):
        replace(cfg, train=replace(cfg.train, main_modes=("full",))).validate()
    with pytest.raises(ValueError, match="requires fixed_per_layer"):
        replace(
            cfg,
            train=replace(cfg.train, main_mode_policy="scheduled"),
        ).validate()


def test_msa_allows_lambda_zero_control() -> None:
    cfg = ExperimentConfig(model=ModelConfig(msa_layer_id=0))
    cfg.validate()


def test_routing_kl_weight_mismatch_rejects_exact_resume() -> None:
    saved = ExperimentConfig(model=ModelConfig(msa_layer_id=0))
    current = replace(
        saved,
        train=replace(saved.train, routing_kl_weight=0.1),
    )
    with pytest.raises(ValueError, match="routing_kl_weight"):
        assert_resume_compatible(current, saved.to_dict())


def test_main_mode_policy_mismatch_rejects_exact_resume() -> None:
    saved = ExperimentConfig(model=ModelConfig(msa_layer_id=0))
    current = replace(
        saved,
        train=replace(saved.train, main_mode_policy="full_only"),
    )
    with pytest.raises(ValueError, match="main_mode_policy"):
        assert_resume_compatible(current, saved.to_dict())


@pytest.mark.parametrize(
    "field",
    ["index_lr_multiplier", "index_grad_clip"],
)
def test_index_optimizer_fields_must_be_positive(field: str) -> None:
    cfg = ExperimentConfig(
        model=ModelConfig(msa_layer_id=0),
        train=TrainConfig(**{field: 0.0}),
    )
    with pytest.raises(ValueError, match=field):
        cfg.validate()


def test_legacy_absolute_index_lr_migrates_to_multiplier() -> None:
    raw = ExperimentConfig(
        model=ModelConfig(msa_layer_id=0),
        train=TrainConfig(learning_rate=2e-3),
    ).to_dict()
    raw["train"].pop("index_lr_multiplier")
    raw["train"]["index_learning_rate"] = 7e-3

    migrated = ExperimentConfig.from_dict(raw)

    assert migrated.train.index_lr_multiplier == pytest.approx(3.5)
    assert "index_learning_rate" not in migrated.to_dict()["train"]


def test_index_only_requires_fixed_teacher_initialization() -> None:
    model = ModelConfig(msa_layer_id=0)
    with pytest.raises(ValueError, match="full_only"):
        ExperimentConfig(
            model=model,
            train=TrainConfig(
                parameter_update_policy="index_only",
                initialize_from="source.pt",
                routing_kl_weight=1.0,
            ),
        ).validate()

    ExperimentConfig(
        model=model,
        train=TrainConfig(
            main_mode_policy="full_only",
            parameter_update_policy="index_only",
            initialize_from="source.pt",
            routing_kl_weight=1.0,
        ),
    ).validate()


def test_fixed_train_batch_is_restricted_to_index_only_diagnostic() -> None:
    with pytest.raises(ValueError, match="fixed_train_batch"):
        ExperimentConfig(train=TrainConfig(fixed_train_batch=True)).validate()

    ExperimentConfig(
        model=ModelConfig(msa_layer_id=0),
        train=TrainConfig(
            main_mode_policy="full_only",
            parameter_update_policy="index_only",
            fixed_train_batch=True,
            initialize_from="source.pt",
            routing_kl_weight=1.0,
        ),
    ).validate()

    with pytest.raises(ValueError, match="fixed_train_pool_size"):
        ExperimentConfig(
            model=ModelConfig(msa_layer_id=0),
            train=TrainConfig(
                main_mode_policy="full_only",
                parameter_update_policy="index_only",
                fixed_train_batch=True,
                fixed_train_pool_size=7,
                batch_size=2,
                initialize_from="source.pt",
                routing_kl_weight=1.0,
            ),
        ).validate()
