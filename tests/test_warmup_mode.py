from dataclasses import replace

import pytest

from msa_lab.config import ExperimentConfig, assert_resume_compatible
from msa_lab.training.mode import main_mode_for_policy, main_mode_for_step


@pytest.mark.parametrize(
    ("global_step", "warmup_steps", "expected"),
    [
        (0, 0, "sparse"),
        (0, 3, "full"),
        (2, 3, "full"),
        (3, 3, "sparse"),
        (4, 3, "sparse"),
    ],
)
def test_main_mode_uses_completed_step_boundary(
    global_step: int, warmup_steps: int, expected: str
) -> None:
    assert main_mode_for_step(global_step, warmup_steps) == expected


@pytest.mark.parametrize("global_step", [0, 3, 320_716, 1_280_000])
def test_full_only_policy_never_switches_to_sparse(global_step: int) -> None:
    assert main_mode_for_policy(global_step, warmup_steps=0, policy="full_only") == "full"


def test_scheduled_policy_preserves_c05_boundary() -> None:
    assert main_mode_for_policy(2, warmup_steps=3, policy="scheduled") == "full"
    assert main_mode_for_policy(3, warmup_steps=3, policy="scheduled") == "sparse"


def test_negative_warmup_steps_is_invalid() -> None:
    base = ExperimentConfig()
    config = replace(base, train=replace(base.train, warmup_steps=-1))

    with pytest.raises(ValueError, match="warmup_steps"):
        config.validate()


def test_warmup_steps_mismatch_rejects_exact_resume() -> None:
    saved = ExperimentConfig()
    current = replace(saved, train=replace(saved.train, warmup_steps=3))

    with pytest.raises(ValueError, match="warmup_steps"):
        assert_resume_compatible(current, saved.to_dict())


def test_operational_step_count_can_change_on_resume() -> None:
    saved = ExperimentConfig()
    current = replace(saved, train=replace(saved.train, steps=saved.train.steps + 1))

    assert_resume_compatible(current, saved.to_dict())
