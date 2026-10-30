from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import time
from typing import Any


@dataclass
class StepMetric:
    global_step: int
    loss: float
    learning_rate: float
    step_time_ms: float
    tokens_seen: int
    tokens_per_sec: float
    run_phase: str
    resumed_from: str | None
    extras: dict[str, Any] = field(default_factory=dict)


class JsonlMetricWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, metric: StepMetric) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(metric), ensure_ascii=False) + "\n")


class TensorBoardMetricWriter:
    def __init__(self, log_dir: str | Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=str(log_dir), max_queue=1, flush_secs=1)
        self.writer.add_custom_scalars(
            {
                "01 LM quality": {
                    "Train batch vs validation": [
                        "Multiline",
                        ["01_lm/train_batch_loss", "01_lm/validation_loss"],
                    ],
                },
                "02 Router tracking": {
                    "Main teacher to Index student KL": [
                        "Multiline",
                        [
                            "02_router_tracking/train_batch_kl_main_to_index",
                            "02_router_tracking/validation_kl_main_to_index",
                        ],
                    ],
                },
                "03 Router distribution": {
                    "Teacher vs student entropy": [
                        "Multiline",
                        [
                            "03_router_distribution/validation_main_teacher_entropy",
                            "03_router_distribution/validation_index_student_entropy",
                        ],
                    ],
                    "Teacher vs student top-1 mass": [
                        "Multiline",
                        [
                            "03_router_distribution/validation_main_teacher_top1_mass",
                            "03_router_distribution/validation_index_student_top1_mass",
                        ],
                    ],
                },
                "04 Index update": {
                    "Q/K gradient before clipping": [
                        "Multiline",
                        [
                            "04_index_update/q_grad_norm_before_clip",
                            "04_index_update/k_grad_norm_before_clip",
                        ],
                    ],
                    "Q/K distance from experiment start": [
                        "Multiline",
                        [
                            "04_index_update/q_distance_from_start",
                            "04_index_update/k_distance_from_start",
                        ],
                    ],
                },
                "05 Main update": {
                    "Gradient before clipping vs threshold": [
                        "Multiline",
                        [
                            "05_main_update/grad_norm_before_clip",
                            "05_main_update/clip_threshold",
                        ],
                    ],
                },
            }
        )

    def write(self, metric: StepMetric) -> None:
        step = metric.global_step
        self.writer.add_scalar("01_lm/train_batch_loss", metric.loss, step)
        if "eval_loss" in metric.extras:
            self.writer.add_scalar("01_lm/validation_loss", metric.extras["eval_loss"], step)
        if "total_loss" in metric.extras:
            self.writer.add_scalar(
                "99_debug/combined_training_objective",
                metric.extras["total_loss"],
                step,
            )
        if "routing_kl" in metric.extras:
            self.writer.add_scalar(
                "02_router_tracking/train_batch_kl_main_to_index",
                metric.extras["routing_kl"],
                step,
            )
        if "eval_routing_kl" in metric.extras:
            self.writer.add_scalar(
                "02_router_tracking/validation_kl_main_to_index",
                metric.extras["eval_routing_kl"],
                step,
            )
        if "fixed_train_pool_routing_kl" in metric.extras:
            self.writer.add_scalar(
                "02_router_tracking/full_fixed_pool_kl_main_to_index",
                metric.extras["fixed_train_pool_routing_kl"],
                step,
            )
        distribution_metrics = {
            "main_teacher_entropy": "train_main_teacher_entropy",
            "index_student_entropy": "train_index_student_entropy",
            "main_teacher_top1_mass": "train_main_teacher_top1_mass",
            "index_student_top1_mass": "train_index_student_top1_mass",
            "eval_main_teacher_entropy": "validation_main_teacher_entropy",
            "eval_index_student_entropy": "validation_index_student_entropy",
            "eval_main_teacher_top1_mass": "validation_main_teacher_top1_mass",
            "eval_index_student_top1_mass": "validation_index_student_top1_mass",
        }
        for source_name, tag_name in distribution_metrics.items():
            if source_name in metric.extras:
                self.writer.add_scalar(
                    f"03_router_distribution/{tag_name}",
                    metric.extras[source_name],
                    step,
                )
        if "main_grad_norm" in metric.extras:
            self.writer.add_scalar(
                "05_main_update/grad_norm_before_clip",
                metric.extras["main_grad_norm"],
                step,
            )
        if "main_grad_clip" in metric.extras:
            self.writer.add_scalar(
                "05_main_update/clip_threshold",
                metric.extras["main_grad_clip"],
                step,
            )
        self.writer.add_scalar(
            "05_main_update/effective_lr",
            metric.extras.get("main_effective_learning_rate", metric.learning_rate),
            step,
        )
        if "index_grad_norm" in metric.extras:
            self.writer.add_scalar(
                "04_index_update/total_grad_norm_before_clip",
                metric.extras["index_grad_norm"],
                step,
            )
        if "index_grad_clip" in metric.extras:
            self.writer.add_scalar(
                "04_index_update/clip_threshold",
                metric.extras["index_grad_clip"],
                step,
            )
        if "index_learning_rate" in metric.extras:
            self.writer.add_scalar(
                "04_index_update/effective_lr",
                metric.extras["index_learning_rate"],
                step,
            )
        if "index_lr_multiplier" in metric.extras:
            self.writer.add_scalar(
                "04_index_update/lr_multiplier",
                metric.extras["index_lr_multiplier"],
                step,
            )
        if "q_index_grad_norm" in metric.extras:
            self.writer.add_scalar(
                "04_index_update/q_grad_norm_before_clip",
                metric.extras["q_index_grad_norm"],
                step,
            )
        if "k_index_grad_norm" in metric.extras:
            self.writer.add_scalar(
                "04_index_update/k_grad_norm_before_clip",
                metric.extras["k_index_grad_norm"],
                step,
            )
        if "q_index_parameter_delta" in metric.extras:
            self.writer.add_scalar(
                "04_index_update/q_distance_from_start",
                metric.extras["q_index_parameter_delta"],
                step,
            )
        if "k_index_parameter_delta" in metric.extras:
            self.writer.add_scalar(
                "04_index_update/k_distance_from_start",
                metric.extras["k_index_parameter_delta"],
                step,
            )
        if "main_mode_is_sparse" in metric.extras:
            self.writer.add_scalar(
                "06_execution/main_is_sparse",
                metric.extras["main_mode_is_sparse"],
                step,
            )
        for name, value in metric.extras.items():
            if name.startswith("layer_") and name.endswith("_main_mode_is_sparse"):
                layer = name.removeprefix("layer_").removesuffix("_main_mode_is_sparse")
                self.writer.add_scalar(
                    f"06_execution/layer_{layer}_is_sparse",
                    value,
                    step,
                )
            elif name.startswith("layer_") and name.endswith("_routing_kl"):
                layer = name.removeprefix("layer_").removesuffix("_routing_kl")
                self.writer.add_scalar(
                    f"02_router_tracking/layer_{layer}_train_kl_main_to_index",
                    value,
                    step,
                )
            elif name.startswith("eval_layer_") and name.endswith("_routing_kl"):
                layer = name.removeprefix("eval_layer_").removesuffix("_routing_kl")
                self.writer.add_scalar(
                    f"02_router_tracking/layer_{layer}_validation_kl_main_to_index",
                    value,
                    step,
                )
            elif name.startswith("layer_") and name.endswith("_q_index_grad_norm"):
                layer = name.removeprefix("layer_").removesuffix("_q_index_grad_norm")
                self.writer.add_scalar(f"04_index_update/layer_{layer}_q_grad_norm", value, step)
            elif name.startswith("layer_") and name.endswith("_k_index_grad_norm"):
                layer = name.removeprefix("layer_").removesuffix("_k_index_grad_norm")
                self.writer.add_scalar(f"04_index_update/layer_{layer}_k_grad_norm", value, step)
            elif name.startswith("layer_") and name.endswith("_q_index_parameter_delta"):
                layer = name.removeprefix("layer_").removesuffix("_q_index_parameter_delta")
                self.writer.add_scalar(f"04_index_update/layer_{layer}_q_distance", value, step)
            elif name.startswith("layer_") and name.endswith("_k_index_parameter_delta"):
                layer = name.removeprefix("layer_").removesuffix("_k_index_parameter_delta")
                self.writer.add_scalar(f"04_index_update/layer_{layer}_k_distance", value, step)
        if "parameter_update_policy_is_index_only" in metric.extras:
            self.writer.add_scalar(
                "06_execution/update_policy_is_index_only",
                metric.extras["parameter_update_policy_is_index_only"],
                step,
            )
        if "training_batch_is_fixed" in metric.extras:
            self.writer.add_scalar(
                "06_execution/training_batch_is_fixed",
                metric.extras["training_batch_is_fixed"],
                step,
            )
        if "initialized_from_step" in metric.extras:
            self.writer.add_scalar(
                "06_execution/initialized_from_step",
                metric.extras["initialized_from_step"],
                step,
            )
        self.writer.add_scalar("07_runtime/step_time_ms", metric.step_time_ms, step)
        self.writer.add_scalar("07_runtime/tokens_per_second", metric.tokens_per_sec, step)
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


class StepTimer:
    def __enter__(self) -> "StepTimer":
        self.start = time.perf_counter()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0
