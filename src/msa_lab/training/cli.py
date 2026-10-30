from __future__ import annotations

import argparse
import signal
from threading import Event

from msa_lab.config import ExperimentConfig
from msa_lab.training.engine import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the msa_lab dense seed trainer.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stop_event = Event()

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        if stop_event.is_set():
            raise KeyboardInterrupt
        print("\n已请求停止；将在当前完整训练 step 后保存 final checkpoint。")
        stop_event.set()

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        result = train(
            ExperimentConfig.load(args.config),
            should_stop=stop_event.is_set,
        )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
    print(
        {
            "final_step": result.final_step,
            "final_loss": result.final_loss,
            "tokens_seen": result.tokens_seen,
            "checkpoint": str(result.checkpoint_path),
        }
    )


if __name__ == "__main__":
    main()
