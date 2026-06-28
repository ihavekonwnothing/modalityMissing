from __future__ import annotations

import argparse
import json

from models.factory import build_model
from train import resolve_model_name
from utils.config import load_config


def count_parameters(model) -> dict[str, float | int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "total_params": total,
        "trainable_params": trainable,
        "total_params_m": total / 1_000_000,
        "trainable_params_m": trainable / 1_000_000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = resolve_model_name(config, args.model)
    model = build_model(model_name, config)
    stats = {"model": model_name, "config": args.config, **count_parameters(model)}
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
