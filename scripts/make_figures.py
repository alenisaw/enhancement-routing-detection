from __future__ import annotations

import argparse

from enhancement_routing.plots import make_all_figures
from enhancement_routing.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper figures from evaluation tables.")
    parser.add_argument("--config", default="configs/experiment.yaml", help="Path to experiment YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    figures = make_all_figures(config["paths"]["tables_dir"], config["paths"]["figures_dir"])
    print("Saved figures:")
    for fig in figures:
        print(f"- {fig}")


if __name__ == "__main__":
    main()
