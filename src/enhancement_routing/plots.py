from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir


def plot_policy_comparison(policy_comparison: pd.DataFrame, figures_dir: str | Path) -> Path:
    figures_dir = ensure_dir(figures_dir)
    data = policy_comparison.sort_values("mean_score", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(data["policy"], data["mean_score"])
    ax.set_xlabel("Mean per-image recall@0.50")
    ax.set_ylabel("Policy")
    ax.set_title("Policy comparison")
    fig.tight_layout()
    out = figures_dir / "policy_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_oracle_gap(policy_comparison: pd.DataFrame, figures_dir: str | Path) -> Path:
    figures_dir = ensure_dir(figures_dir)
    data = policy_comparison[policy_comparison["policy"] != "oracle"].sort_values("oracle_gap", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(data["policy"], data["oracle_gap"])
    ax.set_xlabel("Oracle gap")
    ax.set_ylabel("Policy")
    ax.set_title("Distance from oracle preprocessing action")
    fig.tight_layout()
    out = figures_dir / "oracle_gap.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_harmful_enhancement(policy_comparison: pd.DataFrame, figures_dir: str | Path) -> Path:
    figures_dir = ensure_dir(figures_dir)
    data = policy_comparison.sort_values("harmful_enhancement_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(data["policy"], data["harmful_enhancement_rate"])
    ax.set_xlabel("Harmful enhancement rate")
    ax.set_ylabel("Policy")
    ax.set_title("How often preprocessing reduces detection score")
    fig.tight_layout()
    out = figures_dir / "harmful_enhancement_cases.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_routing_pipeline(figures_dir: str | Path) -> Path:
    figures_dir = ensure_dir(figures_dir)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    boxes = [
        "Input image",
        "No-reference\nfeatures",
        "Routing policy",
        "Preprocessing\naction",
        "YOLO detector",
        "Detection\nmetrics",
    ]
    x_positions = [0.05, 0.22, 0.39, 0.56, 0.73, 0.90]
    for x, label in zip(x_positions, boxes, strict=False):
        ax.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "black"},
            transform=ax.transAxes,
        )
    for x1, x2 in zip(x_positions[:-1], x_positions[1:], strict=False):
        ax.annotate(
            "",
            xy=(x2 - 0.07, 0.5),
            xytext=(x1 + 0.07, 0.5),
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "lw": 1.5},
        )
    fig.tight_layout()
    out = figures_dir / "routing_pipeline.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def make_all_figures(tables_dir: str | Path, figures_dir: str | Path) -> list[Path]:
    tables_dir = Path(tables_dir)
    policy_path = tables_dir / "policy_comparison.csv"
    if not policy_path.exists():
        raise FileNotFoundError(f"Missing {policy_path}. Run scripts/evaluate.py first.")
    policy_comparison = pd.read_csv(policy_path)
    return [
        plot_policy_comparison(policy_comparison, figures_dir),
        plot_oracle_gap(policy_comparison, figures_dir),
        plot_harmful_enhancement(policy_comparison, figures_dir),
        plot_routing_pipeline(figures_dir),
    ]
