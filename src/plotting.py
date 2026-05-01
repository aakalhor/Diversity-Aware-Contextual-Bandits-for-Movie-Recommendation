from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _prepare_output_path(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_cumulative_metric(recommendations: pd.DataFrame, value_column: str, title: str, output_path: str | Path) -> None:
    output_path = _prepare_output_path(output_path)
    plt.figure(figsize=(10, 6))

    for model_name, frame in recommendations.groupby("model"):
        plot_frame = frame.sort_values("global_step").copy()
        plot_frame[f"cumulative_{value_column}"] = plot_frame[value_column].cumsum()
        plt.plot(plot_frame["global_step"], plot_frame[f"cumulative_{value_column}"], label=model_name, linewidth=2)

    plt.title(title)
    plt.xlabel("Interaction Step")
    plt.ylabel(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_reward_vs_diversity(summary: pd.DataFrame, output_path: str | Path) -> None:
    output_path = _prepare_output_path(output_path)
    plt.figure(figsize=(8, 6))
    plt.scatter(summary["intra_list_diversity"], summary["avg_rating_reward"], s=120)

    for row in summary.itertuples(index=False):
        plt.annotate(row.model, (row.intra_list_diversity, row.avg_rating_reward), textcoords="offset points", xytext=(6, 6))

    plt.title("Reward vs Diversity")
    plt.xlabel("Intra-list Diversity")
    plt.ylabel("Average Rating Reward")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_metrics_by_model(summary: pd.DataFrame, output_path: str | Path) -> None:
    output_path = _prepare_output_path(output_path)
    metrics = [
        "hit_rate",
        "avg_rating_reward",
        "cumulative_reward_final",
        "intra_list_diversity",
        "catalog_coverage",
        "genre_coverage",
        "avg_recommended_popularity",
        "cumulative_regret_final",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()

    for axis, metric in zip(axes, metrics):
        axis.bar(summary["model"], summary[metric])
        axis.set_title(metric.replace("_", " ").title())
        axis.tick_params(axis="x", rotation=45)
        axis.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_lambda_sensitivity(summary: pd.DataFrame, output_path: str | Path) -> None:
    output_path = _prepare_output_path(output_path)
    lambda_rows = summary[summary["lambda"].notna()].copy()
    if lambda_rows.empty:
        return

    lambda_rows = lambda_rows.sort_values("lambda")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    metric_pairs = [
        ("hit_rate", "Hit Rate"),
        ("intra_list_diversity", "Intra-list Diversity"),
        ("catalog_coverage", "Catalog Coverage"),
        ("avg_rating_reward", "Average Rating Reward"),
    ]

    for axis, (metric, title) in zip(axes.flatten(), metric_pairs):
        axis.plot(lambda_rows["lambda"], lambda_rows[metric], marker="o", linewidth=2)
        axis.set_title(title)
        axis.set_xlabel("Lambda")
        axis.set_ylabel(title)
        axis.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
