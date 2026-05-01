from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_movielens_data, validate_movielens_files
from src.features import FeatureBuilder
from src.metrics import compute_metrics_summary, write_experiment_summary
from src.models import DiversityAwareLinUCB, EpsilonGreedy, LinUCB
from src.plotting import (
    plot_cumulative_metric,
    plot_lambda_sensitivity,
    plot_metrics_by_model,
    plot_reward_vs_diversity,
)
from src.simulation import run_offline_simulation


def parse_lambda_values(raw_value: str) -> list[float]:
    tokens = [token.strip() for token in raw_value.split(",") if token.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("lambda_values must contain at least one comma-separated float.")

    try:
        return [float(token) for token in tokens]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --lambda_values input '{raw_value}'. Expected comma-separated floats like 0.0,0.1,0.3,0.5."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diversity-aware contextual bandits for MovieLens recommendation.")
    parser.add_argument("--data_dir", "--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output_dir", "--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--min_ratings_per_user", type=int, default=20)
    parser.add_argument("--max_recommendations_per_user", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lambda_values", type=parse_lambda_values, default=parse_lambda_values("0.0,0.1,0.3,0.5"))
    parser.add_argument("--recent_window", "--recent-window", type=int, default=5)
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_models(
    feature_dim: int,
    alpha: float,
    epsilon: float,
    recent_window: int,
    seed: int,
    lambda_values: list[float],
):
    models = [
        EpsilonGreedy(epsilon=epsilon, seed=seed),
        LinUCB(feature_dim=feature_dim, alpha=alpha, seed=seed),
    ]
    models.extend(
        DiversityAwareLinUCB(
            feature_dim=feature_dim,
            alpha=alpha,
            lambda_diversity=lambda_value,
            recent_window=recent_window,
            seed=seed,
        )
        for lambda_value in lambda_values
    )
    return models


def save_outputs(recommendations: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    recommendations_path = output_dir / "recommendations.csv"
    metrics_path = output_dir / "metrics_summary.csv"
    summary_text_path = output_dir / "experiment_summary.txt"

    recommendations.to_csv(recommendations_path, index=False)
    summary.to_csv(metrics_path, index=False)
    write_experiment_summary(summary, summary_text_path)

    plot_cumulative_metric(
        recommendations,
        value_column="reward",
        title="Cumulative Reward",
        output_path=output_dir / "cumulative_reward.png",
    )
    plot_cumulative_metric(
        recommendations,
        value_column="regret",
        title="Cumulative Regret",
        output_path=output_dir / "cumulative_regret.png",
    )
    plot_reward_vs_diversity(summary, output_dir / "reward_vs_diversity.png")
    plot_metrics_by_model(summary, output_dir / "metrics_by_model.png")
    plot_lambda_sensitivity(summary, output_dir / "lambda_sensitivity.png")
    return {
        "recommendations": recommendations_path,
        "metrics": metrics_path,
        "summary_text": summary_text_path,
        "plots_dir": output_dir,
    }


def print_run_summary(
    split,
    feature_builder: FeatureBuilder,
    recommendations: pd.DataFrame,
    metrics_path: Path,
    plots_dir: Path,
) -> None:
    per_model_counts = recommendations.groupby("model").size().to_dict()
    recommendations_text = ", ".join(f"{model}: {count}" for model, count in per_model_counts.items()) or "none"

    print(f"Number of users used: {len(split.selected_user_ids)}")
    print(f"Number of train ratings: {len(split.train)}")
    print(f"Number of test ratings: {len(split.test)}")
    print(f"Number of unique candidate movies: {split.test['movieId'].nunique()}")
    print(f"Number of genres: {len(feature_builder.genre_names)}")
    print(f"Number of recommendations generated per model: {recommendations_text}")
    print(f"Path to metrics_summary.csv: {metrics_path}")
    print(f"Path to plots directory: {plots_dir}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    ratings_path = args.data_dir / "ratings.csv"
    movies_path = args.data_dir / "movies.csv"

    try:
        validate_movielens_files(ratings_path=ratings_path, movies_path=movies_path, data_dir=args.data_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if args.verbose:
        print(f"Using data directory: {args.data_dir}")
        print(f"Writing outputs to: {args.output_dir}")

    try:
        split = load_movielens_data(
            ratings_path=ratings_path,
            movies_path=movies_path,
            min_user_ratings=args.min_ratings_per_user,
            test_fraction=args.test_size,
            seed=args.seed,
            max_users=args.max_users,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if args.verbose and split.skipped_user_ids:
        print(f"Skipped users during splitting: {len(split.skipped_user_ids)}")

    feature_builder = FeatureBuilder(train=split.train, movies=split.movies)
    models = build_models(
        feature_dim=feature_builder.feature_dim,
        alpha=args.alpha,
        epsilon=args.epsilon,
        recent_window=args.recent_window,
        seed=args.seed,
        lambda_values=args.lambda_values,
    )

    recommendation_frames: list[pd.DataFrame] = []
    for model in models:
        if args.verbose:
            print(f"Running model: {model.name}")
        result = run_offline_simulation(
            model=model,
            test=split.test,
            user_train_average=split.user_train_average,
            feature_builder=feature_builder,
            max_recommendations_per_user=args.max_recommendations_per_user,
            recent_window=args.recent_window,
        )
        recommendation_frames.append(result.recommendations)

    recommendations = pd.concat(recommendation_frames, ignore_index=True)
    evaluation_catalog_movie_ids = set(split.test["movieId"].unique().tolist())
    summary = compute_metrics_summary(recommendations, feature_builder, evaluation_catalog_movie_ids)

    output_paths = save_outputs(recommendations, summary, args.output_dir)
    print_run_summary(
        split=split,
        feature_builder=feature_builder,
        recommendations=recommendations,
        metrics_path=output_paths["metrics"],
        plots_dir=output_paths["plots_dir"],
    )


if __name__ == "__main__":
    main()
