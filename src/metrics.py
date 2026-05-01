from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import FeatureBuilder


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def compute_intra_list_diversity(recommendations: pd.DataFrame, feature_builder: FeatureBuilder) -> float:
    user_diversities: list[float] = []

    for _, user_frame in recommendations.groupby("userId"):
        movie_ids = user_frame.sort_values("step")["movieId"].tolist()
        if len(movie_ids) < 2:
            user_diversities.append(0.0)
            continue

        pairwise_dissimilarities = []
        for left_movie, right_movie in combinations(movie_ids, 2):
            left_genres = feature_builder.get_movie_genre_set(int(left_movie))
            right_genres = feature_builder.get_movie_genre_set(int(right_movie))
            pairwise_dissimilarities.append(1.0 - jaccard_similarity(left_genres, right_genres))

        user_diversities.append(float(np.mean(pairwise_dissimilarities)))

    return float(np.mean(user_diversities)) if user_diversities else 0.0


def compute_metrics_summary(
    recommendations: pd.DataFrame,
    feature_builder: FeatureBuilder,
    evaluation_catalog_movie_ids: set[int],
) -> pd.DataFrame:
    columns = [
        "model",
        "hit_rate",
        "avg_rating_reward",
        "intra_list_diversity",
        "catalog_coverage",
        "genre_coverage",
        "avg_recommended_popularity",
        "cumulative_reward_final",
        "cumulative_regret_final",
        "total_recommendations",
        "unique_recommended_movies",
        "lambda",
        "alpha",
        "epsilon",
    ]
    if recommendations.empty:
        return pd.DataFrame(columns=columns)

    summaries: list[dict[str, float | str]] = []

    for model_name, frame in recommendations.groupby("model"):
        frame = frame.sort_values("global_step").reset_index(drop=True)
        recommended_movies = set(frame["movieId"].tolist())
        recommended_genres = set()
        for movie_id in recommended_movies:
            recommended_genres.update(feature_builder.get_movie_genre_set(int(movie_id)))

        summaries.append(
            {
                "model": model_name,
                "hit_rate": float(frame["reward"].mean()),
                "avg_rating_reward": float(frame["rating_reward"].mean()),
                "intra_list_diversity": compute_intra_list_diversity(frame, feature_builder),
                "catalog_coverage": float(len(recommended_movies) / max(len(evaluation_catalog_movie_ids), 1)),
                "genre_coverage": float(len(recommended_genres) / max(len(feature_builder.genre_names), 1)),
                "avg_recommended_popularity": float(frame["movie_popularity"].mean()),
                "cumulative_reward_final": float(frame["reward"].sum()),
                "cumulative_regret_final": float(frame["regret"].sum()),
                "total_recommendations": int(len(frame)),
                "unique_recommended_movies": int(len(recommended_movies)),
                "lambda": frame["lambda"].iloc[0],
                "alpha": frame["alpha"].iloc[0],
                "epsilon": frame["epsilon"].iloc[0],
            }
        )

    return pd.DataFrame(summaries, columns=columns).sort_values("model").reset_index(drop=True)


def write_experiment_summary(summary: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if summary.empty:
        output_path.write_text("No results were available to summarize.\n", encoding="utf-8")
        return

    best_hit_rate = summary.loc[summary["hit_rate"].idxmax()]
    best_rating_reward = summary.loc[summary["avg_rating_reward"].idxmax()]
    best_diversity = summary.loc[summary["intra_list_diversity"].idxmax()]
    best_coverage = summary.loc[summary["catalog_coverage"].idxmax()]

    if best_hit_rate["model"] == best_diversity["model"] == best_coverage["model"]:
        tradeoff_text = (
            f"{best_hit_rate['model']} led both reward and diversity-oriented metrics in this run, "
            "so the reward-diversity trade-off was limited under the current evaluation budget."
        )
    else:
        tradeoff_text = (
            f"{best_hit_rate['model']} achieved the highest hit rate, while {best_diversity['model']} "
            f"delivered the strongest intra-list diversity and {best_coverage['model']} covered the broadest "
            "share of the candidate catalog. This reflects the expected trade-off: stronger diversity pressure "
            "can improve novelty and coverage, but it may reduce short-term reward."
        )

    lines = [
        f"Best model by hit_rate: {best_hit_rate['model']} ({best_hit_rate['hit_rate']:.4f})",
        f"Best model by avg_rating_reward: {best_rating_reward['model']} ({best_rating_reward['avg_rating_reward']:.4f})",
        f"Best model by intra_list_diversity: {best_diversity['model']} ({best_diversity['intra_list_diversity']:.4f})",
        f"Best model by catalog_coverage: {best_coverage['model']} ({best_coverage['catalog_coverage']:.4f})",
        "",
        tradeoff_text,
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
