from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pandas as pd

from src.features import FeatureBuilder


@dataclass
class SimulationResult:
    recommendations: pd.DataFrame


def run_offline_simulation(
    model,
    test: pd.DataFrame,
    user_train_average: pd.Series,
    feature_builder: FeatureBuilder,
    max_recommendations_per_user: int | None = 10,
    recent_window: int = 5,
) -> SimulationResult:
    records: list[dict[str, float | int | str]] = []
    global_step = 0

    for user_id, user_test_frame in test.groupby("userId", sort=True):
        if user_id not in user_train_average.index or user_test_frame.empty:
            continue

        user_average = float(user_train_average.loc[user_id])
        remaining = {
            int(row.movieId): {
                "title": row.title,
                "genres": row.genres,
                "rating": float(row.rating),
            }
            for row in user_test_frame.itertuples(index=False)
        }
        if not remaining:
            continue

        if max_recommendations_per_user is None or max_recommendations_per_user <= 0:
            budget = len(remaining)
        else:
            budget = min(int(max_recommendations_per_user), len(remaining))
        if budget <= 0:
            continue

        recent_genre_history: deque[set[str]] = deque(maxlen=recent_window)

        for step in range(1, budget + 1):
            candidate_movie_ids = list(remaining.keys())
            if not candidate_movie_ids:
                break
            candidate_contexts = {
                movie_id: feature_builder.get_context(int(user_id), int(movie_id)) for movie_id in candidate_movie_ids
            }
            candidate_genres = {
                movie_id: feature_builder.get_movie_genre_set(int(movie_id)) for movie_id in candidate_movie_ids
            }
            best_possible_reward = max((int(candidate["rating"] > user_average) for candidate in remaining.values()), default=0)

            # Each round only considers the user's unseen held-out items.
            selection = model.select(
                user_id=int(user_id),
                candidate_movie_ids=candidate_movie_ids,
                candidate_contexts=candidate_contexts,
                candidate_genres=candidate_genres,
                recent_genre_history=list(recent_genre_history),
            )

            chosen_movie_id = int(selection.movie_id)
            chosen_candidate = remaining.pop(chosen_movie_id)
            chosen_context = candidate_contexts[chosen_movie_id]
            reward = int(chosen_candidate["rating"] > user_average)
            rating_reward = float(chosen_candidate["rating"]) / 5.0
            # Regret is the gap to the best binary reward still available at this step.
            regret = int(best_possible_reward - reward)

            model.update(chosen_movie_id, chosen_context, reward)
            recent_genre_history.append(candidate_genres[chosen_movie_id])

            global_step += 1
            records.append(
                {
                    "model": model.name,
                    "lambda": getattr(model, "lambda_value", None),
                    "alpha": getattr(model, "alpha", None),
                    "epsilon": getattr(model, "epsilon", None),
                    "userId": int(user_id),
                    "step": step,
                    "global_step": global_step,
                    "movieId": chosen_movie_id,
                    "title": chosen_candidate["title"],
                    "genres": chosen_candidate["genres"],
                    "test_rating": float(chosen_candidate["rating"]),
                    "user_train_average": user_average,
                    "reward": reward,
                    "rating_reward": rating_reward,
                    "regret": regret,
                    "score": float(selection.score),
                    "uncertainty": float(selection.uncertainty),
                    "diversity_bonus": float(selection.diversity_bonus),
                    "movie_popularity": float(feature_builder.movie_popularity.get(chosen_movie_id, 0.0)),
                }
            )

    return SimulationResult(recommendations=pd.DataFrame(records))
