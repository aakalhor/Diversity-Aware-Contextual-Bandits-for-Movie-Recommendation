from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class SelectionResult:
    movie_id: int
    score: float
    uncertainty: float
    diversity_bonus: float


class EpsilonGreedy:
    def __init__(self, epsilon: float = 0.1, seed: int = 42) -> None:
        self.name = "EpsilonGreedy"
        self.alpha = float("nan")
        self.epsilon = epsilon
        self.lambda_value = float("nan")
        self.rng = np.random.default_rng(seed)
        self.movie_reward_sum: dict[int, float] = {}
        self.movie_count: dict[int, int] = {}

    def select(
        self,
        user_id: int,
        candidate_movie_ids: Sequence[int],
        candidate_contexts: dict[int, np.ndarray],
        candidate_genres: dict[int, set[str]],
        recent_genre_history: Sequence[set[str]],
    ) -> SelectionResult:
        del user_id, candidate_contexts, candidate_genres, recent_genre_history

        if not candidate_movie_ids:
            raise ValueError("No candidate movies available for selection.")

        if self.rng.random() < self.epsilon:
            chosen_movie = int(self.rng.choice(candidate_movie_ids))
            return SelectionResult(chosen_movie, 0.0, 0.0, 0.0)

        estimates = []
        for movie_id in candidate_movie_ids:
            count = self.movie_count.get(movie_id, 0)
            estimate = self.movie_reward_sum.get(movie_id, 0.0) / count if count > 0 else 0.0
            estimates.append((movie_id, estimate))

        best_estimate = max(score for _, score in estimates)
        best_movies = [movie_id for movie_id, score in estimates if score == best_estimate]
        chosen_movie = int(self.rng.choice(best_movies))
        return SelectionResult(chosen_movie, float(best_estimate), 0.0, 0.0)

    def update(self, movie_id: int, context: np.ndarray, reward: int) -> None:
        del context
        self.movie_reward_sum[movie_id] = self.movie_reward_sum.get(movie_id, 0.0) + float(reward)
        self.movie_count[movie_id] = self.movie_count.get(movie_id, 0) + 1


class LinUCB:
    def __init__(self, feature_dim: int, alpha: float = 0.5, seed: int = 42, name: str = "LinUCB") -> None:
        self.name = name
        self.alpha = alpha
        self.epsilon = float("nan")
        self.feature_dim = feature_dim
        self.lambda_value = float("nan")
        self.A = np.eye(feature_dim, dtype=float)
        self.b = np.zeros(feature_dim, dtype=float)
        self.rng = np.random.default_rng(seed)

    def _score(self, context: np.ndarray) -> tuple[float, float]:
        theta = np.linalg.solve(self.A, self.b)
        mean_reward = float(theta @ context)
        variance = float(context @ np.linalg.solve(self.A, context))
        uncertainty = float(np.sqrt(max(variance, 0.0)))
        return mean_reward, uncertainty

    def select(
        self,
        user_id: int,
        candidate_movie_ids: Sequence[int],
        candidate_contexts: dict[int, np.ndarray],
        candidate_genres: dict[int, set[str]],
        recent_genre_history: Sequence[set[str]],
    ) -> SelectionResult:
        del user_id, candidate_genres, recent_genre_history

        if not candidate_movie_ids:
            raise ValueError("No candidate movies available for selection.")

        best_results: list[SelectionResult] = []
        for movie_id in candidate_movie_ids:
            context = candidate_contexts[movie_id]
            mean_reward, uncertainty = self._score(context)
            score = mean_reward + self.alpha * uncertainty
            current = SelectionResult(int(movie_id), float(score), float(uncertainty), 0.0)
            if not best_results or current.score > best_results[0].score + 1e-12:
                best_results = [current]
            elif abs(current.score - best_results[0].score) <= 1e-12:
                best_results.append(current)

        if not best_results:
            raise RuntimeError("LinUCB failed to select a movie.")
        return best_results[int(self.rng.integers(0, len(best_results)))]

    def update(self, movie_id: int, context: np.ndarray, reward: int) -> None:
        del movie_id
        self.A += np.outer(context, context)
        self.b += float(reward) * context


class DiversityAwareLinUCB(LinUCB):
    def __init__(
        self,
        feature_dim: int,
        alpha: float = 0.5,
        lambda_diversity: float = 0.1,
        recent_window: int = 5,
        seed: int = 42,
    ) -> None:
        super().__init__(
            feature_dim=feature_dim,
            alpha=alpha,
            seed=seed,
            name=f"DiversityAwareLinUCB(lambda={lambda_diversity:.6g})",
        )
        self.lambda_diversity = lambda_diversity
        self.lambda_value = lambda_diversity
        self.recent_window = recent_window

    @staticmethod
    def _jaccard_similarity(left: set[str], right: set[str]) -> float:
        union = left | right
        if not union:
            return 1.0
        return len(left & right) / len(union)

    def _diversity_bonus(self, candidate_genres: set[str], recent_genre_history: Sequence[set[str]]) -> float:
        if not recent_genre_history:
            return 1.0

        trimmed_history = list(recent_genre_history)[-self.recent_window :]
        max_similarity = max(self._jaccard_similarity(candidate_genres, seen_genres) for seen_genres in trimmed_history)
        return 1.0 - max_similarity

    def select(
        self,
        user_id: int,
        candidate_movie_ids: Sequence[int],
        candidate_contexts: dict[int, np.ndarray],
        candidate_genres: dict[int, set[str]],
        recent_genre_history: Sequence[set[str]],
    ) -> SelectionResult:
        del user_id

        if not candidate_movie_ids:
            raise ValueError("No candidate movies available for selection.")

        best_results: list[SelectionResult] = []
        for movie_id in candidate_movie_ids:
            context = candidate_contexts[movie_id]
            mean_reward, uncertainty = self._score(context)
            diversity_bonus = self._diversity_bonus(candidate_genres[movie_id], recent_genre_history)
            score = mean_reward + self.alpha * uncertainty + self.lambda_diversity * diversity_bonus
            current = SelectionResult(int(movie_id), float(score), float(uncertainty), float(diversity_bonus))
            if not best_results or current.score > best_results[0].score + 1e-12:
                best_results = [current]
            elif abs(current.score - best_results[0].score) <= 1e-12:
                best_results.append(current)

        if not best_results:
            raise RuntimeError("Diversity-aware LinUCB failed to select a movie.")
        return best_results[int(self.rng.integers(0, len(best_results)))]
