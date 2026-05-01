from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


NO_GENRE_TOKEN = "no_genres_listed"


def normalize_genre_token(raw_genre: str) -> str:
    if raw_genre == "(no genres listed)":
        return NO_GENRE_TOKEN
    return raw_genre.strip().lower().replace(" ", "_")


@dataclass
class FeatureBuilder:
    train: pd.DataFrame
    movies: pd.DataFrame

    def __post_init__(self) -> None:
        if self.train.empty:
            raise ValueError("Training data is empty. FeatureBuilder requires at least one training interaction.")
        self.genre_names = self._build_genre_vocabulary()
        self.genre_index = {genre: idx for idx, genre in enumerate(self.genre_names)}
        self.movie_genre_sets = self._build_movie_genre_sets()
        self.movie_genre_vectors = self._build_movie_genre_vectors()
        self.user_genre_preferences = self._build_user_genre_preferences()
        self.movie_avg_rating, self.movie_popularity = self._build_movie_statistics()
        self.global_train_average = float(self.train["rating"].mean() / 5.0)
        self.feature_dim = len(self.genre_names) * 3 + 2

    def _build_genre_vocabulary(self) -> list[str]:
        genre_tokens: set[str] = set()
        for genres in self.movies["genres"].fillna("(no genres listed)"):
            for token in self.parse_genres(genres):
                genre_tokens.add(token)
        return sorted(genre_tokens)

    def parse_genres(self, genres: str) -> set[str]:
        if pd.isna(genres) or genres == "":
            return {NO_GENRE_TOKEN}
        tokens = [normalize_genre_token(token) for token in str(genres).split("|") if token.strip()]
        return set(tokens) if tokens else {NO_GENRE_TOKEN}

    def _build_movie_genre_sets(self) -> dict[int, set[str]]:
        genre_sets: dict[int, set[str]] = {}
        for row in self.movies.itertuples(index=False):
            genre_sets[int(row.movieId)] = self.parse_genres(row.genres)
        return genre_sets

    def _build_movie_genre_vectors(self) -> dict[int, np.ndarray]:
        movie_vectors: dict[int, np.ndarray] = {}
        for movie_id, genre_set in self.movie_genre_sets.items():
            vector = np.zeros(len(self.genre_names), dtype=float)
            for genre in genre_set:
                vector[self.genre_index[genre]] = 1.0
            movie_vectors[movie_id] = vector
        return movie_vectors

    def _build_user_genre_preferences(self) -> dict[int, np.ndarray]:
        preference_sums: dict[int, np.ndarray] = {}
        preference_counts: dict[int, np.ndarray] = {}

        for row in self.train.itertuples(index=False):
            user_id = int(row.userId)
            rating_weight = float(row.rating) / 5.0
            movie_vector = self.movie_genre_vectors[int(row.movieId)]

            if user_id not in preference_sums:
                preference_sums[user_id] = np.zeros(len(self.genre_names), dtype=float)
                preference_counts[user_id] = np.zeros(len(self.genre_names), dtype=float)

            # Average a user's normalized ratings over the genres they have seen in training.
            preference_sums[user_id] += rating_weight * movie_vector
            preference_counts[user_id] += movie_vector

        preferences: dict[int, np.ndarray] = {}
        for user_id, sums in preference_sums.items():
            counts = preference_counts[user_id]
            preferences[user_id] = np.divide(
                sums,
                counts,
                out=np.zeros_like(sums),
                where=counts > 0,
            )
        return preferences

    def _build_movie_statistics(self) -> tuple[dict[int, float], dict[int, float]]:
        rating_means = (self.train.groupby("movieId")["rating"].mean() / 5.0).to_dict()
        popularity_counts = self.train.groupby("movieId").size().to_dict()
        max_count = max(popularity_counts.values(), default=1)
        popularity = {movie_id: count / max_count for movie_id, count in popularity_counts.items()}
        return rating_means, popularity

    def get_context(self, user_id: int, movie_id: int) -> np.ndarray:
        movie_vector = self.movie_genre_vectors.get(movie_id)
        if movie_vector is None:
            movie_vector = np.zeros(len(self.genre_names), dtype=float)

        user_vector = self.user_genre_preferences.get(user_id)
        if user_vector is None:
            user_vector = np.zeros(len(self.genre_names), dtype=float)

        interaction_vector = user_vector * movie_vector
        movie_avg = self.movie_avg_rating.get(movie_id, self.global_train_average)
        movie_popularity = self.movie_popularity.get(movie_id, 0.0)

        # The context matches the requested design exactly:
        # [movie genres, user genre preferences, elementwise interaction, avg rating, popularity]
        return np.concatenate(
            [
                movie_vector,
                user_vector,
                interaction_vector,
                np.array([movie_avg, movie_popularity], dtype=float),
            ]
        ).astype(float)

    def get_movie_genre_set(self, movie_id: int) -> set[str]:
        return self.movie_genre_sets.get(movie_id, {NO_GENRE_TOKEN})
