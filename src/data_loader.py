from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MOVIELENS_PAGE_URL = "https://grouplens.org/datasets/movielens/latest/"
MOVIELENS_ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
RATINGS_REQUIRED_COLUMNS = {"userId", "movieId", "rating"}
MOVIES_REQUIRED_COLUMNS = {"movieId", "title", "genres"}


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    movies: pd.DataFrame
    user_train_average: pd.Series
    selected_user_ids: list[int]
    skipped_user_ids: list[int]


def _validate_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"{name} is missing required columns: {missing_str}")


def build_missing_data_message(data_dir: str | Path, ratings_path: str | Path, movies_path: str | Path) -> str:
    return (
        "MovieLens latest-small files were not found or are incomplete.\n"
        f"Download the dataset from:\n"
        f"  {MOVIELENS_PAGE_URL}\n"
        f"  {MOVIELENS_ZIP_URL}\n"
        "Then extract and place these files in your project data directory:\n"
        f"  {Path(ratings_path)}\n"
        f"  {Path(movies_path)}\n"
        f"Current data directory: {Path(data_dir)}"
    )


def validate_movielens_files(
    ratings_path: str | Path,
    movies_path: str | Path,
    data_dir: str | Path | None = None,
) -> None:
    ratings_path = Path(ratings_path)
    movies_path = Path(movies_path)
    data_dir = Path(data_dir) if data_dir is not None else ratings_path.parent

    missing_files = [path for path in [ratings_path, movies_path] if not path.exists()]
    if missing_files:
        missing_list = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            f"Missing required dataset files: {missing_list}\n\n"
            f"{build_missing_data_message(data_dir, ratings_path, movies_path)}"
        )

    try:
        ratings_header = pd.read_csv(ratings_path, nrows=0)
        movies_header = pd.read_csv(movies_path, nrows=0)
    except Exception as exc:
        raise ValueError(
            "Failed to read MovieLens CSV headers.\n\n"
            f"{build_missing_data_message(data_dir, ratings_path, movies_path)}"
        ) from exc

    _validate_columns(ratings_header, RATINGS_REQUIRED_COLUMNS, "ratings.csv")
    _validate_columns(movies_header, MOVIES_REQUIRED_COLUMNS, "movies.csv")


def load_movielens_data(
    ratings_path: str | Path,
    movies_path: str | Path,
    min_user_ratings: int = 20,
    test_fraction: float = 0.2,
    seed: int = 42,
    max_users: int | None = None,
) -> DatasetSplit:
    ratings_path = Path(ratings_path)
    movies_path = Path(movies_path)

    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be between 0 and 1. Received: {test_fraction}")
    if min_user_ratings < 1:
        raise ValueError(f"min_user_ratings must be at least 1. Received: {min_user_ratings}")
    if max_users is not None and max_users < 1:
        raise ValueError(f"max_users must be at least 1 when provided. Received: {max_users}")

    validate_movielens_files(ratings_path=ratings_path, movies_path=movies_path, data_dir=ratings_path.parent)

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    _validate_columns(ratings, RATINGS_REQUIRED_COLUMNS, "ratings.csv")
    _validate_columns(movies, MOVIES_REQUIRED_COLUMNS, "movies.csv")

    user_counts = ratings.groupby("userId").size()
    eligible_users = sorted(int(user_id) for user_id in user_counts[user_counts >= min_user_ratings].index.tolist())
    if max_users is not None:
        eligible_users = eligible_users[:max_users]

    ratings = ratings[ratings["userId"].isin(eligible_users)].copy()

    movies = movies.copy()
    movies["title"] = movies["title"].fillna("Unknown Title")
    movies["genres"] = movies["genres"].fillna("(no genres listed)")

    merged = ratings.merge(movies, on="movieId", how="left")
    merged["title"] = merged["title"].fillna("Unknown Title")
    merged["genres"] = merged["genres"].fillna("(no genres listed)")

    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    selected_user_ids: list[int] = []
    skipped_user_ids: list[int] = []

    for user_id, user_frame in merged.groupby("userId", sort=True):
        user_id = int(user_id)
        user_frame = user_frame.reset_index(drop=True)
        permutation = rng.permutation(len(user_frame))
        user_frame = user_frame.iloc[permutation].reset_index(drop=True)
        train_size = int(np.floor((1.0 - test_fraction) * len(user_frame)))
        test_size = len(user_frame) - train_size

        if train_size < 1 or test_size < 1:
            skipped_user_ids.append(user_id)
            continue

        train_frame = user_frame.iloc[:train_size].copy()
        test_frame = user_frame.iloc[train_size:].copy()
        if train_frame.empty or test_frame.empty:
            skipped_user_ids.append(user_id)
            continue

        train_parts.append(train_frame)
        test_parts.append(test_frame)
        selected_user_ids.append(user_id)

    if not train_parts or not test_parts:
        raise ValueError(
            "No users remained after filtering and train/test splitting. "
            "Try lowering --min_ratings_per_user, increasing --test_size bounds, or removing --max_users."
        )

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    user_train_average = train.groupby("userId")["rating"].mean().sort_index()

    return DatasetSplit(
        train=train.sort_values(["userId", "movieId"]).reset_index(drop=True),
        test=test.sort_values(["userId", "movieId"]).reset_index(drop=True),
        movies=movies.sort_values("movieId").reset_index(drop=True),
        user_train_average=user_train_average,
        selected_user_ids=selected_user_ids,
        skipped_user_ids=skipped_user_ids,
    )
