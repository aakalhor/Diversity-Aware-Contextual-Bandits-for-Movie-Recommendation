"""Sanity tests for the diversity-aware contextual bandit pipeline.

Run from the project root with:

    python tests/test_basic.py

The goal is not full coverage but to catch the kinds of silent bugs a learning
program is most prone to: feature-shape mistakes, mis-applied LinUCB updates,
similarity-metric off-by-ones, and broken simulation invariants.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import FeatureBuilder
from src.metrics import compute_intra_list_diversity, jaccard_similarity
from src.models import DiversityAwareLinUCB, EpsilonGreedy, LinUCB
from src.simulation import run_offline_simulation


PASS = 0
FAIL = 0
FAIL_MESSAGES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        message = f"  FAIL  {name}" + (f" -- {detail}" if detail else "")
        FAIL_MESSAGES.append(message)
        print(message)


def make_toy_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "title": ["A", "B", "C", "D", "E"],
            "genres": ["Action|Drama", "Drama", "Comedy", "Action|Comedy", "Romance"],
        }
    )
    train = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 2, 3, 1, 4, 5],
            "rating": [5.0, 4.0, 3.0, 4.0, 5.0, 2.0],
            "title": ["A", "B", "C", "A", "D", "E"],
            "genres": ["Action|Drama", "Drama", "Comedy", "Action|Drama", "Action|Comedy", "Romance"],
        }
    )
    test = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [4, 5, 2, 3],
            "rating": [5.0, 2.0, 4.5, 3.5],
            "title": ["D", "E", "B", "C"],
            "genres": ["Action|Comedy", "Romance", "Drama", "Comedy"],
        }
    )
    user_train_average = train.groupby("userId")["rating"].mean()
    return train, test, movies, user_train_average


def test_jaccard() -> None:
    print("Jaccard similarity")
    check("identical sets -> 1", jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0)
    check("disjoint sets -> 0", jaccard_similarity({"a"}, {"b"}) == 0.0)
    check(
        "half overlap -> 1/3",
        abs(jaccard_similarity({"a", "b"}, {"b", "c"}) - 1 / 3) < 1e-12,
    )
    check("two empty -> 1 (convention)", jaccard_similarity(set(), set()) == 1.0)


def test_feature_builder() -> None:
    print("FeatureBuilder")
    train, _, movies, _ = make_toy_dataset()
    fb = FeatureBuilder(train=train, movies=movies)

    expected_genres = sorted({"action", "drama", "comedy", "romance"})
    check("genre vocabulary correct", fb.genre_names == expected_genres)
    check(
        "feature_dim = 3 * |genres| + 2",
        fb.feature_dim == 3 * len(expected_genres) + 2,
    )

    context = fb.get_context(user_id=1, movie_id=1)
    check("context length matches feature_dim", context.shape == (fb.feature_dim,))
    check("context is finite", bool(np.isfinite(context).all()))

    # Movie 1 is Action|Drama -> the multi-hot block (first |G| dims) should sum to 2.
    movie_block = context[: len(expected_genres)]
    check("movie multi-hot sums to genre count", float(movie_block.sum()) == 2.0)

    # User-genre prefs are bounded by 1 since ratings are normalized by 5 before averaging.
    user_block = context[len(expected_genres) : 2 * len(expected_genres)]
    check(
        "user genre prefs in [0, 1]",
        bool((user_block >= -1e-12).all() and (user_block <= 1.0 + 1e-12).all()),
    )

    # Avg rating + popularity tail should be in [0, 1].
    tail = context[-2:]
    check(
        "movie_avg and popularity in [0, 1]",
        bool((tail >= -1e-12).all() and (tail <= 1.0 + 1e-12).all()),
    )

    # Unknown user / unknown movie should still return a valid context, not crash.
    fallback = fb.get_context(user_id=999, movie_id=999)
    check("unknown user/movie -> finite fallback context", bool(np.isfinite(fallback).all()))


def test_linucb_update() -> None:
    """After one update with x=e_0 and reward=1, A should be I + e_0 e_0^T and b = e_0."""
    print("LinUCB update math")
    d = 5
    model = LinUCB(feature_dim=d, alpha=0.5, seed=0)
    x = np.zeros(d)
    x[0] = 1.0
    A_before = model.A.copy()
    model.update(movie_id=1, context=x, reward=1)

    expected_A = A_before + np.outer(x, x)
    expected_b = x.copy()
    check("A updated by xx^T", np.allclose(model.A, expected_A))
    check("b updated by reward * x", np.allclose(model.b, expected_b))

    # theta = A^-1 b. With A = diag(2,1,1,1,1) and b = e_0, theta_0 should be 1/2.
    theta = np.linalg.solve(model.A, model.b)
    check("theta solves A theta = b", abs(theta[0] - 0.5) < 1e-12 and np.allclose(theta[1:], 0.0))


def test_diversity_bonus() -> None:
    print("DiversityAwareLinUCB diversity bonus")
    model = DiversityAwareLinUCB(feature_dim=4, alpha=0.5, lambda_diversity=0.3, recent_window=3, seed=0)

    check(
        "no history -> bonus 1.0",
        model._diversity_bonus({"action"}, recent_genre_history=[]) == 1.0,
    )
    check(
        "identical history -> bonus 0.0",
        abs(model._diversity_bonus({"action"}, recent_genre_history=[{"action"}])) < 1e-12,
    )
    bonus = model._diversity_bonus({"action", "drama"}, recent_genre_history=[{"comedy"}, {"romance"}])
    check("disjoint history -> bonus 1.0", abs(bonus - 1.0) < 1e-12)


def test_epsilon_greedy_pure_greedy() -> None:
    print("EpsilonGreedy with epsilon=0 picks the best-known arm")
    model = EpsilonGreedy(epsilon=0.0, seed=0)
    # Train arm 2 to look great, arm 1 to look bad.
    for _ in range(10):
        model.update(movie_id=2, context=np.zeros(1), reward=1)
        model.update(movie_id=1, context=np.zeros(1), reward=0)

    selection = model.select(
        user_id=0,
        candidate_movie_ids=[1, 2],
        candidate_contexts={1: np.zeros(1), 2: np.zeros(1)},
        candidate_genres={1: {"a"}, 2: {"b"}},
        recent_genre_history=[],
    )
    check("greedy picks the higher-reward arm", selection.movie_id == 2)


def test_simulation_invariants() -> None:
    """The simulator must not recommend the same movie to the same user twice."""
    print("Simulation invariants")
    train, test, movies, user_avg = make_toy_dataset()
    fb = FeatureBuilder(train=train, movies=movies)
    model = LinUCB(feature_dim=fb.feature_dim, alpha=0.5, seed=0)

    result = run_offline_simulation(
        model=model,
        test=test,
        user_train_average=user_avg,
        feature_builder=fb,
        max_recommendations_per_user=10,
        recent_window=5,
    )
    df = result.recommendations
    check("simulation produced rows", len(df) > 0)

    duplicates = df.duplicated(subset=["userId", "movieId"]).sum()
    check("no (user, movie) recommended twice", int(duplicates) == 0)

    rewards = set(df["reward"].unique().tolist())
    check("reward is binary {0,1}", rewards.issubset({0, 1}))

    check("regret is non-negative", bool((df["regret"] >= 0).all()))
    check(
        "score / uncertainty / diversity bonus are finite",
        bool(np.isfinite(df[["score", "uncertainty", "diversity_bonus"]].to_numpy()).all()),
    )

    # Each recommendation must come from the user's held-out test set.
    test_pairs = set(zip(test["userId"].astype(int), test["movieId"].astype(int)))
    rec_pairs = set(zip(df["userId"].astype(int), df["movieId"].astype(int)))
    check("every recommendation is in the held-out test set", rec_pairs.issubset(test_pairs))


def test_intra_list_diversity_bounds() -> None:
    print("Intra-list diversity bounds")
    train, _, movies, _ = make_toy_dataset()
    fb = FeatureBuilder(train=train, movies=movies)

    # Two movies with identical genres -> diversity 0.
    same = pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "step": 1},
            {"userId": 1, "movieId": 1, "step": 2},
        ]
    )
    check("identical movies -> diversity 0", compute_intra_list_diversity(same, fb) == 0.0)

    # Disjoint genres (Action|Drama vs Comedy) -> Jaccard 0 -> diversity 1.
    disjoint = pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "step": 1},  # Action|Drama
            {"userId": 1, "movieId": 3, "step": 2},  # Comedy
        ]
    )
    check(
        "disjoint movies -> diversity 1",
        abs(compute_intra_list_diversity(disjoint, fb) - 1.0) < 1e-12,
    )


def main() -> int:
    test_jaccard()
    test_feature_builder()
    test_linucb_update()
    test_diversity_bonus()
    test_epsilon_greedy_pure_greedy()
    test_simulation_invariants()
    test_intra_list_diversity_bounds()

    print()
    print(f"{PASS} passed, {FAIL} failed")
    if FAIL:
        print("Failures:")
        for message in FAIL_MESSAGES:
            print(message)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
