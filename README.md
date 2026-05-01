# Diversity-Aware Contextual Bandits for Movie Recommendation

This project implements an offline evaluation pipeline for movie recommendation with the MovieLens small dataset. It compares:

- `EpsilonGreedy`
- `LinUCB`
- `DiversityAwareLinUCB` with configurable `lambda` values, defaulting to `0.0`, `0.1`, `0.3`, and `0.5`

The pipeline uses a per-user train/test split, train-only feature construction, binary bandit rewards, and diversity-aware scoring.

## Project Structure

```text
src/
  data_loader.py
  features.py
  models.py
  simulation.py
  metrics.py
  plotting.py
main.py
requirements.txt
README.md
data/
outputs/
```

## Dataset

Place the MovieLens small CSV files here:

```text
data/ratings.csv
data/movies.csv
```

The dataset is not included in this repository. Download `ml-latest-small` from the official GroupLens page:

- [MovieLens latest datasets](https://grouplens.org/datasets/movielens/latest/)
- [Direct ml-latest-small zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)

## Methodology

### Data split

- Filter users with fewer than `20` ratings.
- For each remaining user, randomly split ratings into `80%` train and `20%` test using seed `42`.
- Compute each user's average rating from training data only.

### Reward definition

- Binary reward for bandit updates:
  - `reward = 1 if test_rating > user_train_average else 0`
- Rating reward for reporting:
  - `rating_reward = test_rating / 5.0`

### Features

Each movie context vector is:

```text
[movie_genre_vector,
 user_genre_preference_vector,
 user_genre_preference_vector * movie_genre_vector,
 movie_avg_rating,
 movie_popularity]
```

Feature details:

- `movie_genre_vector`: multi-hot genre encoding from `movies.csv`
- `user_genre_preference_vector`: average normalized training rating (`rating / 5`) per genre for that user
- `movie_avg_rating`: training-only movie average rating, normalized by `5`
- `movie_popularity`: training-only movie interaction count, normalized by the maximum training count

### Models

#### Epsilon-Greedy

- `epsilon = 0.1`
- Maintains empirical average binary reward per movie
- Randomly explores with probability `epsilon`

#### LinUCB

- `alpha = 0.5`
- Uses:
  - `score = theta.T @ x + alpha * sqrt(x.T @ A^-1 @ x)`
- Updates:
  - `A = A + x x.T`
  - `b = b + reward * x`

#### Diversity-Aware LinUCB

- Uses the same LinUCB update rule
- Adds diversity bonus:
  - `score = theta.T @ x + alpha * uncertainty + lambda * diversity_bonus`
- Diversity bonus:
  - `1 - max_jaccard_similarity(candidate_genres, recent_recommended_genres)`
- `recent_window = 5`

## Evaluation

Recommendations are generated only from each user's held-out test set, and the same movie is never recommended twice to the same user.

By default, the simulator uses a recommendation budget of `10` items per user. This is configurable with `--max-recommendations-per-user`. The default budget is used so model comparisons remain meaningful; if every held-out movie is recommended, final reward totals become nearly identical across models.

Reported metrics:

- Hit rate
- Average rating reward
- Cumulative reward
- Intra-list diversity
- Catalog coverage
- Genre coverage
- Average recommended movie popularity
- Cumulative regret

Regret is computed per user step as:

```text
best_possible_reward_among_remaining_candidates - chosen_reward
```

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the MovieLens files in:

```text
data/ratings.csv
data/movies.csv
```

Quick debug run:

```bash
python main.py --max_users 100
```

Full run:

```bash
python main.py
```

Run with custom lambda values:

```bash
python main.py --lambda_values 0.0,0.05,0.1,0.3,0.5,1.0
```

Useful options:

```bash
python main.py --data_dir data --output_dir outputs --max_recommendations_per_user 10 --test_size 0.2 --min_ratings_per_user 20 --verbose
```

Use `--max_recommendations_per_user 0` to evaluate the full held-out set for each user.

## Outputs

The run writes:

- `outputs/recommendations.csv`
- `outputs/metrics_summary.csv`
- `outputs/experiment_summary.txt`
- `outputs/cumulative_reward.png`
- `outputs/cumulative_regret.png`
- `outputs/reward_vs_diversity.png`
- `outputs/metrics_by_model.png`
- `outputs/lambda_sensitivity.png`

All outputs are saved under the directory provided by `--output_dir`, which defaults to `outputs/`.

## Reproducibility

- Random seed defaults to `42`
- `random.seed(42)` and `np.random.seed(42)` are both set by default
- Train/test split uses only training data for all learned statistics and features
- No test data is used to build user preferences, movie averages, or popularity features
