# Diversity-Aware Contextual Bandits for Movie Recommendation

**CSCE 550 — Final Project Report**

**Authors:** Shriyansh Singh (Student ID: 028243304) and Amirali Kalhor (Student ID: 035266710)

---

## Abstract

We study a contextual-bandit formulation of top-N movie recommendation and ask whether explicitly trading off short-term reward for *list diversity* changes the regret–satisfaction profile of the recommender. We implement and compare three algorithms — non-contextual ε-greedy, LinUCB, and a diversity-aware variant of LinUCB that adds a Jaccard-based novelty bonus over a sliding window — on the MovieLens-100k (`ml-latest-small`) dataset using a strict offline evaluation protocol with per-user 80/20 splits and binary rewards anchored to each user's training-time average rating. Across 610 users and 5,349 recommendations per algorithm, LinUCB beats ε-greedy by 9.9% in hit rate and 19.5% in cumulative regret, confirming that the linear-payoff assumption pays off on this dataset. Sweeping the diversity weight λ ∈ {0, 0.1, 0.3, 0.5} produces a clean, monotone trade-off: a 2.3% drop in hit rate "buys" a 4.9% gain in intra-list diversity and a 3.6% gain in catalog coverage. The trade-off is small in magnitude but consistent in direction, matching the qualitative prediction of theory.

---

## 1. Introduction

Modern recommender systems sit on top of two competing pressures. They want to deliver items the user is likely to enjoy *now* (exploitation), but they also need to (a) keep learning about items they have not shown enough times (exploration) and (b) avoid producing a monotonous, over-personalized feed that hurts long-run engagement and item discovery (diversity) [Carbonell and Goldstein, 1998; Castells et al., 2015]. Contextual multi-armed bandits formalize the exploration–exploitation half of that picture: the recommender is repeatedly given a context (a user, a candidate item, side information) and must pick an arm whose expected payoff is unknown but assumed to depend in some structured way on the context [Auer, 2002; Li et al., 2010].

The first contribution of this project is a clean, reproducible offline-evaluation pipeline for contextual bandits on MovieLens. The second is an empirical study of the well-known but rarely-quantified claim that *adding a diversity bonus to the bandit's score function improves catalog coverage and intra-list diversity, but at a cost in short-term reward*. We sweep a single diversity weight λ and report the trade-off curve.

The class material covered the statistical-learning view of generalization (PAC, VC dimension, Rademacher complexity, online learning, SVMs). Bandit problems pick up exactly where those tools end: the learner observes only the *bandit feedback* of the action it actually took, which makes them a natural extension of the online-learning framework studied in class [Shalev-Shwartz, 2012; Lattimore and Szepesvári, 2020].

---

## 2. Background and related work

**Multi-armed bandits.** In the classical *K*-armed bandit, the learner chooses one of *K* actions per round and observes the reward of only that action. The performance metric is *regret*, the gap between the learner's cumulative reward and that of the best fixed action in hindsight. UCB-style algorithms achieve regret O(√(KT log T)) by adding to each arm's empirical mean an exploration bonus that shrinks with the number of pulls of that arm [Auer et al., 2002].

**Contextual bandits and LinUCB.** When each round comes with a context vector x ∈ ℝᵈ and the expected reward of pulling arm a in context x is assumed to be linear, E[r | x, a] = θ_aᵀ x, the LinUCB algorithm of Li et al. [2010] maintains a ridge-regression estimate θ̂ for each arm and picks the arm that maximises θ̂ᵀ x + α√(xᵀA⁻¹x). The width term is a Bayesian-style 1−δ confidence radius for the linear estimate. Under standard regularity assumptions, LinUCB enjoys a regret bound of Õ(d√T) — independent of the number of arms — which is what makes it attractive for recommender settings with very large catalogs [Chu et al., 2011; Abbasi-Yadkori et al., 2011].

**Diversity in recommendation.** Carbonell and Goldstein's MMR criterion [1998] re-scores candidates with a "−λ × max similarity to already chosen items" term to balance relevance and novelty. Subsequent work has formalised this under names such as *intra-list diversity* and *catalog coverage* [Ziegler et al., 2005; Castells et al., 2015]. A diversity bonus inside a bandit score function is a small but meaningful generalisation: the bonus changes which item the bandit *explores*, not just which it *displays*, so it interacts with the regret bound rather than sitting outside it.

**Offline evaluation.** Faithful offline evaluation of bandits is hard because the logged data was collected by some other policy [Li et al., 2011]. We sidestep this by using the standard "leave-out-test-ratings" protocol: each user's ratings are split 80/20, the bandit is allowed to recommend only items from the held-out 20%, and reward is read off the held-out rating. This is biased toward items the user actually rated (selection bias) but is the de-facto baseline in the academic literature on MovieLens-style bandit evaluation.

---

## 3. Problem setup and methodology

### 3.1 Dataset

We use the publicly available `ml-latest-small` snapshot of MovieLens [Harper and Konstan, 2015], which contains 100,836 ratings from 610 users on 9,742 movies, plus a multi-label genre tag for each movie (20 genres after normalisation). All 610 users have at least 20 ratings, satisfying our `min_ratings_per_user = 20` filter automatically.

### 3.2 Per-user train/test split

For each user we shuffle their ratings with a fixed RNG (seed 42) and assign the first 80% to training and the remaining 20% to test. Two consequences of this design are worth flagging:

1. *All* learned statistics — the user's average rating, per-genre preference vector, per-movie popularity, per-movie average rating — use **only training data**. No test rating leaks into a feature.
2. The candidate set at evaluation time is the user's held-out 20%. This means hit rates are not directly comparable to a setting that lets the model rank the entire catalog, but it makes regret well-defined: at every step the *best possible* binary reward among the remaining held-out items is computable.

### 3.3 Reward

We use a thresholded binary reward to make UCB-style updates well-conditioned:

> r = 1 if test_rating > user's training-time average rating, else 0.

Anchoring the threshold to the *user's own* training average normalises away systematic rater leniency. We also report `rating_reward = test_rating / 5` for context, but the bandit only ever sees the binary signal.

### 3.4 Context features

Each (user, movie) pair is mapped to a context vector

> x = [m_g , u_g , u_g ⊙ m_g , avg_rating , popularity ] ∈ ℝ³ᴳ⁺²

where G = 20 is the number of genres, m_g is the movie's multi-hot genre vector, u_g is the user's training-time genre preference vector (per-genre average of the user's normalised training ratings, restricted to movies actually tagged with that genre), u_g ⊙ m_g is their element-wise product (a "preference match" feature), and the last two scalars are the movie's training-time average rating (in [0,1]) and its training-time popularity normalised by the most-rated movie's count. This gives `feature_dim = 62`.

### 3.5 Algorithms

**ε-greedy (non-contextual baseline).** Maintains a per-movie running average of the binary reward; with probability ε = 0.1 picks a uniformly random candidate, otherwise picks the candidate with the highest empirical mean (ties broken at random). Provides a sanity floor: if a contextual model cannot beat this, the features are not helping.

**LinUCB.** A single ridge-regression model shared across all arms, with feature_dim-dimensional design matrix A (initialised to I) and target vector b. At each step the score for candidate x is

> score(x) = θ̂ᵀ x + α · √(xᵀ A⁻¹ x), where θ̂ = A⁻¹ b.

We use α = 0.5. After observing reward r for context x, A ← A + xxᵀ and b ← b + r·x.

**Diversity-aware LinUCB.** Same ridge model and same UCB term, but the score gets an additional novelty bonus:

> score(x, g) = θ̂ᵀ x + α · √(xᵀ A⁻¹ x) + λ · (1 − max_{h ∈ H} J(g, h))

where g is the candidate's genre set, H is the genre sets of the last *recent_window* = 5 recommendations to the same user, and J is Jaccard similarity. We sweep λ ∈ {0, 0.1, 0.3, 0.5}. By construction, λ = 0 reduces exactly to LinUCB — a useful invariant we verify empirically (Section 6).

### 3.6 Simulation loop

For each user, we initialise a per-user genre history `H = ∅` and repeat for up to *budget* = 10 steps (or until the candidate set is empty):

1. Build the candidate context for every remaining held-out movie.
2. Compute `best_possible_reward` (max binary reward among remaining candidates).
3. Ask the model to pick one candidate.
4. Observe its binary reward, update the model, append the chosen movie's genre set to H.
5. Record `regret = best_possible_reward − chosen_reward`, plus the score, uncertainty, and diversity bonus that drove the choice.

Limiting the budget to 10 per user is essential: if every model is asked to recommend every held-out movie, all final-reward totals collapse to the same number (the sum over the test set), and only the *order* differs. With a budget of 10 the ordering matters and the metrics separate.

---

## 4. Theory

### 4.1 LinUCB's confidence ellipsoid

LinUCB is the bandit instantiation of a more general principle: *optimism in the face of uncertainty*. Under the assumption E[r | x] = θ*ᵀ x with sub-Gaussian noise, the ridge-regression estimator θ̂_t = A_t⁻¹ b_t with A_t = λ_reg I + Σ_{s≤t} x_s x_sᵀ satisfies

> ‖θ̂_t − θ*‖_{A_t} ≤ β_t with probability ≥ 1 − δ,

for an explicit β_t that grows like √(d log t) [Abbasi-Yadkori et al., 2011]. This means the true reward θ*ᵀ x lies within ±β_t · √(xᵀ A_t⁻¹ x) of θ̂_tᵀ x. LinUCB substitutes that exact width for the heuristic α · √(xᵀ A⁻¹ x), and *always picks the most optimistic estimate among candidates*. Because the upper-confidence bound dominates the true reward with high probability, the chosen action's true reward cannot be far below the optimal action's true reward — which is precisely the engine of the regret bound.

### 4.2 Regret bound

For LinUCB-style algorithms with d-dimensional contexts and horizon T, the cumulative regret R_T satisfies

> R_T = Õ( d · √T )

ignoring polylog factors (Theorem 3, Abbasi-Yadkori et al., 2011). The intuition: the confidence width on the chosen direction shrinks at rate ≈ 1/√n, so summing the per-step regret along the played sequence telescopes through the log-determinant of A_T, which is bounded by d · log(1 + T/d). The bound is *independent of the number of arms*, which is what makes the linear assumption attractive for recommendation: we have ≈9,000 movies but only 62 features.

### 4.3 Why ε-greedy lags

ε-greedy with constant ε ignores both context (its estimate is a per-arm scalar) and uncertainty (it explores uniformly). Its expected regret in the contextual setting is Ω(εT) from forced exploration alone. In our experiments below this shows up as ~24% more cumulative regret than LinUCB.

### 4.4 The diversity bonus

The bonus `b(x, H) = 1 − max_{h ∈ H} J(g(x), h)` is bounded in [0, 1] (taking value 1 when the candidate is genre-disjoint from everything in the recent window, 0 when it duplicates one of them). Since the bonus is bounded and additive, it shifts every candidate's score by an amount in [0, λ]. Crucially, the *gap* between two candidates' UCB scores is preserved up to ±λ. So choosing λ controls how much "exploration toward novelty" the bandit will tolerate at the cost of exploiting its current best estimate. In the limit λ = 0 the algorithm is exactly LinUCB; as λ grows, the bandit increasingly behaves like an MMR re-ranker [Carbonell and Goldstein, 1998] but with the relevance signal coming from a confidence-aware linear model rather than a static similarity score.

### 4.5 What theory predicts for our experiment

1. LinUCB should beat ε-greedy on cumulative reward and regret because the linear payoff model is a reasonable fit to the genre/popularity features and ε-greedy throws away that structure.
2. λ = 0 of DiversityAwareLinUCB should *exactly match* LinUCB step-for-step (same RNG, same updates).
3. As λ increases, hit rate and cumulative reward should fall *monotonically* and intra-list diversity / catalog coverage should rise *monotonically*, with the slope set by the empirical Jaccard distance distribution among MovieLens genres.

---

## 5. Experiments

### 5.1 Setup

We run all six models — ε-greedy, LinUCB, and DiversityAwareLinUCB at λ ∈ {0, 0.1, 0.3, 0.5} — under identical settings: 610 users, 80/20 per-user split (seed 42), `feature_dim = 62`, α = 0.5, ε = 0.1, recent_window = 5, budget of 10 recommendations per user. Each model is reset between users only in the sense that the recent-genre history H starts empty; the LinUCB matrix A and vector b carry across users (a single shared model, as in the original LinUCB paper).

The full run produces 5,349 recommendations per model — fewer than the 6,100 user × budget product because some users have fewer than 10 held-out items. Total wall-clock time on a laptop is roughly 1 minute.

### 5.2 Metrics

| Metric | Definition |
|---|---|
| Hit rate | mean of binary reward |
| Avg rating reward | mean of `test_rating / 5` |
| Cumulative reward | sum of binary reward |
| Cumulative regret | sum of `best_possible − chosen` |
| Intra-list diversity | mean over users of `1 − Jaccard` between all pairs of recommended movies |
| Catalog coverage | unique recommended movies / unique held-out movies |
| Genre coverage | unique recommended genres / total genres |
| Avg recommended popularity | mean of (training-time popularity of recommended item) |

### 5.3 Verification (program-correctness checks)

To guard against the silent-failure modes that machine-learning code is most prone to we wrote a small test script (`tests/test_basic.py`, run with `python tests/test_basic.py`). It contains 27 assertions across seven test groups:

- Jaccard endpoints (identical, disjoint, half-overlap, both-empty).
- FeatureBuilder sanity: vocabulary, dimensions, value ranges, fallback behavior for unseen users/movies.
- LinUCB update math: after one update with reward r and context x, A and b are exactly I + xxᵀ and r·x and the resulting θ̂ solves Aθ̂ = b.
- Diversity-bonus endpoints: empty history → 1, identical history → 0, disjoint history → 1.
- ε-greedy with ε=0 picks the higher-reward arm.
- Simulation invariants: rows are produced, no (user, movie) pair appears twice, rewards are binary, regret is non-negative, every recommendation is in the held-out set.
- Intra-list diversity bounds: identical movies → 0, fully disjoint movies → 1.

All 27 tests pass. Two further passive checks come from the experiment itself: (a) DiversityAwareLinUCB(λ=0) reproduces LinUCB's metrics to the last decimal place — a tight invariant that catches any drift in the diversity-bonus path; and (b) cumulative regret is non-negative and monotone non-decreasing in every plot, as it must be by construction.

---

## 6. Results

### 6.1 Headline numbers

The full per-model summary (from `outputs/metrics_summary.csv`) is:

| Model | Hit rate | Avg rating reward | Intra-list diversity | Catalog coverage | Genre coverage | Cum. reward | Cum. regret |
|---|---|---|---|---|---|---|---|
| EpsilonGreedy | 0.5788 | 0.7468 | 0.7968 | 0.2606 | 0.95 | 3096 | 1893 |
| LinUCB | **0.6362** | **0.7704** | 0.7922 | 0.2951 | 1.00 | **3403** | **1525** |
| DiversityAwareLinUCB(λ=0)   | 0.6362 | 0.7704 | 0.7922 | 0.2951 | 1.00 | 3403 | 1525 |
| DiversityAwareLinUCB(λ=0.1) | 0.6308 | 0.7675 | 0.8071 | 0.2966 | 1.00 | 3374 | 1541 |
| DiversityAwareLinUCB(λ=0.3) | 0.6255 | 0.7652 | 0.8199 | 0.2990 | 1.00 | 3346 | 1583 |
| DiversityAwareLinUCB(λ=0.5) | 0.6218 | 0.7640 | **0.8314** | **0.3056** | 1.00 | 3326 | 1623 |

Three observations:

1. **LinUCB strictly dominates ε-greedy on every reward metric.** Hit rate is +9.9% (0.636 vs. 0.579), cumulative regret is −19.5% (1525 vs. 1893). Intra-list diversity is roughly tied (LinUCB is even slightly *less* diverse), which makes sense: ε-greedy's random exploration injects accidental diversity. Catalog coverage is +13.2% in LinUCB's favor, and ε-greedy is the only model that fails to reach 100% genre coverage. This confirms that the linear payoff structure is a good fit to the features.
2. **LinUCB and DiversityAwareLinUCB(λ=0) match exactly**, validating the implementation: the diversity-bonus path is a strict superset of the LinUCB path, and the sanity test also confirms it.
3. **The λ knob produces a monotone, well-ordered trade-off** across all four sweep points (Section 6.2).

### 6.2 The diversity / reward trade-off

Sweeping λ from 0 to 0.5 produces:

| Quantity | λ=0 | λ=0.1 | λ=0.3 | λ=0.5 | Change |
|---|---|---|---|---|---|
| Hit rate | 0.6362 | 0.6308 | 0.6255 | 0.6218 | −2.3% |
| Cum. regret | 1525 | 1541 | 1583 | 1623 | +6.4% |
| Intra-list diversity | 0.7922 | 0.8071 | 0.8199 | 0.8314 | +4.9% |
| Catalog coverage | 0.2951 | 0.2966 | 0.2990 | 0.3056 | +3.6% |
| Unique movies recommended | 1523 | 1531 | 1543 | 1577 | +3.5% |
| Avg recommended popularity | 0.3200 | 0.3183 | 0.3179 | 0.3160 | −1.3% |

The trade-off is exactly the shape theory predicts: every reward metric drops monotonically, every diversity metric rises monotonically, and the model gradually shifts away from the most popular long-tail head. The magnitudes are modest because MovieLens has only 20 genres, so the Jaccard-distance ceiling is hit quickly; we'd expect a much steeper trade-off on a richer tag vocabulary or with a content-embedding similarity instead of Jaccard.

### 6.3 Cumulative views

`outputs/cumulative_reward.png` shows ε-greedy diverging from the four LinUCB-family curves within the first ~500 interactions and never recovering, while the four λ-variants stay nearly indistinguishable for the first 1000 steps and then fan out by a small but steady amount. `outputs/cumulative_regret.png` is its mirror: ε-greedy's curve has the steepest slope; among the LinUCB family, λ=0 has the shallowest, λ=0.5 the steepest. `outputs/lambda_sensitivity.png` makes the four-point trend curves explicit.

### 6.4 Where each model wins

`outputs/reward_vs_diversity.png` is the headline plot of the report: the six models trace out a Pareto frontier in (intra-list diversity, avg rating reward) space. ε-greedy sits *below and to the left of* the LinUCB family — strictly dominated. Within the LinUCB family the four λ values trace a smooth front from the high-reward / low-diversity corner (λ=0) to the high-diversity / lower-reward corner (λ=0.5). There is no single best operating point; the choice is a product decision about how much short-term satisfaction to trade for catalog exposure.

---

## 7. Discussion

**The contextual gain is real and large.** A 9.9% hit-rate lift over ε-greedy from a 62-dimensional, hand-engineered feature vector is a clean illustration of the value of the linear-payoff assumption — the same assumption that makes LinUCB's regret bound independent of the number of arms.

**The diversity gain is real but small in this dataset.** A 5% gain in intra-list diversity for a 2.3% loss in hit rate is the right shape but a modest magnitude. We attribute this to the coarseness of MovieLens genres: with only 20 genres, the Jaccard-distance distribution between random pairs of movies is already heavily skewed toward 1 (most movie pairs are *already* genre-disjoint), so the marginal value of pushing further is limited. We would expect a richer tag vocabulary (e.g. plot tags, embedding-space neighbors, IMDb keyword tags) to make the trade-off curve substantially steeper. This would be the natural next experiment.

**Was anything surprising?** Two small things. First, ε-greedy's intra-list diversity is *higher* than LinUCB(λ=0)'s (0.797 vs. 0.792); we expected the opposite from a "smarter" model. The explanation is that ε-greedy's random arm of size ε = 0.1 functions as a primitive diversity mechanism — it just doesn't know which way to push. Second, catalog coverage tops out at ~31% even at λ=0.5; the bandit is not exploring anywhere near the full 5,161-movie test catalog. With a larger budget per user this would change, but it is a useful reminder that "coverage" in evaluation has a denominator.

**Limitations.** (a) Offline evaluation on a candidate set restricted to the user's held-out test items is biased toward items the user already chose to rate. (b) MovieLens-100k is small (610 users, ~100k ratings); the variance of the trade-off-curve slopes is non-negligible and we did not bootstrap. (c) We swept only one hyperparameter (λ) and held α and ε fixed; a joint sweep would give a fuller picture but was out of scope.

**Connection to course material.** LinUCB is exactly the contextual-bandit cousin of the online-mistake-bound algorithms covered in class: it is a regret-minimizer in a structured action space that achieves √T regret by playing the upper confidence bound of a ridge-regression estimate. The diversity bonus is a small, principled change to the score function that does not touch the regret machinery.

**Future work.** The two cleanest extensions are (i) replacing the Jaccard-on-genres similarity with a learned content embedding so the diversity bonus becomes meaningful in higher dimensions; and (ii) tuning λ *online*, e.g. via meta-bandit over a finite λ grid, so the diversity weight adapts to the user instead of being a fixed knob.

---

## 8. Conclusion

We showed end-to-end that on MovieLens-100k a LinUCB recommender beats a non-contextual ε-greedy baseline by ~10% in hit rate and ~20% in cumulative regret, that adding a Jaccard-based diversity bonus at λ ∈ {0.1, 0.3, 0.5} produces a clean monotone trade-off (small loss in hit rate, larger relative gain in diversity / catalog coverage), and that the λ=0 variant exactly recovers LinUCB — a tight implementation invariant. The codebase is small, deterministic under a fixed seed, and verified by a 27-assertion test script. The main qualitative prediction of theory — that diversity pressure trades short-term reward for novelty in a controlled, monotone way — is confirmed empirically.

---

## How to reproduce

From the project root, with Python ≥ 3.10:

```bash
pip install -r requirements.txt
# Place ratings.csv and movies.csv in data/ (see README.md for download links).
python tests/test_basic.py     # 27 assertions, runs in ~1 second
python main.py --verbose       # full run, ~1 minute, writes outputs/
```

Outputs land in `outputs/`: `metrics_summary.csv`, `experiment_summary.txt`, `recommendations.csv`, and the five plots (`cumulative_reward.png`, `cumulative_regret.png`, `reward_vs_diversity.png`, `metrics_by_model.png`, `lambda_sensitivity.png`).

---

## References

- Y. Abbasi-Yadkori, D. Pál, and C. Szepesvári. *Improved algorithms for linear stochastic bandits.* NeurIPS, 2011.
- P. Auer. *Using confidence bounds for exploitation–exploration trade-offs.* JMLR, 3:397–422, 2002.
- P. Auer, N. Cesa-Bianchi, and P. Fischer. *Finite-time analysis of the multiarmed bandit problem.* Machine Learning, 47(2-3):235–256, 2002.
- J. Carbonell and J. Goldstein. *The use of MMR, diversity-based reranking for reordering documents and producing summaries.* SIGIR, 1998.
- P. Castells, N. Hurley, and S. Vargas. *Novelty and diversity in recommender systems.* In *Recommender Systems Handbook*, Springer, 2015.
- W. Chu, L. Li, L. Reyzin, and R. E. Schapire. *Contextual bandits with linear payoff functions.* AISTATS, 2011.
- F. M. Harper and J. A. Konstan. *The MovieLens datasets: history and context.* ACM TiiS, 5(4), 2015.
- T. Lattimore and C. Szepesvári. *Bandit Algorithms.* Cambridge University Press, 2020.
- L. Li, W. Chu, J. Langford, and R. E. Schapire. *A contextual-bandit approach to personalized news article recommendation.* WWW, 2010.
- L. Li, W. Chu, J. Langford, and X. Wang. *Unbiased offline evaluation of contextual-bandit-based news article recommendation algorithms.* WSDM, 2011.
- S. Shalev-Shwartz. *Online learning and online convex optimization.* Foundations and Trends in Machine Learning, 4(2):107–194, 2012.
- C.-N. Ziegler, S. M. McNee, J. A. Konstan, and G. Lausen. *Improving recommendation lists through topic diversification.* WWW, 2005.
