# Copyright 2025 The polarix Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Normalization functions given a set of scores."""

import chex
import distrax
import jax
import jax.numpy as jnp


################################################################################
# Single-sided normalizers.
################################################################################


def ptp(scores: chex.Array) -> chex.Array:
  """Returns peak-to-peak normalized scores."""
  chex.assert_rank(scores, 1)
  scores -= jnp.min(scores)
  # Handle all-constant scores gracefully.
  max_score = jnp.max(scores)
  max_score = jnp.where(max_score == 0.0, 1.0, max_score)
  return scores / max_score


def rank(scores: chex.Array) -> chex.Array:
  """Returns rank of array along axis."""
  chex.assert_rank(scores, 1)

  def _rank(array: chex.Array) -> chex.Array:
    return jnp.argsort(jnp.argsort(array)).astype(jnp.float32) / (
        array.size - 1
    )

  # Entries with equal values are assigned equal rank.
  # Assumes _rank uses a stable sort.
  rank_ = _rank(scores)
  inverse_rank_ = _rank(scores[::-1])[::-1]
  return (rank_ + inverse_rank_) / 2


def uvzm(scores: chex.Array) -> chex.Array:
  """Returns unit-variance zero-maximum normalized ratings.

  NOTE: unit-variance should result in less extreme normalization than
  peak-to-peak. Because strong models are the most salient in ranking we
  normalize around the max so that the difference between the max models between
  tasks is zero.

  Args:
    scores: Array with shape [|A|].

  Returns:
      unit-variance zero-maximum normalized scores of shape [|A|].
  """
  chex.assert_rank(scores, 1)
  # Handle all-constant scores gracefully.
  std_score = jnp.std(scores)
  std_score = jnp.where(std_score == 0.0, 1.0, std_score)
  scores /= std_score
  return scores - jnp.max(scores)


def expected_rank(
    scores: chex.Array,
    scores_stddev: chex.Array,
    key: chex.PRNGKey,
    num_samples: int = 512,
) -> chex.Array:
  """Returns a Monte-Carlo estimate of expected-rank.

  Higher is better.

  Args:
    scores: expected scores with shape [|A|].
    scores_stddev: standard deviations of scores with shape [|A|].
    key: PRNG key.
    num_samples: Number of samples to take.

  Returns:
    Normalized expected rank of shape [|A|].
  """
  chex.assert_rank((scores, scores_stddev), 1)
  chex.assert_equal_shape((scores, scores_stddev))
  dist = distrax.Normal(loc=scores, scale=scores_stddev)
  samples = dist.sample(seed=key, sample_shape=(num_samples,))
  # [S, |A|] -> [S, rank(|A|)]
  sample_ranks = jax.vmap(rank)(samples)
  return jnp.mean(sample_ranks, axis=0)


################################################################################
# SxS normalizers using single-sided score distributions/samples.
################################################################################


def win(scores: chex.Array) -> chex.Array:
  """Returns whether an agent wins against another.

  Args:
    scores: Array with shape [|A|].

  Returns:
    Array with shape [|A|,|A|].
  """
  chex.assert_rank(scores, 1)
  diff = scores[:, jnp.newaxis] - scores[jnp.newaxis, :]
  return jnp.sign(diff) / 2 + 0.5


def winrate(
    scores: chex.Array, scores_stddev: chex.Array, min_stddev: float = 1e-6
) -> chex.Array:
  """Returns the winrate of each agent versus each other.

  Args:
    scores: Array with shape [|A|].
    scores_stddev: Array with shape [|A|].
    min_stddev: Minimum standard deviation to use.

  Returns:
    Array with shape [|A|,|A|].
  """
  chex.assert_rank((scores, scores_stddev), 1)
  chex.assert_equal_shape((scores, scores_stddev))
  # The probability that normal random variable X is greater than normal random
  # variable Y is given by:
  #   P(X > Y) = CDF((u_X - u_Y)/sqrt(var_X + var_Y))
  numerator = scores[:, jnp.newaxis] - scores[jnp.newaxis, :]
  var = jnp.square(jnp.clip(scores_stddev, min=min_stddev))
  denominator = jnp.sqrt(var[:, jnp.newaxis] + var[jnp.newaxis, :])
  return jax.scipy.stats.norm.cdf(numerator / denominator)


def winloss_counts(score_samples: chex.Array) -> tuple[chex.Array, chex.Array]:
  """Returns the winloss counts of each agent versus each other.

  Args:
    score_samples: a tensor of sample scores for each agent of shape [|A|, |S|].
      If some agents have fewer samples than |S|, then missing entries are
      padded by `np.nan`.

  Returns:
    A tuple of win, loss counts of shape [|A|, |A|].
  """
  chex.assert_rank(score_samples, 2)

  def _ensure_padding_right(arr: chex.Array) -> chex.Array:
    return arr[jnp.argsort(jnp.isnan(arr), axis=-1)]

  score_samples = jax.vmap(_ensure_padding_right)(score_samples)

  # [|A|, 1, |S|] - [1, |A|, |S|] -> [|A|, |A|, |S|]
  diff_samples = score_samples[:, jnp.newaxis] - score_samples[jnp.newaxis]
  e = 0.5 * jnp.sum(jnp.equal(diff_samples, 0.0), axis=-1)
  a = jnp.sum(jnp.greater(diff_samples, 0.0), axis=-1) + e
  b = jnp.sum(jnp.less(diff_samples, 0.0), axis=-1) + e
  return a, b


def beta(
    score_samples: chex.Array, pseudo_count: float = 1.0
) -> tuple[chex.Array, chex.Array]:
  """Returns the winrate of agents versus each other assuming beta distribution.

  Args:
    score_samples: a tensor of sample scores for each agent of shape [|A|, |S|].
      If some agents have fewer samples than |S|, then missing entries are
      padded by `np.nan`.
    pseudo_count: A pseudo-count to add to the beta distribution.

  Returns:
    A tuple of alpha, beta concentration parameters of shape [|A|, |A|].
  """
  a, b = winloss_counts(score_samples)
  return a + pseudo_count, b + pseudo_count
