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

"""Side-by-side games."""

from collections.abc import Callable, Sequence
import functools

import chex
import distrax
import jax
import jax.numpy as jnp
from polarix._src.games import base
from polarix._src.games import normalize


def diff_game(
    *,
    loc: chex.Array,
    scale: chex.Array,
    actions: Sequence[chex.Array],
    players: Sequence[str],
    utilities: Sequence[Callable[[chex.Array], chex.Array]],
    normalizer: Callable[[chex.Array], chex.Array],
) -> base.Game:
  """Returns a SxS score difference game given a normal score distribution.

  Args:
    loc: A (..., |A|) tensor where each element is the expected score (higher is
      better) for a choice of player actions.
    scale: A (..., |A|) tensor where each element is the stddev of the score for
      a choice of player actions. If not set, variance is assumed to be 0.
    actions: a list of actions whose lengths correspond to the leading
      dimensions of `loc`/`scale` and ends with a vector of length |A|, the
      actions of the player that is being compared pairwise.
    players: a list of player names corresponding to the leading dimensions of
      the `loc`/`scale` tensor and ends with the player name of the player being
      compared pairwise.
    utilities: a sequence of utility functions for all players except the
      trailing player whose utility is the score difference following pairwise
      comparisons. Each utility function takes a (..., |A|, |A|) score
      difference tensor and returns a (..., |A|, |A|) tensor indicating the
      utility to player for each joint action.
    normalizer: a function that takes a (|A|) tensor and returns a (|A|) tensor
      indicating the single-sided normalized score for a choice of co-player
      actions. The normalized scores are then used to compare score differences
      side-by-side for the trailing player.

  Returns:
    A SxS score difference `Game` instance.
  """
  if scale is None:
    scale = jnp.zeros_like(loc)

  if len(actions) != len(players):
    raise ValueError("actions and players must have the same length.")

  if len(utilities) != len(players) - 1:
    raise ValueError("utilities must have length players - 1.")

  if jnp.any(scale < 0):
    raise ValueError("stddev must be non-negative.")

  chex.assert_shape((loc, scale), tuple(len(a) for a in actions))

  dist = distrax.Normal(loc=loc, scale=scale)

  for _ in range(len(players) - 1):
    normalizer = jax.vmap(normalizer)
  normalized = normalizer(loc)

  def _payoffs(scores: chex.Array) -> chex.Array:
    diff = scores[..., jnp.newaxis] - scores[..., jnp.newaxis, :]
    return jnp.stack([u(diff) for u in utilities] + [diff, -diff])

  def _sample(key: chex.PRNGKey) -> chex.Array:
    scores = dist.sample(seed=key)
    return _payoffs(normalizer(scores))

  return base.Game(
      payoffs=_payoffs(normalized),
      sample=_sample,
      actions=(*actions, actions[-1]),
      players=(*players, players[-1]),
      symmetry_groups=(
          tuple(i for i in range(len(players))) + (len(players) - 1,)
      ),
  )


def winrate_game(
    *,
    loc: chex.Array,  # [..., |A|]
    scale: chex.Array,  # [..., |A|]
    actions: Sequence[chex.Array],
    players: Sequence[str],
    utilities: Sequence[Callable[[chex.Array], chex.Array]],
    min_stddev: float = 1e-6,
) -> base.Game:
  """Returns a SxS winrate game given a normal distribution over scores.

  Args:
    loc: A (..., |A|) tensor where each element is the expected score (higher is
      better) for a choice of player actions.
    scale: A (..., |A|) tensor where each element is the stddev of the score for
      a choice of player actions.
    actions: A list of actions whose lengths correspond to the leading
      dimensions of `loc`/`scale` and ends with a vector of length |A|, the
      actions of the player that is being compared pairwise.
    players: A list of player names corresponding to the leading dimensions of
      the `loc`/`scale` tensor and ends with the player name of the player being
      compared pairwise.
    utilities: a sequence of utility functions for all players except the
      trailing player. Each utility function takes a (..., |A|, |A|) winrate
      tensor and returns a (..., |A|, |A|) tensor indicating the utility to that
      player for each joint action.
    min_stddev: minimum standard deviation to use for computing winrate.

  Returns:
    A SxS winrate Game instance.
  """
  num_players = len(players)

  if len(actions) != num_players:
    raise ValueError("actions and players must have the same length.")

  if len(utilities) != num_players - 1:
    raise ValueError("utilities must have length players - 1.")

  if jnp.any(scale < 0):
    raise ValueError("stddev must be non-negative.")

  chex.assert_shape((loc, scale), tuple(len(a) for a in actions))

  to_winrate = functools.partial(normalize.winrate, min_stddev=min_stddev)
  for _ in range(num_players - 1):
    to_winrate = jax.vmap(to_winrate)
  wr = to_winrate(loc, scale)

  # NOTE: Prefer transposing, rather than calculting `1-wr`, because the latter
  # can be numerically different, and result in inexact symmetry.
  wr_t = jnp.transpose(
      wr, axes=(*range(num_players - 1), num_players, num_players - 1)
  )

  payoffs = jnp.stack([u(wr) for u in utilities] + [wr, wr_t])

  return base.Game(
      payoffs=payoffs,
      sample=None,
      actions=tuple(actions) + (actions[-1],),
      players=tuple(players) + (players[-1],),
      symmetry_groups=(
          tuple(i for i in range(len(players))) + (len(players) - 1,)
      ),
  )
