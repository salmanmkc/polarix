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

"""Solvers for computing average ratings."""

from collections.abc import Sequence

import chex
import jax.numpy as jnp
from polarix._src.games import base
from polarix._src.solvers import run


def average(
    game: base.Game,
    marginals: Sequence[chex.Array] | None = None,
) -> run.MarginalRatings:
  """Returns the (optionally weighted) average ratings for actions in a game.

  Args:
    game: the game to solve.
    marginals: marginal action distributions per player to use as weighting.
      uniform distributions are used if not specified.

  Returns:
    A `run.MarginalRatings` result.
  """
  if marginals is None:
    marginals = [jnp.full(len(a), 1.0 / len(a)) for a in game.actions]

  ratings = []
  for p in range(len(game.players)):
    logits = tuple(map(jnp.log, marginals))
    ratings_p = base.marginal_payoffs(p, game.payoffs, logits=logits)
    ratings.append(ratings_p)

  return run.MarginalRatings(
      ratings=tuple(ratings),
      marginals=tuple(marginals),
      terminal=jnp.asarray(True),
  )
