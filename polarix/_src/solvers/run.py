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

"""Run types and helper functions for rating solvers."""

from collections.abc import Callable, Generator, Mapping, Sequence
import dataclasses
import logging
from typing import TypeVar
import chex
import jax
import jax.numpy as jnp
import numpy as np
from polarix._src.games import base
from tqdm import auto as tqdm


@dataclasses.dataclass(kw_only=True, frozen=True)
class Ratings:
  """Ratings for players' actions in a game.

  Attributes:
    ratings: Ratings for all players' actions.
    terminal: A boolean scalar indicating whether the ratings have converged.
    extra: Additional information returned by the solver.
  """

  ratings: Sequence[chex.Array]
  terminal: chex.Array | None = None

  extra: Mapping[str, chex.ArrayTree] | None = None

  def __post_init__(self):
    if self.terminal is not None:
      chex.assert_equal_shape_prefix(
          tuple(self.ratings) + (self.terminal,), self.ratings[0].ndim - 1
      )
    else:
      chex.assert_equal_shape_prefix(self.ratings, self.ratings[0].ndim - 1)

  @property
  def summary(self) -> dict[str, chex.Array]:
    """Returns a mapping of values for display purposes."""
    summary = {}

    if self.extra is None:
      return summary

    for path, v in jax.tree.leaves_with_path(self.extra):
      if v.size in {1, len(self.ratings)}:
        summary[jax.tree_util.keystr(path)] = v
    return summary

  def is_terminal(self) -> bool:
    terminal = self.terminal
    if terminal is None:
      return False
    return np.min(terminal).astype(bool)


@dataclasses.dataclass(kw_only=True, frozen=True)
class MarginalRatings(Ratings):
  """Ratings for players' actions in a game at a marginal strategy profile."""

  marginals: Sequence[chex.Array]

  def __post_init__(self):
    super(MarginalRatings, self).__post_init__()
    chex.assert_trees_all_equal_sizes(self.ratings, self.marginals)

  @property
  def log_marginals(self) -> Sequence[chex.Array]:
    return [jnp.log(m) for m in self.marginals]


@dataclasses.dataclass(kw_only=True, frozen=True)
class JointRatings(Ratings):
  """Ratings for players' actions in a game at a joint strategy profile."""

  joint: chex.Array

  def __post_init__(self):
    super(JointRatings, self).__post_init__()
    chex.assert_shape(
        self.joint,
        list(self.ratings[0].shape[:-1]) + [r.shape[-1] for r in self.ratings],
    )

  @property
  def log_joint(self) -> chex.Array:
    return jnp.log(self.joint)


R = TypeVar("R", bound=Ratings)


def solve(
    game: base.Game,
    solver: Callable[[base.Game], Generator[R, bool | None, R | None]],
    *,
    num_iterations_per_update: int = 1_000,
    max_num_iterations: int = 100_000,
    disable_progress_bar: bool = False,
) -> R:
  """Solves a game using the an iterative solver.

  Args:
    game: The game to solve.
    solver: A callable that takes the game and returns a generator of results.
      For fast inner-loop iterations (where the run-loop does not require
      summary statistics to be updated), a boolean `True` is sent to the
      generator so that the generator can avoid computing evaluation statistics
      unnecessarily.
    num_iterations_per_update: The number of iterations to run before updating
      the progress bar.
    max_num_iterations: The maximum number of iterations to run.
    disable_progress_bar: If `True`, disable `tqdm` visualisation and log the
      terminal rating summary statistics to `logging.INFO`.

  Returns:
    The terminal `Ratings`.

  Raises:
    ValueError: If max_num_iterations is negative.
    RuntimeError: If the solver did not iterate until completion.
  """

  if max_num_iterations < 0:
    raise ValueError("max_num_iterations must be non-negative.")

  if num_iterations_per_update > max_num_iterations:
    logging.warning(
        "Overriding `num_iterations_per_update` (%d) to "
        "`max_num_iterations` (%d) which is lower.",
        num_iterations_per_update,
        max_num_iterations,
    )
    num_iterations_per_update = max_num_iterations

  with tqdm.tqdm(
      desc=f"Solving {str(game)}",
      total=max_num_iterations,
      disable=disable_progress_bar,
  ) as bar:
    solver_generator = solver(game)
    for t in range(max_num_iterations - 1):
      # Update progress bar with summary stats.
      if t % num_iterations_per_update == 0:
        ratings = next(solver_generator)
        bar.update(num_iterations_per_update)
        bar.set_postfix(ratings.summary)
        # Early terminate if results are terminal.
        if ratings.is_terminal():
          break
      else:
        _ = solver_generator.send(True)
    else:
      # Did not early terminate, evaluate final ratings to be returned.
      ratings = next(solver_generator)
      bar.update(max_num_iterations % num_iterations_per_update)
      bar.set_postfix(ratings.summary)
    if disable_progress_bar:
      logging.info(ratings.summary)
    return ratings
