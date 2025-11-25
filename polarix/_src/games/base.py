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

"""Game representation of evaluation data."""

import collections
from collections.abc import Callable, Sequence
import dataclasses
import functools
import itertools
import string
import chex
import jax
import jax.numpy as jnp


Logits = Sequence[chex.Array]
Masks = Sequence[chex.Array]


def marginals_from_joint(joint: chex.Array) -> Sequence[chex.Array]:
  """Returns the marginals of a joint strategy profile."""
  players = set(range(joint.ndim))
  marginals = []
  for p in range(joint.ndim):
    marginals.append(jnp.sum(joint, axis=tuple(players - {p})))
  return marginals


def joint_from_marginals(marginals: Sequence[chex.Array]) -> chex.Array:
  """Returns the joint strategy profile from a sequence of marginals."""
  chex.assert_rank(marginals, 1)
  players = set(range(len(marginals)))
  log_joint = jnp.zeros(tuple(len(x_p) for x_p in marginals))
  for p, x_p in enumerate(marginals):
    log_joint += jnp.expand_dims(jnp.log(x_p), axis=tuple(players - {p}))
  return jnp.exp(log_joint)


def marginal_payoffs(
    player: int,
    payoffs: chex.Array,
    logits: Logits,
    masks: Masks | None = None,
) -> chex.Array:
  """Gradient of player i's utility w.r.t. their mixed strategy (dist).

  Args:
    player: int, player index
    payoffs: npl x na_1 x ... na_npl payoff tensor
    logits: list of jnp arrays with each array being a mixed strategy
    masks: list of jnp arrays with each being a mask for that player's actions

  Returns:
    len-na_i jnp array, player i's gradient
  """
  if masks is None:
    masks = [jnp.ones(nai, dtype=bool) for nai in payoffs.shape[1:]]
  assert len(logits) == len(masks) == payoffs.shape[0]
  payoff = jnp.moveaxis(payoffs[player], player, 0)
  # NOTE: `jnp.einsum`` was slower than looping with jnp.dot.
  for p, (logit_p, mask_p) in enumerate(zip(logits[::-1], masks[::-1])):
    if player == len(logits) - 1 - p:
      continue
    prob_p = jax.nn.softmax(logit_p, where=mask_p, axis=-1)
    payoff = jnp.dot(payoff, prob_p)
  return payoff


@functools.partial(jax.jit, static_argnums=(2, 3))
def joint_payoffs_contribution(
    payoffs: chex.Array,
    joint: chex.Array,
    rating_player: int,
    contrib_player: int,
) -> chex.Array:
  """Returns marginal contribution given a payoff tensor and joint distribution.

  Computes the marginal contribution of each of `contrib_player`'s actions to
  the expected payoffs of each of `rating_player`'s actions under the joint
  strategy.

  Args:
    payoffs: The payoffs tensor with shape [N, |S_1|,...,|S_N|].
    joint: The joint distribution with shape [|S_1|,...,|S_N|].
    rating_player: The player whose ratings are being considered.
    contrib_player: The player whose contribution is being considered.

  Returns:
    The marginal contribution of each of `contrib_player`'s actions to the
    rating of each of `rating_player`'s actions. Has shape
    [|S_rating_player|, |S_contrib_player|].
  """
  assert rating_player != contrib_player
  payoff = payoffs[rating_player]
  num_players = payoff.ndim
  inds = string.ascii_lowercase[:num_players]
  contrib_ind = inds[contrib_player]
  dev_inds = inds[:rating_player] + "D" + inds[rating_player + 1 :]
  dev = jnp.einsum(f"{dev_inds},{inds}->D{contrib_ind}", payoff, joint)
  rec = jnp.einsum(f"{inds},{inds}->{contrib_ind}", payoff, joint)
  return dev - rec


@dataclasses.dataclass(kw_only=True, frozen=True)
class Game:
  """Represents a normal-form game."""

  payoffs: chex.Array
  actions: tuple[chex.Array, ...]
  players: tuple[str, ...]

  sample: Callable[[chex.Array], chex.Array] | None = None

  symmetry_groups: tuple[int, ...] | None = None

  def __str__(self):
    return f"Game({self.payoffs.shape})"

  def __post_init__(self):
    if self.payoffs.ndim != len(self.actions) + 1:
      raise ValueError("Payoffs must have one more dimension than actions.")
    for p, action in enumerate(self.actions):
      if len(action) != self.payoffs.shape[1 + p]:
        raise ValueError(
            f"Player {p} has {len(action)} actions, but payoff shape implies "
            f"{self.payoffs.shape[1 + p]} actions."
        )
    if not jnp.all(jnp.isfinite(self.payoffs)):
      not_finite = jnp.logical_not(
          jnp.all(
              jnp.isfinite(self.payoffs),
              axis=tuple(i for i in range(1, len(self.actions) + 1)),
          )
      )
      raise ValueError(
          "Payoffs must be finite but are not for players"
          f" {jnp.arange(len(self.actions))[not_finite]} (their"
          f" payoffs: {self.payoffs[not_finite]})"
      )
    if len(self.players) != len(self.actions):
      raise ValueError(
          "Players must have the same length as actions, but got"
          f" {len(self.players)} vs {len(self.actions)}"
      )
    if self.symmetry_groups is not None:
      if len(self.symmetry_groups) != len(self.actions):
        raise ValueError(
            "Symmetry groups must have the same length as actions, but got"
            f" {len(self.symmetry_groups)} vs {len(self.actions)}"
        )
      players_per_group = collections.defaultdict(list)
      for p, group in enumerate(self.symmetry_groups):
        players_per_group[group].append(p)
      for group, players in players_per_group.items():
        if len(players) == 1:
          continue
        for pa, pb in itertools.combinations(players, 2):
          if len(self.actions[pa]) != len(self.actions[pb]) or not jnp.all(
              self.actions[pa] == self.actions[pb]
          ):
            raise ValueError(
                "Actions must be the same between players in group"
                f" {group} ({players})"
            )
          axes = list(range(len(self.symmetry_groups)))
          axes[pa], axes[pb] = axes[pb], axes[pa]
          axes = tuple(axes)
          if not jnp.allclose(
              self.payoffs[pa],
              jnp.transpose(self.payoffs[pb], axes),
              atol=1e-6,
          ):
            diff = jnp.abs(
                self.payoffs[pa] - jnp.transpose(self.payoffs[pb], axes)
            )
            raise ValueError(
                "Payoffs are not symmetric between players in "
                f"{group=} ({players=}) {diff.max()=}"
            )
