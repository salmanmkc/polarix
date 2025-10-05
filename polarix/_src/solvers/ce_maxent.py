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

"""Maximum Entropy CE Solver."""

from collections.abc import Generator, Sequence
import functools
import string

import chex
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax
from polarix._src.games import base
from polarix._src.solvers import residual
from polarix._src.solvers import run


DEFAULT_OPTIM = optax.chain(optax.clip(1e2), optax.adam(learning_rate=1e-3))


def _maxent_ce_logit(
    ce_dual_per_player: Sequence[jax.Array],
    *,
    payoffs: jax.Array | None = None,
    max_ce_gain: float | None = None,
    target_logit: jax.Array | None = None,
) -> jax.Array:
  """Returns CE logit."""
  ce_dual_per_player = tuple(ce_dual_per_player)
  num_players = len(ce_dual_per_player)

  def _logit(player, dual, payoff):
    # Calculate logit.
    inds = string.ascii_lowercase[:num_players]
    pind = inds[player]
    inds_ = inds[:player] + "A" + inds[player + 1 :]
    dev_logit = jnp.einsum(f"A{pind},{inds_}->{inds}", dual, payoff)
    rec_logit = jnp.einsum(f"A{pind},{inds}->{inds}", dual, payoff)
    logit = rec_logit - dev_logit
    return logit

  ce_logit_per_player = jax.tree_util.tree_map(
      _logit, tuple(range(num_players)), ce_dual_per_player, tuple(payoffs)
  )
  logit = sum(ce_logit_per_player)

  if max_ce_gain is not None:
    logit += sum(
        jax.tree_util.tree_map(
            lambda dual: max_ce_gain * jnp.sum(dual), ce_dual_per_player
        )
    )

  if target_logit is not None:
    logit += target_logit

  return logit


def _project_ce_dual_per_player(
    ce_dual_per_player: Sequence[jax.Array],
) -> tuple[jax.Array, ...]:
  """Returns CE dual per player."""

  def _calc_ce_dual(ce_dual):
    num_strats = ce_dual.shape[0]
    ce_dual = jnp.maximum(ce_dual, 0.0)
    ce_dual = jnp.where(np.eye(num_strats, dtype=bool), 0, ce_dual)
    return ce_dual

  return tuple(jax.tree_util.tree_map(_calc_ce_dual, ce_dual_per_player))


def _per_group_to_per_player(
    per_group: tuple[chex.Array, ...], symmetry_groups: tuple[int, ...]
) -> tuple[chex.Array, ...]:
  """Returns per-player parameters from per-group parameters."""
  per_player = []
  visited = set()
  for g in symmetry_groups:
    if g not in visited:
      per_player.append(per_group[g])
      visited.add(g)
    else:
      per_player.append(jax.lax.stop_gradient(per_group[g]))
  return tuple(per_player)


def _make_update_fn(
    game: base.Game,
    max_ce_gain: float,
    target_logit: chex.Array | None,
    optim: optax.GradientTransformation,
    l1_weight_decay: float,
    tol: float,
):
  """Return update function for CE solver."""
  payoffs = jnp.asarray(game.payoffs)

  symmetry_groups = game.symmetry_groups
  if symmetry_groups is None:
    symmetry_groups = tuple(range(len(game.players)))

  num_strats_per_group = []
  visited = set()
  for g, ns in sorted(zip(symmetry_groups, payoffs.shape[1:])):
    if g not in visited:
      num_strats_per_group.append(ns)
      visited.add(g)
  num_strats_per_group = tuple(num_strats_per_group)

  def ce_loss_fn(
      ce_dual_per_group: tuple[chex.Array, ...],
  ) -> tuple[chex.Array, chex.Array]:
    ce_dual_per_player = _per_group_to_per_player(
        ce_dual_per_group, symmetry_groups=symmetry_groups
    )
    ce_logit = _maxent_ce_logit(
        ce_dual_per_player,
        payoffs=payoffs,
        max_ce_gain=max_ce_gain,
        target_logit=target_logit,
    )
    me_loss = jax.nn.logsumexp(ce_logit, axis=None)
    return me_loss, ce_logit

  @functools.partial(jax.jit, static_argnums=(2,))
  def update_fn(
      ce_dual_per_group: tuple[chex.Array, ...],
      opt_state: optax.OptState,
      inner_loop: bool | None = False,
  ):
    ce_dual_per_player = _per_group_to_per_player(
        ce_dual_per_group, symmetry_groups=symmetry_groups
    )
    ## Update.
    # Calculate the gradients with respect to an unregularized loss because it
    # is a useful quantity to reuse. Note:
    #   * ce_dual_grad_per_group == -expected_ce_gain_per_group.
    #   * expected_ce_gain_per_group is useful for calculating the ratings.
    # Also return the ce_logit, because it can be used to calculate the joint.
    (me_loss, ce_logit), ce_dual_grad_per_group = jax.value_and_grad(
        ce_loss_fn, has_aux=True
    )(ce_dual_per_group)
    # Gradients should be exactly zero for duals on the diagonal. In practice,
    # they may evaluate to be slightly nonzero because of numerical errors.
    ce_dual_grad_per_group = jax.tree_util.tree_map(
        lambda ns, g: jnp.where(jnp.eye(ns, dtype=bool), 0.0, g),
        num_strats_per_group,
        ce_dual_grad_per_group,
    )

    # Add the weight decay term, if necessary.
    if l1_weight_decay > 0.0:
      ce_dual_reg_grad_per_group = jax.tree_util.tree_map(
          # Dual is always nonnegative.
          lambda g: g + l1_weight_decay,
          ce_dual_grad_per_group,
      )
    else:
      ce_dual_reg_grad_per_group = ce_dual_grad_per_group

    # Update according to a supplied optax optimizer.
    update_per_group, opt_state = optim.update(
        ce_dual_reg_grad_per_group, opt_state, params=ce_dual_per_group
    )
    ce_dual_per_group = optax.apply_updates(ce_dual_per_group, update_per_group)
    # Project duals onto the nonnegative orthant, and ensure diagonal is zero.
    ce_dual_per_group = _project_ce_dual_per_player(ce_dual_per_group)

    ## Evaluation.
    # Everything onwards is only calculated on evaluation steps and is not
    # required to make updates.
    # Calculate the residual gradient. If param is at boundary, and gradient
    # is pointing into boundary, then zero out the gradient.
    ce_dual_residual_per_group = residual.residual_non_negative(
        ce_dual_per_group, ce_dual_reg_grad_per_group
    )
    # The norm of the projected gradient is zero at the optimum so is useful
    # for determinining termination conditions.
    residual_norm = sum(
        jax.tree_util.tree_map(
            jax.numpy.linalg.vector_norm, ce_dual_residual_per_group
        )
    )

    # Manually calculate the losses with the weight decay term.
    wd_loss = sum(jnp.sum(ce_dual) for ce_dual in ce_dual_per_player)
    loss = me_loss + l1_weight_decay * wd_loss
    # Calculate the joint and entropy.
    joint = jax.nn.softmax(ce_logit, axis=None)
    entropy = jnp.sum(jsp.special.entr(joint))
    # The expected CE gain can be determined from the gradients themselves.
    if max_ce_gain != 0.0:
      expected_ce_gain_per_group = jax.tree_util.tree_map(
          lambda ns, g: jnp.where(jnp.eye(ns, dtype=bool), 0, max_ce_gain) - g,
          num_strats_per_group,
          ce_dual_grad_per_group,
      )
    else:
      expected_ce_gain_per_group = jax.tree_util.tree_map(
          lambda g: -g,
          ce_dual_grad_per_group,
      )
    expected_ce_gain_per_player = _per_group_to_per_player(
        expected_ce_gain_per_group, symmetry_groups=symmetry_groups
    )
    # The ratings are functions of the expected CE gains.
    rating_per_player = jax.tree_util.tree_map(
        lambda g: jnp.sum(g, axis=1), expected_ce_gain_per_player
    )
    # The gaps are also functions of the expected CE gains.
    ce_gap_per_player = jax.tree_util.tree_map(
        lambda g: jnp.max(jnp.clip(g, 0, None)), expected_ce_gain_per_player
    )
    ce_gap = jnp.max(jnp.stack(list(ce_gap_per_player)))

    # Terminal.
    terminal = jnp.logical_and(
        jnp.abs(ce_gap - max_ce_gain) < tol, residual_norm < tol
    )

    # Auxiliary.
    extra = dict(
        entropy=entropy,
        me_loss=me_loss,
        wd_loss=wd_loss,
        loss=loss,
        residual_norm=residual_norm,
        ce_gap=ce_gap,
        ce_dual_per_player=ce_dual_per_player,
        expected_ce_gain_per_player=expected_ce_gain_per_player,
        rating_per_player=rating_per_player,
        joint=joint,
    )

    if inner_loop:
      return ce_dual_per_group, opt_state, dict()
    else:
      return (
          ce_dual_per_group,
          opt_state,
          dict(
              ratings=rating_per_player,
              joint=joint,
              extra=extra,
              terminal=terminal,
          ),
      )

  # Initialize the duals to zero, one per symmetry group.
  init = jax.tree_util.tree_map(
      lambda ns: jnp.zeros([ns, ns], dtype=float), num_strats_per_group
  )

  return init, update_fn


def ce_maxent(
    game: base.Game,
    max_ce_gain: float = 0.0,
    target_logit: chex.Array | None = None,
    optim: optax.GradientTransformation = DEFAULT_OPTIM,
    l1_weight_decay: float = 1e-6,
    tol: float = 1e-5,
) -> Generator[run.JointRatings | None, bool | None, None]:
  """Yields an approximate MECE for payoffs.

  Args:
    game: The game to solve containing the payoffs with shape
      [N,|S_1|,...,|S_N|].
    max_ce_gain: Maximum CE gain. Default 0.
    target_logit: Optional target log joint array with shape [|S_1|,...,|S_N|].
    optim: Optax optimizer to use.
    l1_weight_decay: Weight decay for the dual parameters.
    tol: Tolerance for termination condition.

  Yields:
    A `run.JointRatings` at each iteration.
  """
  ce_dual_per_group, update_fn = _make_update_fn(
      game, max_ce_gain, target_logit, optim, l1_weight_decay, tol
  )
  opt_state = optim.init(ce_dual_per_group)
  inner_loop = False
  while True:
    ce_dual_per_group, opt_state, ratings = update_fn(
        ce_dual_per_group, opt_state, inner_loop
    )
    if inner_loop:
      inner_loop = yield None
    else:
      inner_loop = yield run.JointRatings(**ratings)
